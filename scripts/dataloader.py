import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from utils.config import Config # Config entegrasyonu

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Config'den gelen yolları kullanıyoruz
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_relative_path = self.data_info.iloc[idx]['file_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Hata: {img_path} yüklenemedi! {e}")
            # Hatalı resim yerine boş bir tensor dönebilir veya pas geçebiliriz
            image = Image.new('RGB', Config.IMAGE_SIZE)
            
        label = int(self.data_info.iloc[idx]['label']) 

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(batch_size=None):
    # Eğer batch_size dışarıdan verilmediyse Config'dekini kullan
    if batch_size is None:
        batch_size = Config.get_batch_size()
    
    data_dir = Config.DATA_DIR
    
    # 1. Eğitim için Veri Artırma
    train_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    # Test/Val için standart dönüşüm
    val_test_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    # Datasetleri Config'deki CSV isimlerine göre oluştur
    train_ds = PneumoniaDataset(os.path.join(data_dir, Config.TRAIN_CSV), data_dir, transform=train_transforms)
    val_ds = PneumoniaDataset(os.path.join(data_dir, Config.VAL_CSV), data_dir, transform=val_test_transforms)
    test_ds = PneumoniaDataset(os.path.join(data_dir, Config.TEST_CSV), data_dir, transform=val_test_transforms)
    
    # 2. Weighted Sampler (Dengesiz Veri Çözümü)
    labels = train_ds.data_info['label'].values
    class_sample_count = pd.Series(labels).value_counts().sort_index().values
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weights = weights[labels]
    
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # DataLoader'lar
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader