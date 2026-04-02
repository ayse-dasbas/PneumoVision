import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from utils.config import Config

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): CSV dosyasının tam yolu.
            root_dir (string): Resimlerin aranacağı ana dizin (RAW_DATA_DIR).
        """
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Path düzeltmesi: CSV içindeki 'chest_xray/...' yolunu RAW_DATA_DIR ile birleştiriyoruz
        img_relative_path = self.data_info.iloc[idx]['file_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Hata: {img_path} yüklenemedi! Hata: {e}")
            image = Image.new('RGB', Config.IMAGE_SIZE)
            
        label = int(self.data_info.iloc[idx]['label']) 

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(batch_size=None):
    if batch_size is None:
        batch_size = Config.get_batch_size()
    
    # CSV'lerin olduğu yer
    csv_dir = Config.DATA_DIR
    # Resimlerin ana kök dizini
    image_root = Config.RAW_DATA_DIR
    
    # Transformlar
    train_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    # Datasetleri Oluştur (CSV'yi csv_dir'den, resimleri image_root'tan al)
    train_ds = PneumoniaDataset(os.path.join(csv_dir, Config.TRAIN_CSV), image_root, transform=train_transforms)
    val_ds = PneumoniaDataset(os.path.join(csv_dir, Config.VAL_CSV), image_root, transform=val_test_transforms)
    test_ds = PneumoniaDataset(os.path.join(csv_dir, Config.TEST_CSV), image_root, transform=val_test_transforms)
    
    # Sampler (Dengesiz veri seti için)
    labels = train_ds.data_info['label'].values
    class_counts = pd.Series(labels).value_counts().sort_index().values
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Loaderlar
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
