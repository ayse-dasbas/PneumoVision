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
            csv_file (string): Resim yollarını ve etiketleri içeren CSV dosyası.
            root_dir (string): Resimlerin bulunduğu ana dizin (Config.DATA_DIR).
            transform (callable, optional): Görüntü üzerine uygulanacak transformlar.
        """
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Path düzeltmesi: Windows/Linux uyumu için \ -> / çevrimi
        img_relative_path = self.data_info.iloc[idx]['file_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Hata: {img_path} yüklenemedi, boş görsel dönülüyor! Hata: {e}")
            # Eğitim yarıda kalmasın diye siyah görsel dönüyoruz
            image = Image.new('RGB', Config.IMAGE_SIZE)
            
        label = int(self.data_info.iloc[idx]['label']) 

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(batch_size=None):
    """
    Eğitim, Doğrulama ve Test için DataLoader nesnelerini oluşturur.
    """
    if batch_size is None:
        batch_size = Config.get_batch_size()
    
    data_dir = Config.DATA_DIR
    
    # 1. Veri Artırma (Data Augmentation) - Sadece Eğitim İçin
    train_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Değerler biraz artırıldı
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    # Doğrulama ve Test İçin Standart Dönüşüm
    val_test_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN, Config.NORM_STD)
    ])

    # Datasetleri Oluştur
    train_ds = PneumoniaDataset(os.path.join(data_dir, Config.TRAIN_CSV), data_dir, transform=train_transforms)
    val_ds = PneumoniaDataset(os.path.join(data_dir, Config.VAL_CSV), data_dir, transform=val_test_transforms)
    test_ds = PneumoniaDataset(os.path.join(data_dir, Config.TEST_CSV), data_dir, transform=val_test_transforms)
    
    # 2. Weighted Sampler (Sınıf Dengesizliği Çözümü)
    # Eğitim setindeki etiket sayılarını hesaplayıp az olan sınıfa daha fazla şans tanıyoruz.
    labels = train_ds.data_info['label'].values
    class_counts = pd.Series(labels).value_counts().sort_index().values
    
    # Sınıf ağırlığı = 1 / Sınıf Sayısı
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    # 3. DataLoader'ların Oluşturulması
    # Colab ücretsiz GPU (T4) için num_workers=2 idealdir.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, 
        num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
