import os

class Config:
    # --- 📂 DİZİN YOLLARI ---
    # scripts/utils içinde olduğumuz için 2 üst klasöre çıkıp ana dizini buluruz
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    # CSV Dosya İsimleri (datasplit.py'den gelen isimler)
    TRAIN_CSV = "train_list.csv"
    VAL_CSV = "val_list.csv"
    TEST_CSV = "test_list.csv"

    # --- 🖼️ GÖRÜNTÜ PARAMETRELERİ ---
    IMAGE_SIZE = (224, 224)
    
    # ImageNet standart normalizasyon değerleri
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # --- ⚙️ EĞİTİM GENEL AYARLARI ---
    @staticmethod
    def get_batch_size():
        # Train.py içindeki akıllı parametreler bunu ezecek ama 
        # DataLoader'ın hata vermemesi için varsayılan olarak 32 bırakıyoruz.
        return 16

# Klasörlerin varlığını kontrol et, yoksa oluştur
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)