import os

class Config:
    # --- 🤖 COLAB KONTROLÜ VE ANA DİZİN ---
    # Google Colab'da olup olmadığımızı kontrol ediyoruz
    IN_COLAB = os.path.exists('/content/drive')
    
    if IN_COLAB:
        # Colab'daysan Drive üzerindeki klasör yolunu buraya yaz (Klasör adını kontrol et!)
        BASE_DIR = "/content/drive/MyDrive/Tez_Projesi/PneumoVision" 
    else:
        # Yerel bilgisayardaysan mevcut hiyerarşiyi koru
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # --- 📂 DİZİN YOLLARI ---
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    # CSV Dosya İsimleri
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
        # Train.py içindeki adil kıyaslama parametresi bunu ezecek
        return 16

# --- 🛠️ OTOMATİK KLASÖR OLUŞTURMA ---
# Eğitim başlamadan önce klasörlerin var olduğundan emin oluyoruz
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Colab'da çalışırken klasörlerin doğru bağlandığını teyit etmek için:
if Config.IN_COLAB:
    print(f"✅ Google Colab modunda çalışıyor. Çıktılar Drive'a kaydedilecek: {Config.BASE_DIR}")
else:
    print(f"💻 Yerel modda çalışıyor. Ana dizin: {Config.BASE_DIR}")
