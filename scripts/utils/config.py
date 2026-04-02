import os

class Config:
    # --- 🤖 COLAB KONTROLÜ VE ANA DİZİN ---
    IN_COLAB = os.path.exists('/content/drive')
    
    if IN_COLAB:
        # Kodların ve CSV'lerin duracağı ana klasör
        BASE_DIR = "/content/drive/MyDrive/Tez_Projesi/PneumoVision"
        # Ham resimlerin (chest_xray klasörünün) bulunduğu üst dizin
        RAW_DATA_DIR = "/content/drive/MyDrive/Tez_Projesi"
    else:
        # Yerel bilgisayar ayarı
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        RAW_DATA_DIR = BASE_DIR

    # --- 📂 DİZİN YOLLARI ---
    # CSV dosyalarının kaydedileceği ve okunacağı yer: /PneumoVision/data
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Modellerin ve sonuçların kaydedileceği yerler
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
        return 16

# --- 🛠️ OTOMATİK KLASÖR OLUŞTURMA ---
# Eğitim başlamadan önce gerekli klasörlerin (data, models, outputs) var olduğundan emin oluyoruz
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Bilgilendirme Mesajı
if Config.IN_COLAB:
    print(f"✅ Google Colab modunda çalışıyor.")
    print(f"📂 Kod ve CSV Dizini: {Config.BASE_DIR}")
    print(f"🖼️ Ham Veri Dizini: {Config.RAW_DATA_DIR}")
else:
    print(f"💻 Yerel modda çalışıyor. Ana dizin: {Config.BASE_DIR}")
