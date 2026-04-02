import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config import Config # Config'i içeri alıyoruz

# --- AYARLAR ---
# Resimlerin olduğu ham veri yolu: /Tez_Projesi/chest_xray
DATA_PATH = os.path.join(Config.RAW_DATA_DIR, "chest_xray")
# CSV'lerin kaydedileceği yer: /PneumoVision/data
OUTPUT_PATH = Config.DATA_DIR 

def get_patient_id(file_name):
    fn = file_name.lower()
    if "person" in fn:
        return fn.split('_')[0]
    if "normal2-im" in fn:
        parts = file_name.split('-')
        return "-".join(parts[:3])
    if "im-" in fn:
        parts = file_name.split('-')
        return "-".join(parts[:2])
    return file_name.split('_')[0]

def run_split():
    all_data = []
    # Klasör yapısını kontrol et
    sub_folders = ['train', 'test', 'val'] 
    categories = ['NORMAL', 'PNEUMONIA']

    print(f"📂 Veri Kaynağı: {DATA_PATH}")
    print(f"📁 CSV Çıktı Dizini: {OUTPUT_PATH}")

    for sub in sub_folders:
        for cat in categories:
            folder_path = os.path.join(DATA_PATH, sub, cat)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Klasör eksik (Atlanıyor): {sub}/{cat}")
                continue
            
            files = os.listdir(folder_path)
            for img in files:
                if img.lower().endswith(('.jpeg', '.jpg', '.png')):
                    patient_id = get_patient_id(img)
                    
                    # ÖNEMLİ: file_path 'chest_xray' ile başlamalı ki 
                    # dataloader RAW_DATA_DIR ile birleştirdiğinde resme ulaşabilsin.
                    all_data.append({
                        'patient_id': patient_id,
                        'file_path': os.path.join('chest_xray', sub, cat, img),
                        'label': 1 if cat == 'PNEUMONIA' else 0
                    })

    df = pd.DataFrame(all_data)
    if df.empty:
        print(f"❌ HATA: {DATA_PATH} içinde hiç resim bulunamadı! Lütfen Drive klasör yapını kontrol et.")
        return

    # Hasta bazlı split (Patient-wise Split)
    unique_patients = df['patient_id'].unique()
    train_ids, temp_ids = train_test_split(unique_patients, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    train_df = df[df['patient_id'].isin(train_ids)]
    val_df = df[df['patient_id'].isin(val_ids)]
    test_df = df[df['patient_id'].isin(test_ids)]
    
    # CSV'leri kaydet
    train_df.to_csv(os.path.join(OUTPUT_PATH, Config.TRAIN_CSV), index=False)
    val_df.to_csv(os.path.join(OUTPUT_PATH, Config.VAL_CSV), index=False)
    test_df.to_csv(os.path.join(OUTPUT_PATH, Config.TEST_CSV), index=False)
    
    print(f"✅ CSV'ler oluşturuldu: {len(train_df)} Train, {len(val_df)} Val, {len(test_df)} Test")

if __name__ == "__main__":
    run_split()
