import torch
import torch.nn as nn
import torch.optim as optim
import timm
import os
import pandas as pd
from tqdm import tqdm

# --- PNEUMOVISION MODÜLLERİ ---
from dataloader import get_data_loaders
from utils.config import Config

# ==========================================================
# BİLİMSEL KIYASLAMA AYARLARI (SABİTLENMİŞ)
# ==========================================================
# Mimarilerin ham yeteneğini ölçmek için tüm parametreler eşittir.
STANDARD_LR = 1e-4          # Her model için orta yol öğrenme hızı
STANDARD_WD = 1e-2          # Her model için standart regülarizasyon
STANDARD_EPOCHS = 12        # Herkese kendini kanıtlaması için eşit süre
STANDARD_BATCH_SIZE = 16    # Her modelin (Transformer dahil) sığacağı ortak hacim

AKTIF_KULLANICI = "AYSE"    # Sudenaz çalıştırırken "SUDENAZ" yapacak
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_CONFIG = {
    "SUDENAZ": ["convnextv2_base", "cvt_21"],
    "AYSE": ["tf_efficientnetv2_b0", "seresnext50_32x4d"]
}

# ==========================================================
# ADİL EĞİTİM MOTORU
# ==========================================================
def run_fair_training_pipeline(model_name, train_loader, val_loader):
    print(f"\n" + "═"*60)
    print(f"🧬 MİMARİ ANALİZ: {model_name}")
    print(f"⚖️ STANDART ŞARTLAR: LR={STANDARD_LR}, WD={STANDARD_WD}, Epoch={STANDARD_EPOCHS}")
    print(f"💻 CİHAZ: {DEVICE}")
    print("═"*60)

    # 1. Model Oluşturma (Pretrained ağırlıklar aynı kaynaktan)
    model = timm.create_model(model_name, pretrained=True, num_classes=2).to(DEVICE)
    
    # 2. Optimizer & Kayıp Fonksiyonu (Birebir aynı)
    optimizer = optim.AdamW(model.parameters(), lr=STANDARD_LR, weight_decay=STANDARD_WD)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Sabırlı Zamanlayıcı (Eşit tolerans)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0

    for epoch in range(STANDARD_EPOCHS):
        # --- EĞİTİM ---
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{STANDARD_EPOCHS}")
        
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # --- DOĞRULAMA ---
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"📊 SONUÇ: Val Loss: {avg_val_loss:.4f} | Val Acc: %{val_acc:.2f}")
        
        scheduler.step(avg_val_loss)

        # En iyi skoru koru
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(Config.MODEL_SAVE_DIR, f"best_{model_name}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'acc': val_acc,
            }, save_path)
            print(f"⭐ Yeni rekor! %{val_acc:.2f} kaydedildi.")

    return best_val_acc

# ==========================================================
# ÇALIŞTIRICI
# ==========================================================
if __name__ == "__main__":
    # Veri yükleyicileri standart batch size ile al
    print("📦 Veri setleri adil kıyaslama için hazırlanıyor...")
    train_loader, val_loader, _ = get_data_loaders(batch_size=STANDARD_BATCH_SIZE)

    modellerim = MODELS_CONFIG[AKTIF_KULLANICI]
    scores = {}

    for m_name in modellerim:
        final_acc = run_fair_training_pipeline(m_name, train_loader, val_loader)
        scores[m_name] = final_acc

    # --- FİNAL TABLOSU ---
    print("\n" + "🏆" + " KIYASLAMA ÖZETİ ".center(50, "═") + "🏆")
    print(f"{'Model Adı':<30} | {'En İyi Başarı (%)':<15}")
    print("-" * 50)
    for model, score in scores.items():
        print(f"{model:<30} | %{score:.2f}")
    print("═" * 50)
    print(f"🎉 {AKTIF_KULLANICI}, tüm modellerin eşit şartlarda yarıştı!")