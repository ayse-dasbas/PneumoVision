import torch
import torch.nn as nn
import torch.optim as optim
import timm
import os
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score

# --- PNEUMOVISION MODÜLLERİ ---
from dataloader import get_data_loaders
from utils.config import Config

# ==========================================================
# BİLİMSEL KIYASLAMA AYARLARI (SABİTLENMİŞ)
# ==========================================================
STANDARD_LR = 1e-4
STANDARD_WD = 1e-2
STANDARD_EPOCHS = 12
STANDARD_BATCH_SIZE = 16

AKTIF_KULLANICI = "AYSE" # Sudenaz için "SUDENAZ"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_CONFIG = {
    "SUDENAZ": ["convnextv2_base", "cvt_21"],
    "AYSE": ["tf_efficientnetv2_b0", "seresnext50_32x4d"]
}

# ==========================================================
# ADİL EĞİTİM MOTORU (GELİŞMİŞ SÜRÜM)
# ==========================================================
def run_fair_training_pipeline(model_name, train_loader, val_loader):
    print(f"\n" + "═"*60)
    print(f"🧬 MİMARİ ANALİZ: {model_name}")
    print(f"⚖️ STANDART ŞARTLAR: LR={STANDARD_LR}, WD={STANDARD_WD}, Epoch={STANDARD_EPOCHS}")
    print(f"💻 CİHAZ: {DEVICE}")
    print("═"*60)

    # 1. Model Oluşturma
    model = timm.create_model(model_name, pretrained=True, num_classes=2).to(DEVICE)
    
    # 2. Optimizer & Kayıp Fonksiyonu
    optimizer = optim.AdamW(model.parameters(), lr=STANDARD_LR, weight_decay=STANDARD_WD)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    history = [] # Log kayıtları için liste

    for epoch in range(STANDARD_EPOCHS):
        epoch_start_time = time.time()
        
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

        # --- DOĞRULAMA & METRİK HESAPLAMA ---
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Metrikler için listeye ekle
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrik Hesaplamaları
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        recall = 100. * recall_score(all_labels, all_preds)
        f1 = 100. * f1_score(all_labels, all_preds)
        epoch_duration = time.time() - epoch_start_time

        print(f"📊 SONUÇ: Acc: %{val_acc:.2f} | Recall: %{recall:.2f} | F1: %{f1:.2f} | Süre: {epoch_duration:.1f}s")
        
        # Log Kaydı
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'val_recall': recall,
            'val_f1': f1,
            'duration': epoch_duration
        })

        scheduler.step(avg_val_loss)

        # Gelişmiş Checkpoint (En iyi skoru koru ve yedekle)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(Config.MODEL_SAVE_DIR, f"best_{model_name}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': val_acc,
                'recall': recall
            }, save_path)
            print(f"⭐ Yeni rekor! %{val_acc:.2f} Drive'a yedeklendi.")

    # Tüm epochlar bitince logları CSV'ye dök
    log_path = os.path.join(Config.OUTPUT_DIR, f"log_{model_name}.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"📝 Eğitim günlüğü kaydedildi: {log_path}")

    return best_val_acc

# ==========================================================
# ÇALIŞTIRICI
# ==========================================================
if __name__ == "__main__":
    print(f"📦 Veri setleri hazırlanıyor... (Kullanıcı: {AKTIF_KULLANICI})")
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
