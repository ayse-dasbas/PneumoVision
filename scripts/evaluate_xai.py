import os
import cv2
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. YARDIMCI FONKSİYONLAR ---
def get_input_tensors(img_path):
    """Görüntüyü yükler ve modelin beklediği tensör formatına çevirir."""
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(orig_img, (224, 224))
    rgb_img = np.float32(resized_img) / 255
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(resized_img).unsqueeze(0)
    return input_tensor, rgb_img

def main(existing_model): # Modeli dışarıdan alıyoruz
    # --- 2. AYARLAR VE PARAMETRELER ---
    SUTUN_ADI_YOL = 'file_path' 
    SUTUN_ADI_ETIKET = 'label'
    
    # Yolları Colab yapına göre netleştirelim
    BASE_DATA_PATH = '/content/drive/MyDrive/Tez_Projesi/' # CSV'deki file_path ile birleşecek
    CSV_PATH = '/content/drive/MyDrive/Tez_Projesi/PneumoVision/data/test_list.csv'
    OUTPUT_DIR = '/content/drive/MyDrive/Tez_Projesi/XAI_Results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 3. MODEL VE GRAD-CAM KURULUMU ---
    # Fonksiyon içindeki 'model' ismi artık dışarıdaki modelini temsil eder
    existing_model.eval()
    if torch.cuda.is_available():
        existing_model = existing_model.cuda()

    target_layers = [existing_model.stages[-1].blocks[-1]]
    cam = GradCAM(model=existing_model, target_layers=target_layers)

    # --- 4. VERİ SETİ VE ANALİZ DÖNGÜSÜ ---
    test_df = pd.read_csv(CSV_PATH)
    tp_samples = test_df[test_df[SUTUN_ADI_ETIKET] == 1].head(5)
    tn_samples = test_df[test_df[SUTUN_ADI_ETIKET] == 0].head(5)
    
    samples_to_process = [("TP", tp_samples), ("TN", tn_samples)]

    for category, data in samples_to_process:
        for idx, row in data.iterrows():
            # CSV'deki file_path chest_xray/... diye başlıyorsa BASE_DATA_PATH ile tam birleşir
            img_path = os.path.join(BASE_DATA_PATH, row[SUTUN_ADI_YOL])
            
            if not os.path.exists(img_path):
                print(f"Uyarı: Dosya bulunamadı {img_path}")
                continue

            input_tensor, rgb_img = get_input_tensors(img_path)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            with torch.no_grad():
                output = existing_model(input_tensor)
                pred_idx = output.argmax(dim=1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0][pred_idx].item()
            
            true_label = row[SUTUN_ADI_ETIKET]
            status = "TP" if (true_label == 1 and pred_idx == 1) else \
                     "TN" if (true_label == 0 and pred_idx == 0) else \
                     "FP" if (true_label == 0 and pred_idx == 1) else "FN"

            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            file_name = f"{status}_idx{idx}_Conf{confidence:.2f}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), 
                        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print(f"Analiz bitti. Klasör: {OUTPUT_DIR}")

# Çalıştırmak için:
if __name__ == "__main__":
    # Bellekteki modelini fonksiyona gönderiyorsun
    main(model)
