🫁 PneumoVision: Explainable & Robust Diagnostic System
PneumoVision, medikal görüntüleme dünyasında derin öğrenmenin "kara kutu" (black-box) sorununa çözüm getiren, güvenilir ve açıklanabilir bir zatürre teşhis platformudur.

🎯 Temel Misyonumuz
Sıradan bir sınıflandırıcının ötesine geçerek, yapay zekanın tıbbi kararlarını anlamlı ve dayanıklı kılmak.

Explainability (XAI): "Neden bu teşhis?" sorusuna Grad-CAM ile görsel kanıt sunar.

Robustness (Dayanıklılık): "Düşük kaliteli çekimde sonuç ne?" sorusuna stres testleriyle yanıt verir.

🛠️ Teknik Ekosistem
Proje, maksimum modülerlik ve hatasız iş birliği için şu mimari üzerine kurulmuştur:

Plaintext
PneumoVision/
├── scripts/
│   ├── train.py          # Merkezi Eğitim Motoru (Smart-Params)
│   ├── dataloader.py      # Weighted Sampling & Augmentation
│   └── utils/config.py    # Global Hiperparametre Yönetimi
├── notebooks/
│   └── EDA.ipynb         # İstatistiksel Veri Analizi & Outlier Temizliği
├── data/                 # No-Leakage Splitlenmiş Dataset
├── models/               # State-of-the-Art (SOTA) Ağırlıklar (.pth)
└── outputs/              # XAI Isı Haritaları & Robustness Raporları
🧠 Stratejik İş Bölümü (Hybrid Approach)
Modellerimizi mimari karakterlerine göre iki ana kolda optimize ediyoruz:

📈 Geliştirme Protokolü
Smart-Param Logic: Her modelin doğasına göre (CNN vs. Transformer) farklı öğrenme hızları (Learning Rate) uygulanır.

Imbalance Solution: Veri kümesindeki sınıf dengesizliği WeightedRandomSampler ile yazılımsal olarak çözülmüştür.

No-Leakage Policy: Hasta bazlı bölme (Patient ID splitting) yapılarak modellerin ezberlemesi (overfitting) engellenmiştir.

🚀 Çalıştırma Talimatı
Sistemi ayağa kaldırmak için terminalde ana dizindeyken şu komutu yürütmeniz yeterlidir:

# Aktif kullanıcıyı scripts/train.py üzerinden seçmeyi unutmayın!
python scripts/train.py