# CAPTCHA OCR — ResNet-18 Trained from Scratch

**Live Demo:** [https://captcha-ocr-dnfbqbh4vwbr3fimr6vzda.streamlit.app/](https://captcha-ocr-dnfbqbh4vwbr3fimr6vzda.streamlit.app/)

A production-grade CAPTCHA recognition system powered by a ResNet-18 architecture trained entirely from scratch — no pretrained weights, no transfer learning, no external data.

---

## 🎯 Performance

| Metric              | Score     |
|----------------------|-----------|
| Character Accuracy   | **99.94%** |
| Sequence Accuracy    | **99.70%** |
| Character Error Rate | **0.06%**  |

---

## ✨ Features

- **Multi-Image Upload** — Process up to 5 CAPTCHA images simultaneously
- **Real-Time Inference** — Per-image predictions with millisecond timing
- **Confidence Scores** — Softmax-based average confidence per prediction
- **CSV Export** — Download all results in a single click
- **Dark-Mode Dashboard** — Premium AI SaaS visual design
- **Responsive Cards** — Individual result cards with status badges

---

## 🏗️ Architecture

```
Input (1×100×200 grayscale)
    │
    ├── ResNet-18 Backbone (weights=None, random init)
    │       └── Conv1 modified: 3ch → 1ch grayscale
    │
    ├── AdaptiveAvgPool2d → (1, 6) spatial slots
    │
    └── Linear(512, 31) classifier per position
    
Output: (batch, 6, 31) → 6-character sequence
```

- **Vocabulary**: 31 characters — digits `2-9`, uppercase letters `A-Z` (excluding `I`, `L`, `O`)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss with 0.1 label smoothing
- **Scheduler**: ReduceLROnPlateau
- **Epochs**: 40

---

## 📁 Project Structure

```
cig/
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── final_resnet18_captcha.pth    # Trained model weights
├── 01_Preprocessing.ipynb        # Training & EDA notebook
├── 02_predict.ipynb              # Inference notebook
└── submission_*.csv              # Competition submission
```

---

## 🚀 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/captcha-ocr.git
cd captcha-ocr

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place model weights
# Ensure final_resnet18_captcha.pth is in the project root

# 5. Run the application
streamlit run app.py
```

---

## ☁️ Streamlit Community Cloud Deployment

1. Push the repository to GitHub (include `app.py`, `requirements.txt`, and `final_resnet18_captcha.pth`).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

> **Note**: The model file (`~44 MB`) must be committed to the repository or hosted via Git LFS for cloud deployment.

---

## 📸 Screenshots

| Hero & Dashboard | Prediction Results |
|:---:|:---:|
| <img src="screenshots/image%202.png" width="400" alt="Hero Section and KPIs"> | <img src="screenshots/image%201.png" width="400" alt="Prediction Results Grid"> |

---

## 🛠️ Tech Stack

- **Framework**: PyTorch
- **Architecture**: ResNet-18 (random initialization)
- **Frontend**: Streamlit with custom CSS
- **Preprocessing**: OpenCV (grayscale, float32 normalization)

---

## 📄 License

This project is for educational and portfolio demonstration purposes.
