# 🦷 Teeth Classification — AI-Driven Dental Diagnostics

A computer vision project developed for **Cellula Technologies** that classifies dental images into 7 distinct categories using deep learning. The project includes two model approaches, a full preprocessing pipeline, and a deployed Streamlit web app.

---

## 📌 Project Overview

The goal is to build a robust dental image classification system to enhance diagnostic precision in AI-driven dental solutions. The system classifies images into 7 oral condition categories:

| Label | Condition |
|-------|-----------|
| `CaS` | Calculus |
| `CoS` | Caries |
| `Gum` | Gingivitis |
| `MC`  | Mouth Cancer |
| `OC`  | Oral Cancer |
| `OLP` | Oral Lichen Planus |
| `OT`  | Other |

---

## 🗂️ Repository Structure

```
├── Deployment/
│   ├── model/
│   │   ├── cnn_model.keras               # Saved CNN model
│   │   └── efficientnet_b4_model.keras   # Saved EfficientNetB4 model
│   ├── app.py                            # Streamlit inference app
│   ├── Dockerfile                        # Docker configuration
│   ├── requirements.txt                  # Python dependencies
│   ├── .dockerignore
│   └── .gitignore
│
├── paper/
│   ├── 3713-13972-1-PB.pdf               # Reference research paper
│   └── Paper Summary.txt                 # Summary of the paper
│
├── Teeth_Dataset/
│   ├── Training/                         # Training images (per class)
│   ├── Validation/                       # Validation images (per class)
│   └── Testing/                          # Test images (per class)
│
├── Training/
│   ├── Using_CNN_From_Scratch.ipynb      # Custom CNN training notebook
│   └── Using_EfficientNetB4.ipynb        # EfficientNetB4 training notebook
│
├── How to Run Docker.txt                 # Docker run instructions
├── README.md
├── Teeth Classification.pdf              # Project brief
└── Teeth Dataset.zip                     # Compressed dataset
```

---

## 🧪 Approach 1 — Custom CNN From Scratch

A lightweight CNN built entirely from scratch using TensorFlow/Keras, used to establish a performance baseline.

**Architecture:**
- Input: `256 × 256 × 3`
- 3 convolutional blocks (32 → 64 → 128 filters) with BatchNorm + MaxPooling
- Global Average Pooling → Dense softmax output
- Augmentation: Random flip, rotation (0.2), brightness (0.2), rescaling

**Training:**
- Optimizer: Adam (`lr=1e-4`)
- Loss: Categorical Crossentropy
- Metrics: Accuracy

---

## 🚀 Approach 2 — EfficientNetB4 (Pretrained Architecture)

Uses the EfficientNetB4 architecture (without ImageNet weights, trained from scratch on the dental dataset) for improved feature extraction capacity.

**Architecture:**
- Base: `EfficientNetB4` (no pretrained weights, `include_top=False`)
- Input: `256 × 256 × 3`
- Global Average Pooling → Dense softmax (7 classes)
- Same augmentation pipeline as CNN approach

**Training:**
- Optimizer: Adam (`lr=1e-4`)
- Loss: Categorical Crossentropy
- Metrics: Accuracy

---

## 📊 Pipeline

Both notebooks follow the same structured pipeline:

1. **Preprocessing** — Image normalization and augmentation (flip, rotation, brightness)
2. **Visualization** — Class distribution bar charts, sample image grids (before/after augmentation)
3. **Model Training** — Fit on train/val splits with history logging
4. **Evaluation** — Confusion matrix, per-class accuracy, classification report, misclassification visualization
5. **Export** — Model saved as `cnn_model.keras`

---

## 🖥️ Streamlit Web App

A simple inference UI built with Streamlit allowing users to upload a dental image and get a predicted class with confidence score.

**Run locally:**

```bash
pip install streamlit tensorflow pillow
streamlit run app.py
```

The app loads the saved model from `model/cnn_model.keras`, preprocesses the uploaded image using EfficientNet's preprocessing pipeline, and returns the predicted class and confidence.

---

## 🐳 Docker Deployment

The app is containerized using a `python:3.10-slim` base image and the Docker image is available on Docker Hub.

**Pull and run:**

```bash
docker pull <your-dockerhub-username>/teeth-classification-app
docker run -p 8501:8501 <your-dockerhub-username>/teeth-classification-app
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Build locally (from the `Deployment/` folder):**

```bash
cd Deployment
docker build -t teeth-classification-app .
docker run -p 8501:8501 teeth-classification-app
```

> **Note:** The `.dockerignore` excludes `.ipynb` notebooks, `__pycache__`, and virtual environments to keep the image lean. The model `.keras` files must be present inside `Deployment/model/` before building.

---

## 🛠️ Requirements

```
numpy==1.26.4
Pillow==10.4.0
streamlit==1.50.0
tensorflow==2.19.0
```

Install all dependencies (from the `Deployment/` folder):

```bash
pip install -r requirements.txt
```

> `scikit-learn` and `matplotlib` are used in the training notebooks only and are not required for running the app.

---

## 📁 Dataset

The dataset consists of dental images organized into 7 class folders, split into `train/`, `val/`, and `test/` directories. Update the `TRAIN_DIR`, `VAL_DIR`, and `TEST_DIR` paths in the notebooks to match your local setup.

---

## 🏢 About

Developed as part of **Week 1 Project** for [Cellula Technologies](https://www.linkedin.com/company/cellula-technologies) — an AI-driven dental solutions company focused on enhancing diagnostic precision and improving patient outcomes.
