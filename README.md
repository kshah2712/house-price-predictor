# 🏠 House Price Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A production-style regression pipeline that predicts California house prices using the 1990 Census data. Covers advanced feature engineering, multiple regression models, and a beautiful interactive Streamlit web app — all containerized with Docker.

> **Purpose:** Demonstrate regression fundamentals — feature engineering, model comparison, evaluation metrics (RMSE, MAE, R²), and building an interactive UI with Streamlit instead of a raw REST API.

---

## 📌 What This Project Covers

| Concept | Implementation |
|---|---|
| Regression | Predicting continuous house prices |
| Feature Engineering | Creating 3 new meaningful features from existing ones |
| Model Training | Linear Regression, Random Forest, XGBoost |
| Evaluation Metrics | RMSE, MAE, R² Score |
| Model Persistence | joblib serialization |
| Interactive UI | Streamlit web app with sliders and inputs |
| Containerization | Dockerfile + docker-compose |
| Testing | pytest |

---

## 🗂️ Project Structure

```
house-price-predictor/
├── data/
│   ├── raw/                  # Original CSV
│   └── processed/            # Cleaned data
├── notebooks/
│   └── 01_housing_eda.ipynb  # EDA and analysis
├── src/
│   ├── download_data.py      # Download California housing dataset
│   ├── preprocess.py         # Feature engineering + preprocessing
│   └── train.py              # Model training + evaluation
├── models/                   # Saved .pkl model files
├── app/
│   └── streamlit_app.py      # Streamlit web application
├── tests/
│   └── test_preprocess.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/kshah2712/house-price-predictor.git
cd house-price-predictor
```

### 2. Create virtual environment

```bash
conda create -n house-price python=3.11 -y
conda activate house-price
pip install -r requirements.txt
```

### 3. Download dataset

```bash
python src/download_data.py
```

### 4. Train the models

```bash
python src/train.py
```

### 5. Run Streamlit app locally

```bash
streamlit run app/streamlit_app.py
```

### 6. Run with Docker

```bash
docker-compose up --build
```

---

## 🎯 Dataset — California Housing

Real census data from California, 1990. Each row = one block of houses.

| Feature | Description |
|---|---|
| `MedInc` | Median income in block |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per house |
| `AveBedrms` | Average bedrooms per house |
| `Population` | Block population |
| `AveOccup` | Average occupants per house |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |
| `MedHouseVal` | **TARGET** — Median house value ($100,000s) |

### Engineered Features
| Feature | Formula | Why |
|---|---|---|
| `RoomsPerPerson` | AveRooms / AveOccup | Measures spaciousness |
| `BedroomRatio` | AveBedrms / AveRooms | Measures layout quality |
| `IncomePerRoom` | MedInc / AveRooms | Combines income + space |

---

## 📊 Model Results

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Linear Regression | ~0.72 | ~0.53 | ~0.60 |
| Random Forest | ~0.50 | ~0.33 | ~0.80 |
| XGBoost | ~0.48 | ~0.31 | ~0.82 |

> Lower RMSE/MAE = better. Higher R² = better (max 1.0)

---

## 🖥️ Streamlit App

The web app allows users to:
- Adjust house features using **interactive sliders**
- Get **instant price predictions**
- See **feature importance** chart
- Understand which features affect price most

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML:** Scikit-learn, XGBoost
- **UI:** Streamlit
- **Serialization:** joblib
- **Containerization:** Docker, Docker Compose
- **Testing:** pytest

---

## 📚 Key Learnings

- Difference between **regression** and classification
- **Feature engineering** — creating meaningful features from raw data
- Regression evaluation metrics — **RMSE, MAE, R²**
- Building interactive ML apps with **Streamlit** (no frontend code!)
- Understanding **feature importance** in tree-based models

---

## 🗺️ Part of ML Learning Roadmap

This is **Project 2 of 10** in a progressive ML + GenAI portfolio:

| # | Project | Skills |
|---|---|---|
| ✅ 1 | Classic ML Pipeline | EDA, Sklearn, Flask, Docker |
| ✅ 2 | House Price Predictor (this project) | Regression, Feature Eng., Streamlit |
| 3 | Churn Classifier | XGBoost, SHAP, FastAPI |
| 4 | Image Classifier | PyTorch, CNN, MLflow |
| 5 | Sentiment Analyzer | HuggingFace, BERT, NLP |
| ... | ... | ... |

---

## 👤 Author

**Kashyap Shah**
[GitHub](https://github.com/kshah2712) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

