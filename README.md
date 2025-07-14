# 🌬️ BreatheWise - The Smart AQI Predictor

An interactive real-time Air Quality Index (AQI) prediction web app built using **Python, Streamlit, Optuna-tuned ML models, and OpenWeather API**.

---

## 📌 Features

* 🔢 Real-time weather & pollutant data fetching via OpenWeather API
* 🔢 ML-powered AQI prediction — both **category classification** and **index regression**
* 🌀 Live pollutant visualizations with **bar charts & AQI gauge**
* 🌐 Clean, styled UI with background image
* 🌟 Fully deployable on **Streamlit Cloud** or **Render**
* 📁 Project report and result documents available in `docs/`

---

## 📁 Repository Structure

```
BreatheWise/
│
├── 📄 README.md                         # Project overview and instructions
├── 📄 requirements.txt                  # Python dependencies
├── 📄 Dockerfile                        # Docker setup for Render deployment
├── 📄 render.yaml                       # Render deployment config
├── 📄 runtime.txt                       # Python version pinning for Streamlit
│
├── 📁 data/                             # Datasets
│   ├── balanced_station_day_realtime.csv
│   ├── cleaned_station_day_realtime.csv
│   └── station_day.csv
│
├── 📁 models/                           # Trained models and scalers
│   ├── final_stack_model_optimized.pkl
│   ├── final_stack_model_unbalanced.pkl
│   ├── final_stack_regression_optimized.pkl
│   ├── label_encoder_station.pkl
│   ├── label_encoder_unbalanced.pkl
│   ├── scaler_station.pkl
│   ├── scaler_unbalanced.pkl
│   └── scaler_regression.pkl
│
├── 📁 notebooks/                        # EDA plots & analysis
│   └── EDA_PLOTS/
│
├── 📁 scripts/                          # Python scripts for training and prediction
│   ├── app.py                           # Streamlit dashboard
│   ├── balance_aqicategory.py           # Balancing AQI categories
│   ├── classification.py                # Classification model training
│   ├── regression.py                    # Regression model training
│   ├── eda.py                           # EDA visualizations
│   ├── preprocessing.py                 # Preprocessing and feature engineering
│   ├── predictor_balanced.py            # Streamlit app for balanced model prediction
│   ├── predictor_unbalanced.py          # Streamlit app for unbalanced model prediction
│   ├── sizecalc.py                      # Utility: dataset size calculation
│   └── unbalanced_classi.py             # Unbalanced dataset classification script
│
├── 📁 logs/                             # CatBoost logs
│   └── catboost_info/
│
├── 📁 assets/                           # Static files & images
│   └── pexels-pixabay-266558.jpg
│
├── 📁 report/                             # Project report and result documents
│   ├── PROJECT REPORT ONLINE SUMMER TRAINING.pdf
│   └── FINAL RESULT.pdf
│
└── .gitignore                           # Ignore list for Git
```

---

## 📊 Models

* **Classification Model:**
  Stacking-based ensemble using **XGBoost**, **LightGBM**, and **CatBoost**, tuned with Optuna
  📊 Predicts AQI categories: *Good, Satisfactory, Moderate, Poor, Very Poor, Severe*

* **Regression Model:**
  Stacking-based regression ensemble predicting AQI numeric index values

---

## 📰 API Used

* 🌐 [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution)

---

## 📷 UI Design

* **Streamlit ECharts** for interactive visualizations
* Custom background image (`pexels-pixabay-266558.jpg`)
* Cleanly styled Streamlit UI

---

## 🚀 How to Run

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the app**

```bash
streamlit run app.py
```

> 📌 **Important:** Replace the placeholder `API_KEY` value in `app.py` with your actual OpenWeather API key before running the app.

---

## 📁 Project Documents

* 📄 [Final Project Report (PDF)](docs/FINAL%20PROJECT%20REPORT.pdf)
* 📄 [Final Result Screenshots (PDF)](docs/FINAL%20RESULT.pdf)

---

## 📌 Deployed at

* Local test: [http://localhost:8501/](http://localhost:8501/)

---

## ✨ Author

**Shreya**
📎 GitHub: [shreyaaaah](https://github.com/shreyaaaah)

---
