
# 🌬️ BreatheWise - The Smart AQI Predictor

An interactive real-time Air Quality Index (AQI) predictor app built using Python, Streamlit, and OpenWeather API.

## 📌 Features
- Real-time weather and pollutant data fetching via OpenWeather API.
- ML-powered AQI value prediction (regression) and category classification.
- Live pollutant visualizations (bar charts and AQI gauge).
- Clean, styled UI with background images.
- Deploy-ready on Streamlit Cloud.

## 🚀 How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

## 📁 Repository Structure
```
├── app.py
├── final_stack_model_optimized.pkl
├── final_stack_regression_optimized.pkl
├── scaler_station.pkl
├── label_encoder_station.pkl
├── pexels-pixabay-266558.jpg
├── requirements.txt
├── ...
```

## 📊 Models
- Classification Model: Stacking-based classifier predicting AQI categories.
- Regression Model: Stacking-based regressor predicting AQI index values.

## 📡 API Used
- [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution)

## 📷 UI Design
- Streamlit ECharts for interactive plots.
- Custom background image for UI aesthetics.

## 📌 Deployed at:
[🔗 Streamlit Cloud link here after deploying]

---

## ✨ Author
**Shreya** (GitHub: [shreyaaaah](https://github.com/shreyaaaah))
