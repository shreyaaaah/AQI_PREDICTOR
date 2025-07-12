import requests
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 📌 Load classification model, scaler, label encoder
clf_model = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/final_stack_model_optimized.pkl')
scaler_clf = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/scaler_station.pkl')
le = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/label_encoder_station.pkl')

# 📌 Load regression model and scaler
reg_model = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/final_stack_regression_optimized.pkl')
scaler_reg = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/scaler_regression.pkl')

# 📌 API Key
api_key = '493a53955cbe6e63d28a621da8c6561b'

# 📌 Health advisories
advisories = {
    'Good': "✅✨ Air quality is excellent. No restrictions — enjoy outdoor activities freely! 🏃‍♂️🌸",
    'Satisfactory': "🙂🌿 Air is acceptable. Sensitive individuals should limit prolonged outdoor exertion. 🚶‍♂️💨",
    'Moderate': "⚠️😷 Mild health concern. Sensitive groups should wear a mask outdoors. 🏞️🛡️",
    'Poor': "🚨😷 Everyone should reduce outdoor exertion. Use N95 masks. 🏠",
    'Very Poor': "❌🚫 Unhealthy air. Avoid going outside. Stay indoors. 🏠🔒",
    'Severe': "☠️⚠️ Dangerous air. Stay indoors. Seek medical help for symptoms. 🆘🏥"
}

# 📌 Fetch city from user
city = input("Enter city name: ")

# 📌 Fetch city coordinates and weather
weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
weather_response = requests.get(weather_url).json()

try:
    # 📌 Extract coordinates
    lat = weather_response['coord']['lat']
    lon = weather_response['coord']['lon']

    # 📌 Weather details
    print(f"\n🌤️ Real-Time Weather in {city.capitalize()}:")
    print(f"Temperature (°C): {weather_response['main']['temp']}")
    print(f"Feels Like (°C): {weather_response['main']['feels_like']}")
    print(f"Humidity (%): {weather_response['main']['humidity']}")
    print(f"Pressure (hPa): {weather_response['main']['pressure']}")
    print(f"Visibility (m): {weather_response.get('visibility', 'N/A')}")
    print(f"Wind Speed (m/s): {weather_response['wind']['speed']}")
    print(f"Weather Condition: {weather_response['weather'][0]['description'].capitalize()}")

    # 📌 Fetch air pollution data using coordinates
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    pollution_response = requests.get(pollution_url).json()

    components = pollution_response['list'][0]['components']

    # 📌 Extract pollutants
    PM25 = components.get('pm2_5', 0)
    PM10 = components.get('pm10', 0)
    NO = components.get('no', 0)
    NO2 = components.get('no2', 0)
    NH3 = components.get('nh3', 0)
    CO = components.get('co', 0)
    SO2 = components.get('so2', 0)
    O3 = components.get('o3', 0)

    # 📌 Display real-time pollutants
    print("\n📊 Real-Time Pollutant Concentrations (μg/m³):")
    print(f"PM2.5: {PM25}")
    print(f"PM10 : {PM10}")
    print(f"NO   : {NO}")
    print(f"NO2  : {NO2}")
    print(f"NH3  : {NH3}")
    print(f"CO   : {CO}")
    print(f"SO2  : {SO2}")
    print(f"O3   : {O3}")

    # 📌 Derived features
    PM25_PM10_ratio = PM25 / (PM10 + 1)
    NO_NO2_ratio = NO / (NO2 + 1)

    # 📌 Display derived features
    print(f"\n📊 Derived Features:")
    print(f"PM2.5 / PM10 Ratio: {PM25_PM10_ratio:.4f}")
    print(f"NO / NO2 Ratio    : {NO_NO2_ratio:.4f}")

    # 📌 Final feature array
    input_features = np.array([[PM25, PM10, NO, NO2, NH3, CO, SO2, O3, PM25_PM10_ratio, NO_NO2_ratio]])

    # 📌 Scale features
    input_scaled_clf = scaler_clf.transform(input_features)
    input_scaled_reg = scaler_reg.transform(input_features)

    # 📌 Predict AQI category
    pred_category_encoded = clf_model.predict(input_scaled_clf)
    pred_category = le.inverse_transform(pred_category_encoded)[0]

    # 📌 Predict AQI value
    pred_aqi_value = reg_model.predict(input_scaled_reg)[0]

    # 📌 Display predictions
    print(f"\n📈 Predicted AQI Value: {pred_aqi_value:.2f}")
    print(f"📊 Predicted AQI Category: {pred_category}")
    print("💡 Health Advisory:", advisories[pred_category])

except Exception as e:
    print("❌ Error fetching data. Please check city name or API access.")
    print(e)
