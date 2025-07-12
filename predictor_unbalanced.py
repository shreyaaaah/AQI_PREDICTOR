import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 📌 Load models and encoders
reg_model = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/final_stack_regression_optimized.pkl')
clf_model = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/final_stack_model_unbalanced.pkl')
scaler_reg = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/scaler_regression.pkl')
scaler_clf = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/scaler_unbalanced.pkl')
le = joblib.load(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/label_encoder_unbalanced.pkl')

# 📌 API Key
api_key = '493a53955cbe6e63d28a621da8c6561b'

# 📌 Get city input
city = input("Enter city name: ")

# 📌 Fetch weather & pollution data from OpenWeather
weather_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
pollution_url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={{lat}}&lon={{lon}}&appid={api_key}'

weather_response = requests.get(weather_url).json()

if weather_response['cod'] != 200:
    print("❌ Invalid city or API error.")
    exit()

lat = weather_response['coord']['lat']
lon = weather_response['coord']['lon']

pollution_response = requests.get(pollution_url.format(lat=lat, lon=lon)).json()

if 'list' not in pollution_response or len(pollution_response['list']) == 0:
    print("❌ Pollution data unavailable.")
    exit()

# 📌 Extract full weather info
main_weather = weather_response['main']
wind = weather_response['wind']
sys = weather_response['sys']
clouds = weather_response['clouds']
weather_desc = weather_response['weather'][0]['description'].capitalize()

# 📌 Convert sunrise & sunset
sunrise_time = datetime.utcfromtimestamp(sys['sunrise']).strftime('%H:%M:%S') + " UTC"
sunset_time = datetime.utcfromtimestamp(sys['sunset']).strftime('%H:%M:%S') + " UTC"

# 📌 Extract pollutant values (µg/m³)
pollutants = pollution_response['list'][0]['components']

# 📌 Map pollutant values
input_data = {
    'PM2.5': pollutants.get('pm2_5', 0),
    'PM10': pollutants.get('pm10', 0),
    'NO': pollutants.get('no', 0),
    'NO2': pollutants.get('no2', 0),
    'NH3': pollutants.get('nh3', 0),
    'CO': pollutants.get('co', 0),
    'SO2': pollutants.get('so2', 0),
    'O3': pollutants.get('o3', 0)
}

# 📌 Create engineered features
input_data['PM2.5/PM10'] = input_data['PM2.5'] / (input_data['PM10'] + 1)
input_data['NO/NO2'] = input_data['NO'] / (input_data['NO2'] + 1)

# 📌 Convert to DataFrame for model
df_input = pd.DataFrame([input_data])

# 📌 Predict AQI Value (Regression)
scaled_reg_input = scaler_reg.transform(df_input)
predicted_aqi = reg_model.predict(scaled_reg_input)[0]

# 📌 Predict AQI Category (Classification)
scaled_clf_input = scaler_clf.transform(df_input)
predicted_category_encoded = clf_model.predict(scaled_clf_input)
predicted_category = le.inverse_transform(predicted_category_encoded)[0]

# 📌 Health Advisory Mapping (your custom messages)
advisory_messages = {
    'Good': "✅✨ Air quality is excellent. No restrictions — enjoy outdoor activities freely! 🏃‍♂️🌸",
    'Satisfactory': "🙂🌿 Air is acceptable. Sensitive individuals should limit prolonged outdoor exertion. 🚶‍♂️💨",
    'Moderate': "⚠️😷 Mild health concern. Sensitive groups should wear a mask outdoors. 🏞️🛡️",
    'Poor': "🚨😷 Everyone should reduce outdoor exertion. Use N95 masks. 🏠",
    'Very Poor': "❌🚫 Unhealthy air. Avoid going outside. Stay indoors. 🏠🔒",
    'Severe': "☠️⚠️ Dangerous air. Stay indoors. Seek medical help for symptoms. 🆘🏥"
}

# 📌 Display Results
print(f"\n🌤️ Real-Time Weather in {city.capitalize()}:")
print(f"Weather Description: {weather_desc}")
print(f"Temperature (°C): {main_weather['temp']}")
print(f"Feels Like (°C): {main_weather['feels_like']}")
print(f"Min Temperature (°C): {main_weather['temp_min']}")
print(f"Max Temperature (°C): {main_weather['temp_max']}")
print(f"Humidity (%): {main_weather['humidity']}")
print(f"Pressure (hPa): {main_weather['pressure']}")
print(f"Wind Speed (m/s): {wind['speed']}")
print(f"Wind Direction (°): {wind.get('deg', 'N/A')}")
print(f"Cloudiness (%): {clouds['all']}")
print(f"Visibility (m): {weather_response.get('visibility', 'N/A')}")
print(f"Sunrise: {sunrise_time}")
print(f"Sunset: {sunset_time}")

print(f"\n📊 Real-Time Pollutant Concentrations (µg/m³):")
for k, v in input_data.items():
    print(f"{k}: {v:.2f}")

print(f"\n Predicted AQI Value: {predicted_aqi:.2f}")
print(f" Predicted AQI Category: {predicted_category}")

# 📌 Show health advisory
print(f"\n💡 Health Advisory: {advisory_messages.get(predicted_category, 'No advisory available.')}")
