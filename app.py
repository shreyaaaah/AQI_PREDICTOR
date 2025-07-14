import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from streamlit_echarts import st_echarts
import base64

# Load models and scalers
clf_model = joblib.load('final_stack_model_optimized.pkl')
reg_model = joblib.load('final_stack_regression_optimized.pkl')
scaler = joblib.load('scaler_station.pkl')
label_encoder = joblib.load('label_encoder_station.pkl')

api_key = '493a53955cbe6e63d28a621da8c6561b'

# Set page config and background image
st.set_page_config(page_title='BreatheWise - The Smart AQI Predictor', page_icon='🌬️')

def set_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .result-box {{
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set your background image path here
set_bg_image("C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/pexels-pixabay-266558.jpg")

# Title
st.markdown("<h1 style='text-align: center; color: white;'>🌬️ BreatheWise - The Smart AQI Predictor</h1>", unsafe_allow_html=True)

# City input
city = st.text_input("Enter city name:")

if city:
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    geo_response = requests.get(geo_url).json()

    if geo_response:
        lat, lon = geo_response[0]['lat'], geo_response[0]['lon']

        # Weather API call
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_response = requests.get(weather_url).json()

        # Air Pollution API call
        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        pollution_response = requests.get(pollution_url).json()

        if 'list' in pollution_response:
            weather = weather_response['main']
            wind = weather_response['wind']
            visibility = weather_response.get('visibility', 0)
            condition = weather_response['weather'][0]['description'].title()

            # 📌 Weather details
            st.subheader("🌤️ Real-Time Weather Details")
            st.markdown(f"""
            - **Temperature:** {weather['temp']}°C  
            - **Feels Like:** {weather['feels_like']}°C  
            - **Humidity:** {weather['humidity']}%  
            - **Pressure:** {weather['pressure']} hPa  
            - **Visibility:** {visibility / 1000} km  
            - **Wind Speed:** {wind['speed']} m/s  
            - **Condition:** {condition}
            """)

            # 📌 Pollutant data
            components = pollution_response['list'][0]['components']
            pollutants = {
                'PM2.5': components.get('pm2_5', 0),
                'PM10': components.get('pm10', 0),
                'NO': components.get('no', 0),
                'NO2': components.get('no2', 0),
                'NH3': components.get('nh3', 0),
                'CO': components.get('co', 0),
                'SO2': components.get('so2', 0),
                'O3': components.get('o3', 0)
            }
            pollutants['PM2.5/PM10'] = pollutants['PM2.5'] / (pollutants['PM10'] + 1)
            pollutants['NO/NO2'] = pollutants['NO'] / (pollutants['NO2'] + 1)

            pollutant_df = pd.DataFrame(list(pollutants.items()), columns=['Pollutant', 'Concentration'])

            st.subheader("🧪 Real-Time Pollutant Levels (µg/m³)")
            st.table(pollutant_df)

            # 📌 Pollutant Graph
            st.subheader("📈 Pollutant Concentrations Graph")
            option_bar = {
                "tooltip": {"trigger": "axis"},
                "xAxis": {
                    "type": "category",
                    "data": pollutant_df['Pollutant'].tolist(),
                    "axisLabel": {"interval": 0, "rotate": 30, "color": "white", "fontSize": 14}
                },
                "yAxis": {
                    "type": "value",
                    "axisLabel": {"color": "white", "fontSize": 14}
                },
                "series": [{
                    "data": pollutant_df['Concentration'].tolist(),
                    "type": "bar",
                    "itemStyle": {"color": "#ffc300"}
                }],
                "grid": {"bottom": 80},
                "backgroundColor": "transparent"
            }
            st_echarts(options=option_bar, height="500px")

            # 📌 Prediction
            feature_values = np.array(list(pollutants.values())).reshape(1, -1)
            scaled_values = scaler.transform(feature_values)
            pred_category = clf_model.predict(scaled_values)
            pred_label = label_encoder.inverse_transform(pred_category)[0]

            # Custom category mapping
            category_map = {
                'Good': 'Good', 'Satisfactory': 'Good', 'Moderate': 'Satisfactory',
                'Poor': 'Moderate', 'Very Poor': 'Poor', 'Severe': 'Very Poor'
            }
            mapped_category = category_map.get(pred_label, pred_label)
            pred_aqi = reg_model.predict(scaled_values)[0]

            # Actual OpenWeather AQI category
            actual_code = pollution_response['list'][0]['main']['aqi']
            aqi_scale = {1: 'Good', 2: 'Satisfactory', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
            actual_category = aqi_scale.get(actual_code, 'Unknown')

            # Color mapping
            color_classes = {
                'Good': '#00e400', 'Satisfactory': '#a3c853', 'Moderate': '#ffc300',
                'Poor': '#ff7e00', 'Very Poor': '#ff0000'
            }

            # 📌 AQI Gauge
            st.subheader("📊 AQI Gauge Meter")
            option_gauge = {
                "series": [{
                    "type": 'gauge', "min": 0, "max": 500,
                    "splitNumber": 5, "axisLine": {"lineStyle": {"width": 20}},
                    "pointer": {"width": 5},
                    "detail": {"formatter": f'{{value}}'},
                    "data": [{"value": round(pred_aqi)}]
                }]
            }
            st_echarts(options=option_gauge, height="400px")

            # 📌 AQI Value
            st.markdown(f"""
            <div class='result-box' style='background-color: white; color: black;'>
            📊 Predicted AQI Value: {round(pred_aqi)}
            </div>
            """, unsafe_allow_html=True)

            # 📌 Predicted Category
            st.markdown(f"""
            <div class='result-box' style='background-color: {color_classes[mapped_category]};'>
            📊 Predicted AQI Category: {mapped_category}
            </div>
            """, unsafe_allow_html=True)

            # 📌 Actual Category
            st.markdown(f"""
            <div class='result-box' style='background-color: {color_classes[actual_category]};'>
            📊 OpenWeather Actual Category: {actual_category}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("Could not fetch pollutant data.")
    else:
        st.error("City not found.")