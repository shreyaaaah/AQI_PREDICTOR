import pandas as pd
import os

# 📌 Confirm working directory
print("📂 Current working directory:", os.getcwd())

# ✅ Path to your dataset (absolute path)
file_path = r'C:\Users\LENOVO\OneDrive\Desktop\AQI PREDICTOR\station_day.csv'

# 1️⃣ Read the CSV file
df = pd.read_csv(file_path)

# 2️⃣ Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 3️⃣ Drop unnecessary columns if they exist
columns_to_drop = ['Benzene', 'Toluene', 'Xylene', 'NOx']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# 4️⃣ Drop any rows with missing values
df.dropna(inplace=True)

# 5️⃣ Final data check
print("\n✅ Final Clean Data Summary:")
print(df.info())
print("\nUnique AQI categories:", df['AQI_Bucket'].unique())

# 6️⃣ Save cleaned dataset
output_path = r'C:\Users\LENOVO\OneDrive\Desktop\AQI PREDICTOR\cleaned_station_day_realtime.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved at: '{output_path}'")
