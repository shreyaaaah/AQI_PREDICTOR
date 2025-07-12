import pandas as pd
import os

# 📌 Path to your cleaned dataset
file_path = r'C:\Users\LENOVO\OneDrive\Desktop\AQI PREDICTOR\cleaned_station_day_realtime.csv'

# 1️⃣ Load the cleaned dataset
df = pd.read_csv(file_path)

# 2️⃣ Check category distribution before balancing
print("\n📊 Original AQI_Bucket distribution:")
print(df['AQI_Bucket'].value_counts())

# 3️⃣ Find minimum count of any category (to balance others down to this)
min_count = df['AQI_Bucket'].value_counts().min()

# 4️⃣ Downsample each category to min_count
balanced_df = df.groupby('AQI_Bucket').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

# 5️⃣ Check category distribution after balancing
print("\n📊 Balanced AQI_Bucket distribution:")
print(balanced_df['AQI_Bucket'].value_counts())

# 6️⃣ Save balanced dataset
output_path = r'C:\Users\LENOVO\OneDrive\Desktop\AQI PREDICTOR\balanced_station_day_realtime.csv'
balanced_df.to_csv(output_path, index=False)

print(f"\n✅ Balanced dataset saved at: '{output_path}'")
