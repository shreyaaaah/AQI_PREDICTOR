import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import os

# 📌 Load unbalanced dataset
unbalanced_df = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\AQI PREDICTOR\cleaned_station_day_realtime.csv')

# 📌 Load balanced dataset
balanced_df = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\AQI PREDICTOR\balanced_station_day_realtime.csv')

# 📌 Create folder for saving plots if not exists
output_dir = r'C:\Users\LENOVO\OneDrive\Desktop\AQI PREDICTOR\EDA_PLOTS'
os.makedirs(output_dir, exist_ok=True)

# 📊 DATA OVERVIEW
print("🔍 Unbalanced Dataset:")
print(unbalanced_df.info())
print(unbalanced_df.describe())

print("\n🔍 Balanced Dataset:")
print(balanced_df.info())
print(balanced_df.describe())

# 📉 MISSING VALUE CHECK
print("\n🚨 Missing values in Unbalanced Data:\n", unbalanced_df.isnull().sum())
print("\n🚨 Missing values in Balanced Data:\n", balanced_df.isnull().sum())

# 📊 AQI CATEGORY DISTRIBUTIONS
plt.figure(figsize=(8,4))
sns.countplot(x='AQI_Bucket', data=unbalanced_df, palette='viridis')
plt.title("Unbalanced AQI Category Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'unbalanced_aqi_category_distribution.png'))
plt.close()

plt.figure(figsize=(8,4))
sns.countplot(x='AQI_Bucket', data=balanced_df, palette='magma')
plt.title("Balanced AQI Category Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'balanced_aqi_category_distribution.png'))
plt.close()

# 📈 HISTOGRAMS OF POLLUTANTS
unbalanced_df[['PM2.5','PM10','NO','NO2','CO','SO2','O3','NH3']].hist(figsize=(14,10), color='steelblue', edgecolor='black')
plt.suptitle("Unbalanced Data: Pollutant Distributions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'unbalanced_pollutant_distributions.png'))
plt.close()

balanced_df[['PM2.5','PM10','NO','NO2','CO','SO2','O3','NH3']].hist(figsize=(14,10), color='coral', edgecolor='black')
plt.suptitle("Balanced Data: Pollutant Distributions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'balanced_pollutant_distributions.png'))
plt.close()

# 📊 BOXPLOTS FOR OUTLIERS
plt.figure(figsize=(12,6))
sns.boxplot(data=unbalanced_df[['PM2.5','PM10','NO','NO2','CO','SO2','O3','NH3']])
plt.title("Unbalanced Data: Boxplots for Outlier Detection")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'unbalanced_boxplot.png'))
plt.close()

plt.figure(figsize=(12,6))
sns.boxplot(data=balanced_df[['PM2.5','PM10','NO','NO2','CO','SO2','O3','NH3']])
plt.title("Balanced Data: Boxplots for Outlier Detection")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'balanced_boxplot.png'))
plt.close()

# 📈 CORRELATION HEATMAPS
plt.figure(figsize=(10,8))
sns.heatmap(unbalanced_df[['PM2.5','PM10','NO','NO2','CO','SO2','O3','NH3','AQI']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Unbalanced Data: Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'unbalanced_correlation_heatmap.png'))
plt.close()

plt.figure(figsize=(10,8))
sns.heatmap(balanced_df[['PM2.5','PM10','NO','NO2','CO','SO2','O3','NH3','AQI']].corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title("Balanced Data: Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'balanced_correlation_heatmap.png'))
plt.close()

# 📌 FEATURE IMPORTANCE (for classification AQI_Bucket — using Mutual Information)
label_encoder = LabelEncoder()
balanced_df['AQI_Bucket_encoded'] = label_encoder.fit_transform(balanced_df['AQI_Bucket'])

from sklearn.feature_selection import mutual_info_classif
X = balanced_df[['PM2.5','PM10','NO','NO2','CO','SO2','O3','NH3']]
y = balanced_df['AQI_Bucket_encoded']

mi_scores = mutual_info_classif(X, y, random_state=42)
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Mutual_Info_Score': mi_scores})
feature_importance_df.sort_values(by='Mutual_Info_Score', ascending=False, inplace=True)

# 📈 PLOT FEATURE IMPORTANCES
plt.figure(figsize=(8,5))
sns.barplot(x='Mutual_Info_Score', y='Feature', data=feature_importance_df, palette='crest')
plt.title("Feature Importance (Mutual Information) — Classification")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_mutual_info.png'))
plt.close()

# 📊 Also print feature importance
print("\n📌 Feature Importance (Mutual Information):\n", feature_importance_df)

