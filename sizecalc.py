import os

files = [
    "final_stack_model_optimized.pkl",
    "final_stack_regression_optimized.pkl",
    "scaler_station.pkl",
    "label_encoder_station.pkl"
]

for file in files:
    size = os.path.getsize(file) / (1024 * 1024)  # size in MB
    print(f"{file}: {size:.2f} MB")
