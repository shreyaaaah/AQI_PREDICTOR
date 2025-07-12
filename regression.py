import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")

# 📌 Load balanced dataset
df = pd.read_csv(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/cleaned_station_day_realtime.csv')
df.drop(['StationId', 'Date', 'AQI_Bucket'], axis=1, inplace=True)

# 📌 Feature Engineering
df['PM2.5/PM10'] = df['PM2.5'] / (df['PM10'] + 1)
df['NO/NO2'] = df['NO'] / (df['NO2'] + 1)

# 📌 Features and target
X = df.drop(['AQI'], axis=1)
y = df['AQI']

# 📌 Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 📌 Optuna Hyperparameter Tuning functions
def tune_xgb(trial):
    return XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 400, 1000),
        max_depth=trial.suggest_int('max_depth', 4, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.3),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        gamma=trial.suggest_float('gamma', 0, 2),
        random_state=42
    )

def tune_lgb(trial):
    return LGBMRegressor(
        n_estimators=trial.suggest_int('n_estimators', 400, 1000),
        max_depth=trial.suggest_int('max_depth', 4, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.3),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        random_state=42
    )

def tune_cat(trial):
    return CatBoostRegressor(
        iterations=trial.suggest_int('iterations', 400, 1000),
        depth=trial.suggest_int('depth', 4, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.3),
        random_seed=42,
        verbose=0
    )

# 📌 Optuna optimization
def optuna_objective(trial):
    xgb_reg = tune_xgb(trial)
    lgb_reg = tune_lgb(trial)
    cat_reg = tune_cat(trial)
    meta_reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.07, random_state=42)

    stack_model = StackingRegressor(
        estimators=[('xgb', xgb_reg), ('lgb', lgb_reg), ('cat', cat_reg)],
        final_estimator=meta_reg,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    stack_model.fit(X_train, y_train)
    preds = stack_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    return r2

study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=20)

# 📌 Best Hyperparameters
print("✅ Best trial parameters:\n", study.best_params)

# 📌 Train final model with best hyperparameters
best_trial = study.best_trial

xgb_best = tune_xgb(best_trial)
lgb_best = tune_lgb(best_trial)
cat_best = tune_cat(best_trial)
meta_reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.07, random_state=42)

stack_model = StackingRegressor(
    estimators=[('xgb', xgb_best), ('lgb', lgb_best), ('cat', cat_best)],
    final_estimator=meta_reg,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

stack_model.fit(X_train, y_train)

# 📌 Predictions
y_pred = stack_model.predict(X_test)

# 📌 Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'\n✅ Final Optimized Stacking Regression Performance:')
print(f'R² Score: {r2:.4f}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# 📌 Save final model and scaler
joblib.dump(stack_model, r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/final_stack_regression_optimized.pkl')
joblib.dump(scaler, r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/scaler_regression.pkl')

print("\n✅ Final optimized stacking regression model and scaler saved successfully.")
