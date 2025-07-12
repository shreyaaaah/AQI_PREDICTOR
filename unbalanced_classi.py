import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import joblib
import warnings
warnings.filterwarnings("ignore")

# 📌 Load unbalanced dataset
df = pd.read_csv(r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/cleaned_station_day_realtime.csv')
df.drop(['StationId', 'Date', 'AQI'], axis=1, inplace=True)

# 📌 Feature Engineering
df['PM2.5/PM10'] = df['PM2.5'] / (df['PM10'] + 1)
df['NO/NO2'] = df['NO'] / (df['NO2'] + 1)

# 📌 Features and target
X = df.drop(['AQI_Bucket'], axis=1)
y = df['AQI_Bucket']

# 📌 Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 📌 Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# 📌 Optuna Hyperparameter Tuning functions
def tune_xgb(trial):
    return XGBClassifier(
        n_estimators=trial.suggest_int('n_estimators', 400, 1000),
        max_depth=trial.suggest_int('max_depth', 4, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.3),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        gamma=trial.suggest_float('gamma', 0, 2),
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

def tune_lgb(trial):
    return LGBMClassifier(
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
    return CatBoostClassifier(
        iterations=trial.suggest_int('iterations', 400, 1000),
        depth=trial.suggest_int('depth', 4, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.3),
        random_seed=42,
        verbose=0
    )

# 📌 Optuna optimization
def optuna_objective(trial):
    xgb_clf = tune_xgb(trial)
    lgb_clf = tune_lgb(trial)
    cat_clf = tune_cat(trial)
    meta_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.07, random_state=42)

    stack_model = StackingClassifier(
        estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('cat', cat_clf)],
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    stack_model.fit(X_train, y_train)
    preds = stack_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(optuna_objective, n_trials=20)

# 📌 Best Hyperparameters
print("✅ Best trial parameters:\n", study.best_params)

# 📌 Train final model with best hyperparameters
best_trial = study.best_trial

xgb_best = tune_xgb(best_trial)
lgb_best = tune_lgb(best_trial)
cat_best = tune_cat(best_trial)
meta_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.07, random_state=42)

stack_model = StackingClassifier(
    estimators=[('xgb', xgb_best), ('lgb', lgb_best), ('cat', cat_best)],
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

stack_model.fit(X_train, y_train)

# 📌 Predictions
y_pred = stack_model.predict(X_test)

# 📌 Evaluation
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(label_binarize(y_test, classes=np.unique(y_encoded)),
                        label_binarize(y_pred, classes=np.unique(y_encoded)),
                        average='macro')

print(f'\n✅ Final Optimized Stacking Ensemble Performance (Unbalanced Data):')
print(f'Accuracy: {acc:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
print(f'ROC-AUC Score: {roc_auc:.4f}')
print(f'\n📊 Confusion Matrix:\n{conf_matrix}')

# 📌 Save final model and encoders
joblib.dump(stack_model, r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/final_stack_model_unbalanced.pkl')
joblib.dump(le, r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/label_encoder_unbalanced.pkl')
joblib.dump(scaler, r'C:/Users/LENOVO/OneDrive/Desktop/AQI PREDICTOR/scaler_unbalanced.pkl')

print("\n✅ Final optimized stacking model and encoders (unbalanced) saved successfully.")
 