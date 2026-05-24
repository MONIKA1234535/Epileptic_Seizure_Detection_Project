#py 3.11
# ==============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib   # for saving/loading models

# ==============================================
# 1. Load Dataset
# ==============================================
df = pd.read_csv("eeg_data.csv")

print("Dataset Shape:", df.shape)

# Features and Labels
X = df.drop(columns=["epileptic_"])
y = df["epileptic_"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================================
# 2. Random Forest with Hyperparameter Tuning
# ==============================================
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\n==============================")
print("🌲 Random Forest (Tuned) Results")
print("==============================")
print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save Random Forest model
joblib.dump(best_rf, "random_forest_model.joblib")
print("✅ Random Forest model saved as random_forest_model.joblib")

# Save scaler (important for prediction later)
joblib.dump(scaler, "scaler.joblib")
print("✅ Scaler saved as scaler.joblib")

# ==============================================
# 3. XGBoost Classifier
# ==============================================
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\n==============================")
print("⚡ XGBoost Results")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Save XGBoost model
joblib.dump(xgb_model, "xgboost_model.joblib")
print("✅ XGBoost model saved as xgboost_model.joblib")

# ==============================================
# 4. Example: Load and Predict
# ==============================================
# Load back models
rf_loaded = joblib.load("random_forest_model.joblib")
scaler_loaded = joblib.load("scaler.joblib")

# Example prediction using first row of X_test
sample = X_test[0].reshape(1, -1)
prediction = rf_loaded.predict(sample)
print("\n🔮 Example Prediction (Random Forest) on one test sample:", prediction[0])
