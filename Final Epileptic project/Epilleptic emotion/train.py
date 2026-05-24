import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ======================================================
# 1️⃣ Load Dataset
# ======================================================
file_path = "dataset_new.csv"
df = pd.read_csv(file_path)
print("✅ Balanced dataset loaded:", df.shape)

wave_keywords = ["delta", "theta", "alpha", "beta", "gamma"]
eeg_cols = [col for col in df.columns if any(k in col.lower() for k in wave_keywords)]
print("Detected EEG columns:", eeg_cols)

# ======================================================
# 2️⃣ Derived EEG Features
# ======================================================
df["beta_alpha_ratio"]  = df["beta_wave_intensity"]  / (df["alpha_wave_intensity"] + 1e-6)
df["theta_alpha_ratio"] = df["theta_wave_intensity"] / (df["alpha_wave_intensity"] + 1e-6)
df["alpha_theta_ratio"] = df["alpha_wave_intensity"] / (df["theta_wave_intensity"] + 1e-6)
df["alpha_beta_diff"]   = df["alpha_wave_intensity"] - df["beta_wave_intensity"]
df["theta_gamma_ratio"] = df["theta_wave_intensity"] / (df["gamma_wave_intensity"] + 1e-6)
df["entropy_sum"] = df[[
    "delta_wave_entropy", "theta_wave_entropy", "alpha_wave_entropy",
    "beta_wave_entropy", "gamma_wave_entropy"
]].sum(axis=1)
df["engagement_index"] = df["beta_wave_intensity"] / (df["alpha_wave_intensity"] + df["theta_wave_intensity"] + 1e-6)
df["relaxation_index"] = df["alpha_wave_intensity"] / (df["beta_wave_intensity"] + 1e-6)
df["stress_index"] = df["beta_wave_intensity"] / (df["delta_wave_intensity"] + 1e-6)

feature_cols = eeg_cols + [
    "beta_alpha_ratio", "theta_alpha_ratio", "alpha_theta_ratio",
    "alpha_beta_diff", "theta_gamma_ratio", "entropy_sum",
    "engagement_index", "relaxation_index", "stress_index"
]

X = df[feature_cols]
y = df["emotion"]

# ======================================================
# 3️⃣ Encode Labels
# ======================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Label classes:", le.classes_)

# ======================================================
# 4️⃣ Scale + Balance Data (SMOTE)
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y_encoded)
print("✅ SMOTE applied:", X_bal.shape)

# ======================================================
# 5️⃣ Optional PCA
# ======================================================
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_bal)
print("PCA reduced dimensions:", X_pca.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ======================================================
# 6️⃣ Random Forest
# ======================================================
rf = RandomForestClassifier(
    n_estimators=600, max_depth=14, min_samples_leaf=3, class_weight='balanced', random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
print(f"🌲 Random Forest Accuracy: {rf_acc*100:.2f}%, F1: {rf_f1*100:.2f}%")

# ======================================================
# 7️⃣ XGBoost
# ======================================================
xgb = XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=10,
    subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
print(f"🔥 XGBoost Accuracy: {xgb_acc*100:.2f}%, F1: {xgb_f1*100:.2f}%")

# ======================================================
# 8️⃣ LightGBM
# ======================================================
lgb = LGBMClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=10, class_weight='balanced', random_state=42
)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_f1 = f1_score(y_test, lgb_pred, average='weighted')
print(f"💡 LightGBM Accuracy: {lgb_acc*100:.2f}%, F1: {lgb_f1*100:.2f}%")

# ======================================================
# 9️⃣ Deep Neural Network
# ======================================================
num_classes = len(np.unique(y_bal))
dnn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
dnn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)

dnn_acc = dnn.evaluate(X_test, y_test, verbose=0)[1]
print(f"🧠 DNN Accuracy: {dnn_acc*100:.2f}%")

# ======================================================
# 10️⃣ Compare Results
# ======================================================
results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "LightGBM", "Deep NN"],
    "Accuracy": [rf_acc, xgb_acc, lgb_acc, dnn_acc],
    "F1 Score": [rf_f1, xgb_f1, lgb_f1, None]
})
print("\n🔍 Model Comparison:\n", results)

best_model = results.loc[results["Accuracy"].idxmax()]
print(f"\n🏆 Best Model: {best_model['Model']} with {best_model['Accuracy']*100:.2f}% accuracy")

# ======================================================
# 11️⃣ Confusion Matrix for Best Model
# ======================================================
best_pred = {
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "LightGBM": lgb_pred,
    "Deep NN": np.argmax(dnn.predict(X_test), axis=1)
}[best_model['Model']]

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix ({best_model['Model']})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

print(f"\nConfusion Matrix for {best_model['Model']}:\n")
print(cm)

report = classification_report(y_test, best_pred, target_names=le.classes_)
print(f"\nClassification Report for {best_model['Model']}:\n")
print(report)

# -----------------------------
# Save preprocessing objects
# -----------------------------
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n✅ Scaler, PCA, and Label Encoder saved successfully")


# ======================================================
# 12️⃣ Save Best Model
# ======================================================
best_model_name = best_model['Model']

if best_model_name == "Deep NN":
    save_model(dnn, "best_model_dnn.h5")
    print(f"\n✅ Deep NN model saved as: best_model_dnn.h5")
else:
    best_model_obj = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "LightGBM": lgb
    }[best_model_name]
    joblib.dump(best_model_obj, f"best_model_{best_model_name.replace(' ', '_')}.pkl")
    print(f"\n✅ {best_model_name} model saved as: best_model_{best_model_name.replace(' ', '_')}.pkl")
