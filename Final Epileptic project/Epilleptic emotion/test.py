import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# 1️⃣ Load Saved Objects
# -----------------------------
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
le = joblib.load("label_encoder.pkl")

# Replace with your actual saved best model
# It could be "best_model_dnn.h5" or "best_model_Random_Forest.pkl", etc.
best_model_path = "best_model_dnn.h5"

if best_model_path.endswith(".h5"):
    model = load_model(best_model_path)
else:
    model = joblib.load(best_model_path)

# -----------------------------
# 2️⃣ Get User Input
# -----------------------------
wave_features = [
    "delta_wave_intensity", "theta_wave_intensity", "alpha_wave_intensity",
    "beta_wave_intensity", "gamma_wave_intensity",
    "delta_wave_entropy", "theta_wave_entropy", "alpha_wave_entropy",
    "beta_wave_entropy", "gamma_wave_entropy"
]

print("Enter EEG feature values:")
user_input = {}
for f in wave_features:
    user_input[f] = float(input(f"{f}: "))

# -----------------------------
# 3️⃣ Compute Derived Features
# -----------------------------
derived_features = {}
derived_features["beta_alpha_ratio"]  = user_input["beta_wave_intensity"] / (user_input["alpha_wave_intensity"] + 1e-6)
derived_features["theta_alpha_ratio"] = user_input["theta_wave_intensity"] / (user_input["alpha_wave_intensity"] + 1e-6)
derived_features["alpha_theta_ratio"] = user_input["alpha_wave_intensity"] / (user_input["theta_wave_intensity"] + 1e-6)
derived_features["alpha_beta_diff"]   = user_input["alpha_wave_intensity"] - user_input["beta_wave_intensity"]
derived_features["theta_gamma_ratio"] = user_input["theta_wave_intensity"] / (user_input["gamma_wave_intensity"] + 1e-6)
derived_features["entropy_sum"] = sum([
    user_input["delta_wave_entropy"], user_input["theta_wave_entropy"], user_input["alpha_wave_entropy"],
    user_input["beta_wave_entropy"], user_input["gamma_wave_entropy"]
])
derived_features["engagement_index"] = user_input["beta_wave_intensity"] / (
    user_input["alpha_wave_intensity"] + user_input["theta_wave_intensity"] + 1e-6
)
derived_features["relaxation_index"] = user_input["alpha_wave_intensity"] / (user_input["beta_wave_intensity"] + 1e-6)
derived_features["stress_index"] = user_input["beta_wave_intensity"] / (user_input["delta_wave_intensity"] + 1e-6)

# -----------------------------
# 4️⃣ Prepare DataFrame
# -----------------------------
input_df = pd.DataFrame([{**user_input, **derived_features}])

# -----------------------------
# 5️⃣ Scale + PCA
# -----------------------------
X_scaled = scaler.transform(input_df)
X_pca = pca.transform(X_scaled)

# -----------------------------
# 6️⃣ Make Prediction
# -----------------------------
if best_model_path.endswith(".h5"):
    pred_idx = np.argmax(model.predict(X_pca), axis=1)[0]
else:
    pred_idx = model.predict(X_pca)[0]

pred_label = le.inverse_transform([pred_idx])[0]
print(f"\n🎯 Predicted Emotion: {pred_label}")
