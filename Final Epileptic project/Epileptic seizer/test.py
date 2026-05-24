import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# ==============================================
# 1. Load trained models and scaler
# ==============================================
rf_model = joblib.load("random_forest_model.joblib")
xgb_model = joblib.load("xgboost_model.joblib")
scaler = joblib.load("scaler.joblib")

# ==============================================
# 2. Feature names (must match training dataset)
# ==============================================
feature_names = [
    "FP1_power_delta",
    "FP1_power_alpha",
    "FP2_power_beta",
    "F3_power_theta",
    "C3_power_gamma",
    "P4_power_alpha",
    "O1_power_beta",
    "T4_power_delta",
    "C3_LSWT_D3",
    "F4_Mean_D2"
]

# ==============================================
# 3. Create GUI window
# ==============================================
root = tk.Tk()
root.title("Epileptic Seizure Detection (EEG)")
root.geometry("400x600")

entries = {}

tk.Label(root, text="Enter EEG Feature Values", font=("Arial", 14, "bold")).pack(pady=10)

# Create input boxes
for feat in feature_names:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    lbl = tk.Label(frame, text=feat, width=15, anchor="w")
    lbl.pack(side=tk.LEFT)
    ent = tk.Entry(frame, width=20)
    ent.pack(side=tk.RIGHT)
    entries[feat] = ent

# ==============================================
# 4. Prediction function
# ==============================================
def predict():
    try:
        user_input = []
        for feat in feature_names:
            val = float(entries[feat].get() or 0.0)  # default = 0 if empty
            user_input.append(val)

        # Convert and scale
        user_input = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        # Predictions
        rf_pred = rf_model.predict(user_input_scaled)[0]
        xgb_pred = xgb_model.predict(user_input_scaled)[0]

        # Show result
        result_msg = f"""
        🌲 Random Forest: {'Seizure' if rf_pred == 1 else 'Non-Seizure'}
        ⚡ XGBoost: {'Seizure' if xgb_pred == 1 else 'Non-Seizure'}
        """
        messagebox.showinfo("Prediction Result", result_msg)

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong!\n{str(e)}")

# ==============================================
# 5. Predict Button
# ==============================================
tk.Button(root, text="Predict", command=predict, bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=20)

# ==============================================
# Run App
# ==============================================
root.mainloop()
