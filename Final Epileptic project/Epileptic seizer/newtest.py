import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Load Model & Scaler
# ===============================
try:
    model = load_model("best_cnn_rnn_model.h5")  # trained model
    scaler = joblib.load("scaler.joblib")           # scaler saved during training
except Exception as e:
    print("⚠️ Make sure 'best_cnn_rnn_model.h5' and 'scaler.joblib' are in the same folder")
    raise e

# Feature names (10 inputs)
FEATURES = [
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

entries = {}  # to hold entry widgets

# ===============================
# Prediction Function
# ===============================
def predict_input():
    try:
        values = []
        for feat in FEATURES:
            val = entries[feat].get().strip()
            if val == "":
                messagebox.showerror("Error", f"Missing value for {feat}")
                return
            values.append(float(val))

        # Convert to numpy
        values = np.array([values])  # shape (1, 10)

        # Scale input
        values = scaler.transform(values)

        # Reshape for CNN+RNN
        values = values.reshape((values.shape[0], values.shape[1], 1))

        # Predict
        y_pred = model.predict(values)
        y_class = int((y_pred > 0.5).astype("int32")[0][0])

        if y_class == 1:
            result_text.set("⚠️ Seizure Detected")
        else:
            result_text.set("✅ Non-Seizure")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {str(e)}")

# ===============================
# GUI
# ===============================
root = tk.Tk()
root.title("EEG Seizure Detection (Manual Input)")
root.geometry("500x600")

title_label = tk.Label(root, text="Epileptic Seizure Detection", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Create entry fields for each feature
frame = tk.Frame(root)
frame.pack(pady=10)

for feat in FEATURES:
    row = tk.Frame(frame)
    lbl = tk.Label(row, text=feat, width=20, anchor="w", font=("Arial", 11))
    ent = tk.Entry(row, width=25, font=("Arial", 11))
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    lbl.pack(side=tk.LEFT)
    ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[feat] = ent

predict_btn = tk.Button(root, text="Predict", command=predict_input, font=("Arial", 12), bg="lightblue")
predict_btn.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14), fg="darkgreen")
result_label.pack(pady=20)

exit_btn = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12), bg="lightcoral")
exit_btn.pack(pady=10)

root.mainloop()
