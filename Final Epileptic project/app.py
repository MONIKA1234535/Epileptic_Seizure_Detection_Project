from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib
import os
import sqlite3
import json

# ---------------- Flask App Config ----------------
app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"

# ---------------- Database Setup ----------------
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- Load Models and Scalers ----------------
SEIZURE_MODEL_PATH = "Epileptic seizer/best_cnn_rnn_model.h5"
SEIZURE_SCALER_PATH = "Epileptic seizer/scaler.joblib"

EMOTION_SCALER_PATH = "Epilleptic emotion/scaler.pkl"
EMOTION_PCA_PATH = "Epilleptic emotion/pca.pkl"
EMOTION_ENCODER_PATH = "Epilleptic emotion/label_encoder.pkl"
EMOTION_MODEL_PATH = "Epilleptic emotion/best_model_dnn.h5"

# Verify existence
for path in [SEIZURE_MODEL_PATH, SEIZURE_SCALER_PATH,
             EMOTION_SCALER_PATH, EMOTION_PCA_PATH, EMOTION_ENCODER_PATH, EMOTION_MODEL_PATH]:
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")

# Load models
seizure_model = load_model(SEIZURE_MODEL_PATH)
seizure_scaler = joblib.load(SEIZURE_SCALER_PATH)

emotion_scaler = joblib.load(EMOTION_SCALER_PATH)
emotion_pca = joblib.load(EMOTION_PCA_PATH)
emotion_encoder = joblib.load(EMOTION_ENCODER_PATH)

if EMOTION_MODEL_PATH.endswith(".h5"):
    emotion_model = load_model(EMOTION_MODEL_PATH)
else:
    emotion_model = joblib.load(EMOTION_MODEL_PATH)

# ---------------- Feature Names ----------------
SEIZURE_FEATURES = [
    "FP1_power_delta", "FP1_power_alpha", "FP2_power_beta",
    "F3_power_theta", "C3_power_gamma", "P4_power_alpha",
    "O1_power_beta", "T4_power_delta", "C3_LSWT_D3", "F4_Mean_D2"
]

EMOTION_FEATURES = [
    "delta_wave_intensity", "theta_wave_intensity", "alpha_wave_intensity",
    "beta_wave_intensity", "gamma_wave_intensity",
    "delta_wave_entropy", "theta_wave_entropy", "alpha_wave_entropy",
    "beta_wave_entropy", "gamma_wave_entropy"
]

# ---------------- Helper: Check Login ----------------
def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_email" not in session:
            flash("⚠️ Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- Registration ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        if cursor.fetchone():
            flash("Email already registered.", "danger")
            conn.close()
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password)
        cursor.execute("INSERT INTO users (name, email,  password) VALUES (?, ?, ?)",
                       (name, email, hashed_pw))
        conn.commit()
        conn.close()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# ---------------- Login ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[4], password):
            session["user_email"] = user[2]
            session["user_name"] = user[1]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.", "danger")

    return render_template("login.html")

# ---------------- Logout ----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))


# ---------------- Contact ----------------
@app.route("/contact")
def contact():
    return render_template("contact.html")

# ---------------- About ----------------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- Seizure Detection ----------------

seizure_df = pd.read_csv("Epileptic seizer/eeg_data.csv")
SEIZURE_FEATURES = ["FP1_power_delta", "FP1_power_alpha", "FP2_power_beta","F3_power_theta","C3_power_gamma","P4_power_alpha","O1_power_beta","T4_power_delta","C3_LSWT_D3","F4_Mean_D2"]  

@app.route("/seizure", methods=["GET", "POST"])
@login_required
def seizure():
    prediction = None
    matching_row = None
    row_color = "#fff"  # default white
    if request.method == "POST":
        try:
            values = []
            input_dict = {}
            for feat in SEIZURE_FEATURES:
                val = request.form.get(feat)
                if not val or val.strip() == "":
                    flash(f"Missing value for {feat}", "danger")
                    return redirect(url_for("seizure"))
                val_float = float(val)
                values.append(val_float)
                input_dict[feat] = val_float

            # Prepare input for model
            values_array = np.array([values])
            values_array = seizure_scaler.transform(values_array)
            values_array = values_array.reshape((values_array.shape[0], values_array.shape[1], 1))

            # Predict seizure
            y_pred = seizure_model.predict(values_array)
            y_class = int((y_pred > 0.5).astype("int32")[0][0])
            prediction = "⚠️ Seizure Detected" if y_class == 1 else "✅ Non-Seizure"

            # Check F4_Mean_D2 in dataset
            f4_val = input_dict["F4_Mean_D2"]
            row_match = seizure_df[seizure_df["F4_Mean_D2"] == f4_val]
            if not row_match.empty:
                epileptic_status = row_match.iloc[0]["epileptic_"]
                matching_row = f"Row check: F4_Mean_D2={f4_val}, Epileptic={epileptic_status}"
                row_color = "#e74c3c" if epileptic_status == 1 else "#27ae60"
            else:
                matching_row = f"No matching row found for F4_Mean_D2={f4_val}"
                row_color = "#fff"

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")
            return redirect(url_for("seizure"))

    return render_template(
        "seizure.html",
        features=SEIZURE_FEATURES,
        prediction=prediction,
        matching_row=matching_row,
        row_color=row_color
    )


# ---------------- Emotion Detection ----------------
@app.route("/emotion", methods=["GET", "POST"])
@login_required
def emotion():
    prediction = None
    if request.method == "POST":
        try:
            values = {}
            for f in EMOTION_FEATURES:
                val = request.form.get(f)
                if not val or val.strip() == "":
                    flash(f"Missing value for {f}", "danger")
                    return redirect(url_for("emotion"))
                values[f] = float(val)

            derived = {
                "beta_alpha_ratio": values["beta_wave_intensity"] / (values["alpha_wave_intensity"] + 1e-6),
                "theta_alpha_ratio": values["theta_wave_intensity"] / (values["alpha_wave_intensity"] + 1e-6),
                "alpha_theta_ratio": values["alpha_wave_intensity"] / (values["theta_wave_intensity"] + 1e-6),
                "alpha_beta_diff": values["alpha_wave_intensity"] - values["beta_wave_intensity"],
                "theta_gamma_ratio": values["theta_wave_intensity"] / (values["gamma_wave_intensity"] + 1e-6),
                "entropy_sum": sum([
                    values["delta_wave_entropy"], values["theta_wave_entropy"],
                    values["alpha_wave_entropy"], values["beta_wave_entropy"],
                    values["gamma_wave_entropy"]
                ]),
                "engagement_index": values["beta_wave_intensity"] /
                    (values["alpha_wave_intensity"] + values["theta_wave_intensity"] + 1e-6),
                "relaxation_index": values["alpha_wave_intensity"] / (values["beta_wave_intensity"] + 1e-6),
                "stress_index": values["beta_wave_intensity"] / (values["delta_wave_intensity"] + 1e-6)
            }

            input_df = pd.DataFrame([{**values, **derived}])
            X_scaled = emotion_scaler.transform(input_df)
            X_pca = emotion_pca.transform(X_scaled)

            if EMOTION_MODEL_PATH.endswith(".h5"):
                pred_idx = np.argmax(emotion_model.predict(X_pca), axis=1)[0]
            else:
                pred_idx = emotion_model.predict(X_pca)[0]

            pred_label = emotion_encoder.inverse_transform([pred_idx])[0]
            prediction = f"🎯 Predicted Emotion: {pred_label}"

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")
            return redirect(url_for("emotion"))

    return render_template("emotion.html", features=EMOTION_FEATURES, prediction=prediction)

# ---------------- Chatbot Route ----------------
@app.route("/chatbot", methods=["GET", "POST"])
@login_required
def chatbot():
    if request.method == "POST":
        user_message = request.json.get("message", "").lower().strip()
        if not user_message:
            return jsonify({"response": "Please say something to start the chat."})

        try:
            with open("chatbot_data.json", "r") as f:
                chatbot_data = json.load(f)

            # Simple keyword-based response
            for key, reply in chatbot_data.items():
                if key in user_message:
                    return jsonify({"response": reply})

            # Default fallback
            return jsonify({"response": "I'm not sure about that. Please consult a doctor for accurate medical advice."})

        except Exception as e:
            return jsonify({"response": f"Error: {str(e)}"})

    return render_template("chatbot.html")

# ---------------- Run Flask App ----------------
if __name__ == "__main__":
    app.run(debug=True)
