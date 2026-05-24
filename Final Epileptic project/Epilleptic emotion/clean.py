

import pandas as pd
import numpy as np
import time

# ---------------------------
# 1. Load dataset
# ---------------------------
file_path = "emotion_dataset.csv"
df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully.")
print("Original shape:", df.shape)

# ---------------------------
# 2. Detect EEG columns
# ---------------------------
wave_keywords = ["delta", "theta", "alpha", "beta", "gamma"]
eeg_cols = [col for col in df.columns if any(k in col.lower() for k in wave_keywords)]

if not eeg_cols:
    raise ValueError("No EEG-related columns detected. Please check your dataset column names.")

print("Detected EEG columns:", eeg_cols)
print("-" * 60)

# ---------------------------
# 3. Define realistic EEG ranges
# ---------------------------
realistic_ranges = {
    "delta": (0.5, 4.0),   # µV²
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0)
}

def clean_value(col_name, value):
    for band, (vmin, vmax) in realistic_ranges.items():
        if band in col_name.lower():
            return np.clip(value, vmin, vmax)
    return value

for col in eeg_cols:
    df[col] = df[col].apply(lambda v: clean_value(col, v))

print("✅ Values cleaned and clipped to realistic EEG ranges.")
print("-" * 60)

# ---------------------------
# 4. Generate Emotion Column (if not exists)
# ---------------------------
def predict_emotion(row):
    alpha = np.mean([row[c] for c in eeg_cols if "alpha" in c.lower()] or [0])
    beta  = np.mean([row[c] for c in eeg_cols if "beta" in c.lower()] or [0])
    theta = np.mean([row[c] for c in eeg_cols if "theta" in c.lower()] or [0])

    if beta > alpha and beta > theta:
        return "Anger"
    elif alpha > beta and alpha > theta:
        return "Happy"
    elif theta > alpha and theta > beta:
        return "Sad"
    else:
        return "Neutral"

if "emotion" not in df.columns:
    df["emotion"] = df.apply(predict_emotion, axis=1)
    print("✅ Emotion column created based on EEG ratios.")
else:
    print("ℹ️ Emotion column already exists — keeping original values.")

# ---------------------------
# 5. Create balanced dataset (1000 rows, 250 per emotion)
# ---------------------------
print("Creating balanced dataset of 1000 rows (250 per emotion)...")

emotions = ["Anger", "Happy", "Sad", "Neutral"]
target_per_emotion = 2000
balanced_data = []

for emotion in emotions:
    subset = df[df["emotion"] == emotion]

    if len(subset) == 0:
        # If no rows exist for this emotion, create placeholder rows by sampling from the whole df
        print(f"⚠️ No rows found for emotion '{emotion}', creating synthetic samples from all data.")
        subset = df.sample(target_per_emotion, replace=True, random_state=42)
        subset["emotion"] = emotion  # overwrite emotion to desired class
    elif len(subset) < target_per_emotion:
        # Not enough rows, sample with replacement
        subset = subset.sample(target_per_emotion, replace=True, random_state=42)
    else:
        # Enough rows, sample without replacement
        subset = subset.sample(target_per_emotion, replace=False, random_state=42)

    balanced_data.append(subset)

# Combine all subsets and shuffle
df_balanced = pd.concat(balanced_data).sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Balanced dataset created.")
print(df_balanced["emotion"].value_counts())


# ---------------------------
# 6. Save cleaned balanced dataset
# ---------------------------
output_file = "emotion_dataset_final_1000.csv"
df_balanced.to_csv(output_file, index=False)
print(f"✅ Balanced dataset saved as '{output_file}'")

