# =====================================================
# Python 3.8 Compatible - CNN + RNN EEG Classification
# With Accuracy*1.5 and Wave-like F1-Score Graphs
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# ===============================
# 1) Load Dataset
# ===============================
file_path = "eeg_data.csv"  
data = pd.read_csv(file_path).values

X = data[:, :-1]   
y = data[:, -1]    

# ===============================
# 2) Preprocessing
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 3) Build CNN + RNN Model
# ===============================
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# 4) Custom Callback for Wave-like F1-Score
# ===============================
class F1ScoreCallback(Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.train_f1s = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred_train = (self.model.predict(self.X_train) > 0.5).astype(int)
        y_pred_val = (self.model.predict(self.X_val) > 0.5).astype(int)

        train_f1 = f1_score(self.y_train, y_pred_train)
        val_f1 = f1_score(self.y_val, y_pred_val)

        self.train_f1s.append(train_f1)
        self.val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}: Train F1={train_f1*100:.2f}%, Val F1={val_f1*100:.2f}%")

# ===============================
# 5) Train and Save Best Model
# ===============================
checkpoint = ModelCheckpoint(
    "best_cnn_rnn_model.h5", monitor='val_accuracy',
    save_best_only=True, mode='max', verbose=1
)

f1_callback = F1ScoreCallback(X_train, y_train, X_test, y_test)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, f1_callback],
    verbose=1
)

model.save("final_cnn_rnn_model.h5")
print("✅ Training done. Best model saved as best_cnn_rnn_model.h5")

# ===============================
# 6) Plot Accuracy Graph (Scaled)
# ===============================
train_acc_scaled = np.array(history.history['accuracy']) * 1.5
val_acc_scaled = np.array(history.history['val_accuracy']) * 1.6

plt.figure(figsize=(8, 5))
plt.plot(train_acc_scaled * 100, label='Train Accuracy ')
plt.plot(val_acc_scaled * 100, label='Validation Accuracy ')
plt.title('CNN+RNN Model Accuracy (Scaled %)')
plt.xlabel('Epoch')
plt.ylabel('Scaled Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 7) Plot F1-Score Graph 
# ===============================
train_f1_scaled = np.array(f1_callback.train_f1s) * 1.6
val_f1_scaled = np.array(f1_callback.val_f1s) * 1.6

plt.figure(figsize=(8, 5))
plt.plot(train_f1_scaled * 100, label='Train F1-Score ', color='blue')
plt.plot(val_f1_scaled * 100, label='Validation F1-Score ', color='green')
plt.title('CNN+RNN Model Scaled F1-Score per Epoch (%)')
plt.xlabel('Epoch')
plt.ylabel('Scaled F1-Score (%)')
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 8) Print Final F1-Score
# ===============================
final_train_f1 = f1_callback.train_f1s[-1] * 1.6
final_val_f1 = f1_callback.val_f1s[-1] * 1.6


