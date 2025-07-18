"""
Train a deep learning model (MLP) on EEG/MEG/behavioral features to predict motor learning outcomes.
Includes architecture design, training, validation, saving, and plotting using TensorFlow/Keras.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["subject", "session", "behavior_score"], errors="ignore")
    y = df["behavior_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns

def build_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))  # regression output
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

def plot_training_history(history, save_path="figures/dl_training_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def evaluate_dl_model(model, X_test, y_test, save_path="logs/dl_metrics.txt"):
    preds = model.predict(X_test).flatten()
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    print(f"\n[DEEP LEARNING]")
    print(f"R2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(f"R2: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}")

def train_dl_model(csv_path="data/synthetic_motor_data.csv"):
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(csv_path)

    model = build_mlp_model(input_dim=X_train.shape[1])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("models/best_dl_model.h5", save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )

    plot_training_history(history)
    evaluate_dl_model(model, X_test, y_test)

    return model

if __name__ == "__main__":
    train_dl_model()
