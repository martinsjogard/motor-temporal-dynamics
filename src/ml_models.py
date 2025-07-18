"""
Train and evaluate classical machine learning models on synthetic EEG/MEG/behavioral data.
Includes Ridge, Lasso, Random Forest, and XGBoost with cross-validation, grid search,
SHAP interpretation, and metric reporting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["subject", "session"])
    X = df.drop(columns=["behavior_score"])
    y = df["behavior_score"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_ridge(X_train, y_train):
    params = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = GridSearchCV(Ridge(), params, cv=5, scoring="neg_mean_squared_error")
    ridge.fit(X_train, y_train)
    return ridge

def train_lasso(X_train, y_train):
    params = {"alpha": [0.01, 0.1, 1.0, 10.0]}
    lasso = GridSearchCV(Lasso(), params, cv=5, scoring="neg_mean_squared_error")
    lasso.fit(X_train, y_train)
    return lasso

def train_rf(X_train, y_train):
    params = {"n_estimators": [100, 200],
              "max_depth": [None, 10, 20],
              "min_samples_split": [2, 5]}
    rf = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5)
    rf.fit(X_train, y_train)
    return rf

def train_xgb(X_train, y_train):
    params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }
    xgb = GridSearchCV(XGBRegressor(random_state=42), params, cv=5)
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(model, X_test, y_test, model_name, save_dir="logs"):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    print(f"\n[{model_name.upper()}]")
    print(f"R2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{model_name}_metrics.txt"), "w") as f:
        f.write(f"R2: {r2:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}")

    return preds

def plot_shap(model, X_train, X_test, model_name, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    explainer = shap.Explainer(model.best_estimator_, X_train)
    shap_values = explainer(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_shap_summary.png"))
    plt.close()

def run_all_models(csv_path="data/synthetic_motor_data.csv"):
    X_train, X_test, y_train, y_test = load_and_prepare_data(csv_path)

    models = {
        "ridge": train_ridge(X_train, y_train),
        "lasso": train_lasso(X_train, y_train),
        "rf": train_rf(X_train, y_train),
        "xgb": train_xgb(X_train, y_train)
    }

    results = {}
    for name, model in models.items():
        preds = evaluate_model(model, X_test, y_test, model_name=name)
        if name in ["rf", "xgb"]:
            plot_shap(model, X_train, X_test, model_name=name)
        results[name] = model

    return results

if __name__ == "__main__":
    run_all_models()
