"""
Linear Mixed-Effects Modeling of Behavioral Data with EEG/MEG Features.
Implemented using statsmodels for within-subject modeling across sessions.
"""

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

def load_data(csv_path="data/synthetic_motor_data.csv"):
    return pd.read_csv(csv_path)

def fit_mixed_model(df):
    """
    Random intercept model: subject as random effect
    Fixed effects: theta_power, spindle_rate, so_sp_coupling, aec_mean, session
    """
    print("\nFitting mixed-effects model...")
    md = smf.mixedlm("behavior_score ~ theta_power + spindle_rate + so_sp_coupling + aec_mean + session",
                     df, groups=df["subject"])
    mdf = md.fit(reml=False)
    print(mdf.summary())
    return mdf

def plot_residuals(mdf, df, save_path="figures/mixed_model_residuals.png"):
    df["resid"] = mdf.resid
    plt.figure(figsize=(10, 5))
    for session in df["session"].unique():
        subset = df[df["session"] == session]
        plt.hist(subset["resid"], bins=30, alpha=0.5, label=session)
    plt.title("Residuals by Session")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def run_mixed_model_pipeline(csv_path="data/synthetic_motor_data.csv"):
    df = load_data(csv_path)
    mdf = fit_mixed_model(df)
    plot_residuals(mdf, df)
    return mdf

if __name__ == "__main__":
    run_mixed_model_pipeline()
