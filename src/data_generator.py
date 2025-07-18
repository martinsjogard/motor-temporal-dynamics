"""
Generates realistic synthetic EEG, MEG, and behavioral data across sessions and participants.
Each subject performs a motor sequence task across Adapt, Baseline, and Learning days.
Includes sleep-related features (e.g., SO, spindle rate), resting-state FC, and task performance.
"""

import numpy as np
import pandas as pd

def generate_synthetic_data(n_subjects=60, random_state=42):
    np.random.seed(random_state)
    sessions = ["Adapt", "Baseline", "Learning"]
    records = []

    for subj in range(n_subjects):
        subject_id = f"subj_{subj+1:03d}"
        base_skill = np.random.normal(50, 8)
        learning_slope = np.random.normal(6, 1.5)

        for i, session in enumerate(sessions):
            # Simulate behavior
            behavior_score = base_skill + i * learning_slope + np.random.normal(0, 3)

            # Simulate EEG: theta power, spindle rate, SO-SP coupling
            theta_power = np.random.normal(loc=4.0 + 0.3 * i, scale=0.5)
            spindle_rate = np.random.normal(loc=2.0 + 0.2 * i, scale=0.3)
            so_sp_coupling = np.random.normal(loc=0.5 + 0.1 * i, scale=0.1)

            # MEG: amplitude envelope connectivity (AEC)
            aec_mean = np.random.normal(loc=0.25 + 0.05 * i, scale=0.03)
            aec_variability = np.random.normal(loc=0.04 - 0.005 * i, scale=0.01)

            # Sleep: total sleep time, NREM duration
            total_sleep = np.random.normal(loc=420, scale=40)
            nrem_proportion = np.random.uniform(0.4, 0.6)

            # Save data
            records.append({
                "subject": subject_id,
                "session": session,
                "behavior_score": behavior_score,
                "theta_power": theta_power,
                "spindle_rate": spindle_rate,
                "so_sp_coupling": so_sp_coupling,
                "aec_mean": aec_mean,
                "aec_variability": aec_variability,
                "total_sleep": total_sleep,
                "nrem_proportion": nrem_proportion
            })

    df = pd.DataFrame(records)
    return df

def expand_with_task_features(df):
    # Add motor task-derived features
    df["reaction_time"] = np.random.normal(loc=300, scale=30, size=len(df)) - 0.5 * df["theta_power"]
    df["motor_variability"] = np.random.normal(loc=20, scale=5, size=len(df)) / df["spindle_rate"]
    return df

def add_noise_and_artifacts(df):
    # Simulate noise from movement, bad sleep, etc.
    noise = np.random.normal(0, 2, len(df))
    df["behavior_score"] += noise
    df["artifact_flag"] = (noise > 2.5).astype(int)
    return df

def save_synthetic_data(df, path="data/synthetic_motor_data.csv"):
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")

if __name__ == "__main__":
    df = generate_synthetic_data()
    df = expand_with_task_features(df)
    df = add_noise_and_artifacts(df)
    save_synthetic_data(df)
