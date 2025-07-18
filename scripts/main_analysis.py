"""
Main Analysis: Full Pipeline Coordination and Explanation
This script demonstrates how all components of the project fit together,
with detailed technical justification for each design choice.
"""

from src.data_generator import generate_synthetic_data, expand_with_task_features, add_noise_and_artifacts, save_synthetic_data
from src.ml_models import run_all_models
from src.dl_models import train_dl_model
from src.mixed_models import run_mixed_model_pipeline

def main():
    print("Step 1: Data Generation")
    df = generate_synthetic_data()
    df = expand_with_task_features(df)
    df = add_noise_and_artifacts(df)
    save_synthetic_data(df)

    print("\nStep 2: Train Classical ML Models")
    # Ridge, Lasso = robust to multicollinearity and feature sparsity
    # RF and XGBoost = capture nonlinearities and feature interactions
    run_all_models()

    print("\nStep 3: Train Deep Learning Model")
    # MLP can capture higher-order interactions, but overfitting risk managed with dropout + early stopping
    train_dl_model()

    print("\nStep 4: Mixed-Effects Model")
    # Random intercept model captures subject-specific variance
    run_mixed_model_pipeline()

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
