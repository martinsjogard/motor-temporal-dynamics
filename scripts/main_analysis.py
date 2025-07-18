# scripts/main_analysis.py
import pandas as pd
from src.preprocessing import clean_data
from src.feature_extraction import zscore_features
from src.modeling import run_mixed_model
from src.plotting import plot_learning_curve, plot_predictors

# Load and preprocess
raw_df = pd.read_csv("data/synthetic_dataset.csv")
df = clean_data(raw_df)

# Feature engineering
features = ['SP_density', 'beta_rest', 'theta_power', 'SO_SP_coupling']
df = zscore_features(df, features)

# Run model
result = run_mixed_model(df)
print(result.summary())

# Visualize
plot_learning_curve(df)
plot_predictors(df)
