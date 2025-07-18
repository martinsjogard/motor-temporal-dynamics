# scripts/visualize_results.py
import pandas as pd
from src.plotting import plot_learning_curve, plot_predictors

if __name__ == '__main__':
    df = pd.read_csv("data/synthetic_dataset.csv")
    plot_learning_curve(df)
    plot_predictors(df)
