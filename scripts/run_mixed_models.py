# scripts/run_mixed_models.py
import pandas as pd
from src.modeling import run_mixed_model

df = pd.read_csv("data/synthetic_dataset.csv")
result = run_mixed_model(df)
print(result.summary())
