import pandas as pd
from src.ml_models import train_models

df = pd.read_csv("data/synthetic_motor_data.csv")
results = train_models(df)
print("Model results:", results)
