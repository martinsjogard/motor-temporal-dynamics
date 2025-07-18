# scripts/generate_data.py
from src.data_generation import generate_synthetic_data

df = generate_synthetic_data()
df.to_csv("data/synthetic_dataset.csv", index=False)
print("Synthetic dataset saved.")
