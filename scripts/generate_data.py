from src.data_generator import generate_synthetic_data, save_synthetic_data

df = generate_synthetic_data()
save_synthetic_data(df, "data/synthetic_motor_data.csv")
print("Synthetic data saved.")
