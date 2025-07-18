import pandas as pd
from src.ml_models import run_all_models
from src.dl_models import train_dl_model
from src.mixed_models import run_mixed_model_pipeline

def test_pipeline():
    print("Loading synthetic data...")
    df = pd.read_csv("data/synthetic_motor_data.csv")
    assert not df.empty, "Data failed to load"

    print("Running ML models...")
    results = run_all_models()
    assert "ridge" in results, "Ridge not in ML results"

    print("Running DL model...")
    model = train_dl_model()
    assert model is not None, "DL model failed"

    print("Running mixed models...")
    run_mixed_model_pipeline()
    print("Test complete.")

if __name__ == "__main__":
    test_pipeline()
