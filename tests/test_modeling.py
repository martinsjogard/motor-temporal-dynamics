# tests/test_modeling.py
import pandas as pd
from src.data_generation import generate_synthetic_data
from src.modeling import run_mixed_model

def test_model_runs():
    df = generate_synthetic_data(n_subjects=10, n_days=2)
    result = run_mixed_model(df)
    assert result is not None
    assert hasattr(result, 'summary')
