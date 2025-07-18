# src/data_generation.py

import numpy as np
import pandas as pd

def generate_synthetic_data(n_subjects=30, n_days=3):
    np.random.seed(42)
    data = []
    for subj in range(n_subjects):
        baseline_perf = np.random.normal(50, 5)
        for day in range(n_days):
            so_power = np.random.normal(1 + day*0.2, 0.1)
            sp_density = np.random.normal(2 + day*0.3, 0.2)
            beta_rest = np.random.normal(5 - day*0.4, 0.3)
            theta_power = np.random.normal(3 + day*0.1, 0.2)
            so_sp_coupling = np.random.uniform(0.1, 1.0)
            behavior = baseline_perf + (day * np.random.normal(5, 1)) + 0.5 * sp_density - 0.3 * beta_rest + 0.2 * theta_power
            data.append({
                'subject': f'S{subj}', 'day': day,
                'SO_power': so_power, 'SP_density': sp_density, 'beta_rest': beta_rest,
                'theta_power': theta_power, 'SO_SP_coupling': so_sp_coupling,
                'behavior': behavior
            })
    return pd.DataFrame(data)

