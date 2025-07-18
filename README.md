# motor-temporal-dynamics
Temporal Dynamics of Motor Learning

This project uses a longitudinal neurocognitive dataset to model behavioral improvements across days using neural markers from rest, sleep, and task EEG/MEG data. It demonstrates some new approaches to statistical modeling using MEG and EEG as well as interpretable data visualizations. The collected data consists of:

Simultaneous MEG & EEG:
  Baseline day: 5 min resting-state
                + 90 mins sleep
  Task day:     5 min resting-state
                + motor sequence task (learning)
                  12 online (tapping) periods, 30s each
                  11 offline (rest) periods, 30s each
                + 5 min resting-state
                + 90 min sleep
                + motor sequence task (testing)
                  12 online (tapping) periods, 30s each
                  11 offline (rest) periods, 30s each
  Behavioral data:
    Motor-sequence task: speed & total sequences per trial
    Demographics: age, sex, educational level
  


###  Features
- Pseudosynthetic dataset from a 3-day motor sequence learning study
- EEG/MEG feature generation (SOs, spindles, beta power)
- Behavioral performance over time
- Mixed-effects modeling of behavioral gains using EEG/MEG
- Publication-quality figures and interpretable summaries

### How to Run
```bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/main_analysis.py
```

### Requirements
- Python 3.10+
- pandas, numpy, statsmodels, seaborn, matplotlib, scikit-learn

---

# Root directory structure
motor-temporal-dynamics/
├── README.md
├── data/
│   └── synthetic_dataset.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/
│   ├── main_analysis.py
│   ├── generate_data.py
│   ├── run_mixed_models.py
│   └── visualize_results.py
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── modeling.py
│   └── plotting.py
├── tests/
│   └── test_modeling.py
└── requirements.txt

