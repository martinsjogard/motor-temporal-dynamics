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
project/<br />
├── README.md<br />
├── data/<br />
│   └── synthetic_dataset.csv<br />
├── notebooks/<br />
│   └── exploratory_analysis.ipynb<br />
├── scripts/<br />
│   ├── main_analysis.py<br />
│   ├── generate_data.py<br />
│   ├── run_mixed_models.py<br />
│   └── visualize_results.py<br />
├── src/<br />
│   ├── __init__.py<br />
│   ├── data_generation.py<br />
│   ├── preprocessing.py<br />
│   ├── feature_extraction.py<br />
│   ├── modeling.py<br />
│   └── plotting.py<br />
├── tests/<br />
│   └── test_modeling.py<br />
└── requirements.txt<br />
