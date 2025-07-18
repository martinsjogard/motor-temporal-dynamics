# src/preprocessing.py

def clean_data(df):
    df_clean = df.dropna()
    return df_clean
