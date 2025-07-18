# src/feature_extraction.py

def zscore_features(df, feature_cols):
    df_z = df.copy()
    for col in feature_cols:
        mean = df_z[col].mean()
        std = df_z[col].std()
        df_z[col + '_z'] = (df_z[col] - mean) / std
    return df_z
