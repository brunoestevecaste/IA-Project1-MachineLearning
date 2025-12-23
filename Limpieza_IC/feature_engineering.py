import numpy as np
import pandas as pd

def create_features(df):

    df = df.copy()

    numeric_cols = ['age', 'thalach', 'trestbps', 'chol', 'cp', 'sex', 'exang', 'oldpeak']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['hr_achievement'] = df['thalach'] / (220 - df['age'])
    df['bp_risk'] = (df['trestbps'] - 120).abs()
    df['chol_risk'] = np.where(df['chol'] > 200, 1, 0)
    df['age_chol_interaction'] = df['age'] * df['chol'] / 1000
    df['chest_pain_severity'] = 4 - df['cp']
    df['combined_risk'] = (
        (df['age'] > 55).astype(int) + 
        (df['sex'] == 1).astype(int) +
        (df['chol'] > 240).astype(int) +
        (df['trestbps'] > 140).astype(int) +
        df['exang']
    )
    df['oldpeak_age_adjusted'] = df['oldpeak'] * (df['age'] / 50)
    
    return df
