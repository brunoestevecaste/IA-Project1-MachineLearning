import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def median_mode_imputation_minus_nine(df_train, df_test):

    df_train = df_train.copy()
    df_test = df_test.copy()

    fill_values = {}
    for col in df_train.columns:
        if col != 'label':
            temp_col = df_train[col].replace([-9, '-9', -9.0, '-9.0', '?'], np.nan)
            temp_col = pd.to_numeric(temp_col, errors='coerce')
            fill_values[col] = temp_col.median()

    for df in [df_train, df_test]:
        for col in df.columns:
            if col != 'label':
                df[col] = df[col].replace([-9, '-9', -9.0, '-9.0'], fill_values[col])

    return df_train, df_test

def median_mode_imputation_nan(df_train, df_test):

    df_train = df_train.copy()
    df_test = df_test.copy()

    fill_values = {}
    for col in df_train.columns:
        if col != 'label':
            temp_col = pd.to_numeric(df_train[col].replace('?', np.nan), errors='coerce')
            fill_values[col] = temp_col.median()

    for df in [df_train, df_test]:
        for col in df.columns:
            if col != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace('?', np.nan, inplace=True)
        
        for col in df.columns:
            if col != 'label':
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(fill_values[col])

    return df_train, df_test

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

def run_pipeline(train, test, label_vars, onehot_vars):

    train_clean, test_clean = median_mode_imputation_minus_nine(train, test)
    train_imputed, test_imputed = median_mode_imputation_nan(train_clean, test_clean)

    drop_cols = ['ca', 'thal']

    X_train = train_imputed.drop(['label'] + drop_cols, axis=1, errors='ignore').copy()
    y_train = train_imputed['label'].copy()

    X_test = test_imputed.drop(drop_cols, axis=1, errors='ignore').copy()

    X_train = create_features(X_train)
    X_test = create_features(X_test)

    for col in label_vars:
        if col in X_train.columns:
            le = LabelEncoder()
            all_values = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_values)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

    for col in onehot_vars:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

    X_train = pd.get_dummies(X_train, columns=onehot_vars, prefix=onehot_vars)
    X_test = pd.get_dummies(X_test, columns=onehot_vars, prefix=onehot_vars)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.reindex(columns=X_test.columns, fill_value=0)
    X_test = X_test[X_train.columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled