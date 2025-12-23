import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

INTEGER_COLUMNS = ['sex', 'cp', 'restecg', 'fbs', 'exang', 'slope', 'ca', 'thal']
FLOAT_COLUMNS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
ALL_COLUMNS = INTEGER_COLUMNS + FLOAT_COLUMNS

def categorize_minus_nine(df_train, df_test):

    df_train = df_train.copy()
    df_test = df_test.copy()

    question_masks = {}
    for df_name, df in [('train', df_train), ('test', df_test)]:
        question_masks[df_name] = {}
        for col in df.columns:
            if col != 'label':
                question_masks[df_name][col] = df[col] == '?'

    for df_name, df in [('train', df_train), ('test', df_test)]:
        for col in df.columns:
            if col != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.replace([-9, -9.0], np.nan)
        
        for col in df.columns:
            if col != 'label':
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(-1)
                elif df[col].dtype == 'object':
                    df[col] = df[col].fillna('Otros')
                
                df.loc[question_masks[df_name][col], col] = '?'

    return df_train, df_test

def healthy_values_minus_nine(df_train, df_test):

    df_train = df_train.copy()
    df_test = df_test.copy()
    
    question_masks = {}
    for df_name, df in [('train', df_train), ('test', df_test)]:
        question_masks[df_name] = {}
        for col in df.columns:
            if col != 'label':
                question_masks[df_name][col] = df[col] == '?'
    
    valores_sanos = {
        'sex': 1, 'cp': 4, 'fbs': 0, 'restecg': 0, 'exang': 0, 
        'slope': 1, 'ca': 0, 'thal': 3, 
        'trestbps': 100, 'chol': 180, 'oldpeak': 0
    }
    
    for df_name, df in [('train', df_train), ('test', df_test)]:
        for col in df.columns:
            if col != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.replace([-9, -9.0], np.nan)
        
        if 'age' in df.columns:
            df.loc[df['age'].isna(), 'age'] = 35
        
        if 'thalach' in df.columns and 'age' in df.columns:
            mask_thalach_na = df['thalach'].isna()
            if mask_thalach_na.any():
                df.loc[mask_thalach_na, 'thalach'] = 220 - df.loc[mask_thalach_na, 'age']
        
        for col, val in valores_sanos.items():
            if col in df.columns:
                df.loc[df[col].isna(), col] = val
        
        for col in df.columns:
            if col != 'label':
                df.loc[question_masks[df_name][col], col] = '?'

    return df_train, df_test

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

def knn_imputation_minus_nine(df_train, df_test):

    df_train = df_train.copy()
    df_test = df_test.copy()

    question_masks = {}
    for df_name, df in [('train', df_train), ('test', df_test)]:
        question_masks[df_name] = {}
        for col in df.columns:
            if col != 'label':
                question_masks[df_name][col] = df[col] == '?'

    for col in df_train.columns:
        if col != 'label':
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    
    for col in df_test.columns:
        if col != 'label':
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    QUESTION_MARKER = -99999
    for col in ALL_COLUMNS:
        if col in df_train.columns:
            df_train.loc[question_masks['train'][col], col] = QUESTION_MARKER
        if col in df_test.columns:
            df_test.loc[question_masks['test'][col], col] = QUESTION_MARKER

    df_train = df_train.replace([-9, -9.0], np.nan)
    df_test = df_test.replace([-9, -9.0], np.nan)

    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(df_train[ALL_COLUMNS])
    
    df_train[ALL_COLUMNS] = imputer.transform(df_train[ALL_COLUMNS])
    df_test[ALL_COLUMNS] = imputer.transform(df_test[ALL_COLUMNS])
    
    for col in ALL_COLUMNS:
        if col in df_train.columns:
            df_train.loc[question_masks['train'][col], col] = '?'
        if col in df_test.columns:
            df_test.loc[question_masks['test'][col], col] = '?'
    
    for df in [df_train, df_test]:
        for col in INTEGER_COLUMNS:
            if col in df.columns:
                mask_not_question = df[col] != '?'
                df.loc[mask_not_question, col] = pd.to_numeric(df.loc[mask_not_question, col], errors='coerce').round()
                if col == 'sex':
                    df.loc[mask_not_question, col] = df.loc[mask_not_question, col].astype(float).astype('Int64')
                else:
                    df.loc[mask_not_question, col] = df.loc[mask_not_question, col].astype(int).astype(str)
        
        for col in FLOAT_COLUMNS:
            if col in df.columns:
                mask_not_question = df[col] != '?'
                df.loc[mask_not_question, col] = pd.to_numeric(df.loc[mask_not_question, col], errors='coerce')

    return df_train, df_test


def iterative_imputation_minus_nine(df_train, df_test):

    df_train = df_train.copy()
    df_test = df_test.copy()

    question_masks = {}
    for df_name, df in [('train', df_train), ('test', df_test)]:
        question_masks[df_name] = {}
        for col in df.columns:
            if col != 'label':
                question_masks[df_name][col] = df[col] == '?'

    for col in df_train.columns:
        if col != 'label':
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    
    for col in df_test.columns:
        if col != 'label':
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    QUESTION_MARKER = -99999
    for col in ALL_COLUMNS:
        if col in df_train.columns:
            df_train.loc[question_masks['train'][col], col] = QUESTION_MARKER
        if col in df_test.columns:
            df_test.loc[question_masks['test'][col], col] = QUESTION_MARKER

    df_train = df_train.replace([-9, -9.0], np.nan)
    df_test = df_test.replace([-9, -9.0], np.nan)

    imputer = IterativeImputer(max_iter=100, random_state=42)
    imputer.fit(df_train[ALL_COLUMNS])
    
    df_train[ALL_COLUMNS] = imputer.transform(df_train[ALL_COLUMNS])
    df_test[ALL_COLUMNS] = imputer.transform(df_test[ALL_COLUMNS])
    
    for col in ALL_COLUMNS:
        if col in df_train.columns:
            df_train.loc[question_masks['train'][col], col] = '?'
        if col in df_test.columns:
            df_test.loc[question_masks['test'][col], col] = '?'

    for df in [df_train, df_test]:
        for col in INTEGER_COLUMNS:
            if col in df.columns:
                mask_not_question = df[col] != '?'
                df.loc[mask_not_question, col] = pd.to_numeric(df.loc[mask_not_question, col], errors='coerce').round()
                if col == 'sex':
                    df.loc[mask_not_question, col] = df.loc[mask_not_question, col].astype(float).astype('Int64')
                else:
                    df.loc[mask_not_question, col] = df.loc[mask_not_question, col].astype(int).astype(str)
        
        for col in FLOAT_COLUMNS:
            if col in df.columns:
                mask_not_question = df[col] != '?'
                df.loc[mask_not_question, col] = pd.to_numeric(df.loc[mask_not_question, col], errors='coerce')

    return df_train, df_test

# ------------------------------------------------------------------------------

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

def knn_imputation_nan(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    for col in df_train.columns:
        if col != 'label':
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    
    for col in df_test.columns:
        if col != 'label':
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    df_train = df_train.replace('?', np.nan)
    df_test = df_test.replace('?', np.nan)

    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(df_train[ALL_COLUMNS])
    
    df_train[ALL_COLUMNS] = imputer.transform(df_train[ALL_COLUMNS])
    df_test[ALL_COLUMNS] = imputer.transform(df_test[ALL_COLUMNS])
    
    for df in [df_train, df_test]:
        for col in INTEGER_COLUMNS:
            if col in df.columns:
                df[col] = df[col].round()
                if col == 'sex':
                    df[col] = df[col].astype(float).astype('Int64')
                else:
                    df[col] = df[col].astype(int).astype(str)
        
        for col in FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(float)

    return df_train, df_test

def iterative_imputation_nan(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    for col in df_train.columns:
        if col != 'label':
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    
    for col in df_test.columns:
        if col != 'label':
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    df_train = df_train.replace('?', np.nan)
    df_test = df_test.replace('?', np.nan)

    imputer = IterativeImputer(max_iter=100, random_state=42)
    imputer.fit(df_train[ALL_COLUMNS])
    
    df_train[ALL_COLUMNS] = imputer.transform(df_train[ALL_COLUMNS])
    df_test[ALL_COLUMNS] = imputer.transform(df_test[ALL_COLUMNS])
    
    for df in [df_train, df_test]:
        for col in INTEGER_COLUMNS:
            if col in df.columns:
                df[col] = df[col].round()
                if col == 'sex':
                    df[col] = df[col].astype(float).astype('Int64')
                else:
                    df[col] = df[col].astype(int).astype(str)
        
        for col in FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(float)

    return df_train, df_test