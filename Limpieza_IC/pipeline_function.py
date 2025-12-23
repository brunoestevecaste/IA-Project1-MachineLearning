import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def run_pipeline(train, test, 
                 impute_minus_nine_func, 
                 impute_question_func,
                 label_vars,
                 onehot_vars,
                 model_params,
                 param_grid,
                 grid_search=False,
                 cv_splits=5,
                 submission=False,
                 submission_path='./datasets/submission.csv'):
    
    train_clean, test_clean = impute_minus_nine_func(train, test)
    train_imputed, test_imputed = impute_question_func(train_clean, test_clean)
    
    drop_cols = ['ca', 'thal']
    
    X_train = train_imputed.drop(['label'] + drop_cols, axis=1, errors='ignore').copy()
    y_train = train_imputed['label'].copy()
    
    X_test = test_imputed.drop(drop_cols, axis=1, errors='ignore').copy()

    import feature_engineering
    X_train = feature_engineering.create_features(X_train)
    X_test = feature_engineering.create_features(X_test)
    
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
    
    model = LogisticRegression(**model_params)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    import model_evaluation

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    if grid_search:
            
        base_model = LogisticRegression(max_iter=2000, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        best_index = grid_search.best_index_
        mean_accuracy = grid_search.cv_results_['mean_test_score'][best_index]
        std_accuracy = grid_search.cv_results_['std_test_score'][best_index]
        
        model = grid_search.best_estimator_
        accuracy = mean_accuracy 
        
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Accuracy Media (CV): {mean_accuracy:.4f}")
        print(f"Rango de Rendimiento (CV): {mean_accuracy:.4f} +/- {std_accuracy:.4f} \n")
        
    else:
        accuracy = model_evaluation.evaluate(model, X_train_scaled, y_train, cv=cv)

    import model_submission
    if submission:
        predictions = model_submission.submit(model, X_train_scaled, y_train, X_test_scaled, submission_path)
        return predictions, accuracy, X_train_scaled, X_test_scaled, y_train, model, X_train.columns
    else:
        return accuracy, X_train_scaled, X_test_scaled, y_train, model, X_train.columns