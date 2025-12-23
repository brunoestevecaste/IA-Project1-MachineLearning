import pandas as pd

def submit(model, X_train_scaled, y_train, X_test_scaled, submission_path):
    
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    pd.DataFrame({
        'ID': range(len(predictions)), 
        'label': predictions
    }).to_csv(submission_path, index=False)
    
    print(f"\n Submission guardado en: {submission_path}")
    
    print(f"\nDistribución de predicciones:")
    print(pd.Series(predictions).value_counts().sort_index())
    
    return predictions