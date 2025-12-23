import pandas as pd
from sklearn.model_selection import cross_val_score

def evaluate(model, X_train_scaled, y_train, cv):
    
    accuracy_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    mean_accuracy = accuracy_scores.mean()
    std_accuracy = accuracy_scores.std()
    
    print(f"Accuracy Media (CV): {mean_accuracy:.4f}")
    print(f"Rango de Rendimiento (CV): {mean_accuracy:.4f} +/- {std_accuracy:.4f} \n")
    
    return mean_accuracy