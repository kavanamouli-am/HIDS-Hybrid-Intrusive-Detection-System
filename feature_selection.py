import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_features(X_train, y_train, n_features=20, n_estimators=50, random_state=42, n_jobs=-1):
    """
    Fits a RandomForestClassifier to extract Gini feature importances
    and selects the top `n_features` features.
    """
    print(f"Fitting RandomForestClassifier on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf.fit(X_train, y_train)
    
    # Extract importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Sort all features
    all_sorted_features = [
        (X_train.columns[i], float(importances[i]))
        for i in indices
    ]
    
    # Select top N
    selected = [X_train.columns[i] for i in indices[:n_features]]
    print(f"Successfully selected top {n_features} features.")
    
    return selected, all_sorted_features

def save_selected_features(features, filepath):
    """
    Saves selected features list to a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(features, f, indent=4)
    print(f"Selected features list saved to {filepath}")

def load_selected_features(filepath):
    """
    Loads selected features list from a JSON file.
    """
    with open(filepath, 'r') as f:
        features = json.load(f)
    return features

if __name__ == "__main__":
    print("Testing Feature Selection Module...")
    # Create dummy data
    X = pd.DataFrame(np.random.randn(100, 30), columns=[f"feat_{i}" for i in range(30)])
    y = np.random.randint(0, 2, size=100)
    
    selected, full_ranking = select_features(X, y, n_features=10)
    print("Selected Top 10 features:", selected)
    print("Total Ranked Features list size:", len(full_ranking))
    
    test_file = "models/test_selected_features.json"
    save_selected_features(selected, test_file)
    loaded = load_selected_features(test_file)
    print("Loaded matches selected:", loaded == selected)
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    print("Success!")
