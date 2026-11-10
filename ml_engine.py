import os
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_ml_engine(X_train, y_train, tune_hyperparameters=True, n_jobs=-1):
    """
    Trains an XGBClassifier on the selected top features.
    Optionally performs GridSearchCV to find optimal hyperparameters.
    """
    print("Initializing XGBoost training...")
    
    if tune_hyperparameters:
        print("Running GridSearchCV for hyperparameter tuning...")
        # Define a focused, efficient grid to prevent excessively long training times
        param_grid = {
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [50, 100]
        }
        
        xgb = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=n_jobs
        )
        
        # Use 3-fold CV for speed, scoring by ROC-AUC
        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            n_jobs=n_jobs
        )
        
        grid.fit(X_train, y_train)
        print(f"GridSearchCV complete. Best Parameters: {grid.best_params_}")
        best_model = grid.best_estimator_
    else:
        # Default high-performance configuration
        print("Training with default high-performance hyperparameters...")
        best_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42,
            n_jobs=n_jobs
        )
        best_model.fit(X_train, y_train)
        
    return best_model

def predict_ml_prob(model, X):
    """
    Predicts the probability of attack (class 1) using the trained XGBoost model.
    """
    return model.predict_proba(X)[:, 1]

def save_ml_model(model, filepath):
    """
    Saves the trained model to a pickle file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"ML Model saved to {filepath}")

def load_ml_model(filepath):
    """
    Loads the trained model from a pickle file.
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    print("Testing ML Engine Module...")
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    X = pd.DataFrame(np.random.randn(200, 10), columns=[f"feat_{i}" for i in range(10)])
    y = np.random.randint(0, 2, size=200)
    
    # Train with tuning on small set
    model = train_ml_engine(X, y, tune_hyperparameters=True)
    probs = predict_ml_prob(model, X)
    print("Predictions shape:", probs.shape)
    
    # Test save/load
    test_file = "models/test_xgb.pkl"
    save_ml_model(model, test_file)
    loaded_model = load_ml_model(test_file)
    loaded_probs = predict_ml_prob(loaded_model, X)
    print("Loaded model check: Max diff =", np.max(np.abs(probs - loaded_probs)))
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    print("Success!")
