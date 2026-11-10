import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def clean_columns(df):
    """
    Cleans DataFrame columns by removing whitespace and the BOM mark if present.
    """
    df = df.copy()
    df.columns = df.columns.str.replace(r'^\ufeff', '', regex=True).str.strip()
    return df

def engineer_features(df):
    """
    Performs feature engineering for UNSW-NB15 dataset.
    Creates 5 custom features:
      1. total_bytes = sbytes + dbytes
      2. byte_ratio = sbytes / (dbytes + 1)
      3. packet_ratio = spkts / (dpkts + 1)
      4. ttl_difference = abs(sttl - dttl)
      5. load_ratio = sload / (dload + 1)
    """
    df = df.copy()
    df['total_bytes'] = df['sbytes'] + df['dbytes']
    df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1.0)
    df['packet_ratio'] = df['spkts'] / (df['dpkts'] + 1.0)
    df['ttl_difference'] = (df['sttl'] - df['dttl']).abs()
    df['load_ratio'] = df['sload'] / (df['dload'] + 1.0)
    return df

def fit_preprocessors(df_train, categorical_cols=['proto', 'service', 'state'], target_col='label', drop_cols=['id', 'attack_cat']):
    """
    Fits OrdinalEncoder and StandardScaler on the training data.
    Computes medians and modes for handling missing values in future uploads.
    """
    df = clean_columns(df_train)
    df = engineer_features(df)
    
    # Identify target and drop columns
    columns_to_drop = [col for col in drop_cols if col in df.columns]
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=columns_to_drop + [target_col])
    else:
        y = None
        X = df.drop(columns=columns_to_drop)
        
    all_features = list(X.columns)
    
    # Missing values imputation strategies
    medians = {}
    modes = {}
    
    numeric_cols = [col for col in all_features if col not in categorical_cols]
    
    for col in numeric_cols:
        medians[col] = float(X[col].median())
        
    for col in categorical_cols:
        if col in X.columns:
            modes[col] = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
            
    # Apply imputation to X before fitting
    X_imputed = X.copy()
    for col in numeric_cols:
        X_imputed[col] = X_imputed[col].fillna(medians[col])
    for col in categorical_cols:
        if col in X_imputed.columns:
            X_imputed[col] = X_imputed[col].fillna(modes[col])
            
    # Encoders
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if all(col in X_imputed.columns for col in categorical_cols):
        encoder.fit(X_imputed[categorical_cols])
        X_encoded_cats = encoder.transform(X_imputed[categorical_cols])
        X_imputed[categorical_cols] = X_encoded_cats
        
    # Scaling
    scaler = StandardScaler()
    if numeric_cols:
        scaler.fit(X_imputed[numeric_cols])
        
    preprocessors = {
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'drop_cols': drop_cols,
        'target_col': target_col,
        'all_features': all_features,
        'medians': medians,
        'modes': modes,
        'encoder': encoder,
        'scaler': scaler
    }
    
    return preprocessors

def transform_data(df, preprocessors):
    """
    Transforms any DataFrame (train, test, or dashboard batch upload)
    using the fitted preprocessors.
    """
    df = clean_columns(df)
    df = engineer_features(df)
    
    target_col = preprocessors['target_col']
    drop_cols = preprocessors['drop_cols']
    categorical_cols = preprocessors['categorical_cols']
    numeric_cols = preprocessors['numeric_cols']
    medians = preprocessors['medians']
    modes = preprocessors['modes']
    encoder = preprocessors['encoder']
    scaler = preprocessors['scaler']
    all_features = preprocessors['all_features']
    
    # Save target if present
    y = None
    if target_col in df.columns:
        y = df[target_col].values
        
    # Extract only features that were in the training set
    # Ensure they exist in df (fill missing columns with NaN)
    X = pd.DataFrame(index=df.index)
    for col in all_features:
        if col in df.columns:
            X[col] = df[col]
        else:
            # Recreate missing engineered/original columns
            X[col] = np.nan
            
    # Impute missing values
    for col in numeric_cols:
        X[col] = X[col].fillna(medians.get(col, 0.0))
    for col in categorical_cols:
        X[col] = X[col].fillna(modes.get(col, 'unknown'))
        
    # Label encode categorical features
    X[categorical_cols] = encoder.transform(X[categorical_cols])
    
    # Scale numeric features
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    
    return X, y

def save_preprocessors(preprocessors, filepath):
    """
    Saves preprocessors dictionary to a file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessors, f)
    print(f"Preprocessors saved successfully to {filepath}")

def load_preprocessors(filepath):
    """
    Loads preprocessors dictionary from a file.
    """
    with open(filepath, 'rb') as f:
        preprocessors = pickle.load(f)
    return preprocessors

if __name__ == "__main__":
    # Quick sanity check
    print("Testing Preprocessing Module...")
    train_df = pd.read_csv("UNSW_NB15_training-set.csv", nrows=100)
    preprocessors = fit_preprocessors(train_df)
    X_train, y_train = transform_data(train_df, preprocessors)
    print("Preprocessed X shape:", X_train.shape)
    print("Preprocessed y shape:", y_train.shape if y_train is not None else None)
    print("Success!")
