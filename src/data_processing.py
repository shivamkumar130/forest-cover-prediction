import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import yaml
import os

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path):
    """Load data from Excel file"""
    return pd.read_excel(file_path)

def preprocess_data(df):
    """Preprocess the forest cover data"""
    df = df.dropna()
    
    target_counts = df['Cover_Type'].value_counts()
    print(f"Target distribution:\n{target_counts}")
  
    X = df.drop(['Id', 'Cover_Type'], axis=1)
    y = df['Cover_Type']
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    config = load_config()
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    df = load_data(config['data']['raw_path'])
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
        X, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )
 
    processed_df = pd.concat([pd.DataFrame(X, columns=X.columns), y], axis=1)
    processed_df.to_csv(config['data']['processed_path'], index=False)
    
    print("Data processing completed successfully!")
    print(f"Original data shape: {df.shape}")
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")