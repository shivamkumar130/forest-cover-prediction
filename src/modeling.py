import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def train_models(X_train, y_train, n_estimators=200, max_depth=20, random_state=42):
    """Train multiple models and return the best one"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'),
        'Support Vector Machine': SVC(random_state=random_state, probability=True, class_weight='balanced'),
        'XGBoost': XGBClassifier(random_state=random_state, scale_pos_weight=1)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}
    
    for name, model in models.items():
       
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = cv_scores.mean()
        
        results[name] = {
            'model': model,
            'cv_score': mean_score
        }
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = name
    
    print(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
    
    return best_model, results, best_model_name

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and generate reports"""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def save_model(model, model_path):
    """Save the trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

if __name__ == "__main__":
    config = load_config()
    

    processed_df = pd.read_csv(config['data']['processed_path'])
    X = processed_df.drop('Cover_Type', axis=1)
    y = processed_df['Cover_Type']
    
    scaler = joblib.load('models/scaler.pkl')
    X_scaled = scaler.transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state'],
        stratify=y
    )

    best_model, all_models, best_model_name = train_models(
        X_train, y_train,
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        random_state=config['model']['random_state']
    )
    
    results = evaluate_model(best_model, X_test, y_test)
    
    model_filename = "random_forest_model.pkl"
    save_model(best_model, os.path.join(config['paths']['models'], model_filename))
    
    os.makedirs(config['paths']['reports'], exist_ok=True)
    with open(os.path.join(config['paths']['reports'], 'model_performance.txt'), 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(results['report'])
    
    print("Model training and evaluation completed!")