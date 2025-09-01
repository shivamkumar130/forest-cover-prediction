import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import yaml
import os
import joblib

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def plot_correlation_matrix(df, save_path):
    """Plot and save correlation matrix"""
    plt.figure(figsize=(16, 14))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    """Plot and save feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 10))
        plt.title('Feature Importance')
        plt.bar(range(20), importance[indices][:20])  # Top 20 features
        plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_importance.png'))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def plot_class_distribution(y, save_path):
    """Plot and save class distribution"""
    plt.figure(figsize=(10, 6))
    y.value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Cover Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_distribution.png'))
    plt.close()

def plot_numeric_features_distributions(df, save_path):
    """Plot distributions of numeric features"""
    numeric_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                        'Horizontal_Distance_To_Fire_Points']
    
    n_cols = 3
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_features):
        sns.histplot(data=df, x=feature, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'numeric_features_distributions.png'))
    plt.close()

def create_visualizations(df, model, evaluation_results, feature_names):
    """Create all visualizations"""
    config = load_config()
    figures_path = config['paths']['figures']
    
 
    os.makedirs(figures_path, exist_ok=True)
    

    plot_correlation_matrix(df, figures_path)
    plot_feature_importance(model, feature_names, figures_path)
    plot_confusion_matrix(evaluation_results['y_test'], evaluation_results['y_pred'], figures_path)
    plot_class_distribution(df['Cover_Type'], figures_path)
    plot_numeric_features_distributions(df, figures_path)

if __name__ == "__main__":
    config = load_config()
    
    df = pd.read_csv(config['data']['processed_path'])

    model = joblib.load(os.path.join(config['paths']['models'], 'random_forest_model.pkl'))
   
    from sklearn.model_selection import train_test_split
    processed_df = pd.read_csv(config['data']['processed_path'])
    X = processed_df.drop('Cover_Type', axis=1)
    y = processed_df['Cover_Type']
    
    scaler = joblib.load('models/scaler.pkl')
    X_scaled = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=config['model']['test_size'], 
        random_state=config['model']['random_state'],
        stratify=y
    )
    
    y_pred = model.predict(X_test)
    
    evaluation_results = {
        'y_test': y_test,
        'y_pred': y_pred
    }
    create_visualizations(processed_df, model, evaluation_results, X.columns)
    
    print("Visualizations created and saved!")