# src/__init__.py

"""
Forest Cover Prediction Package

This package contains modules for data processing, modeling, and visualization
for the Forest Cover Type Prediction project.
"""

__version__ = "1.0.0"


from .data_processing import load_config, load_data, preprocess_data, split_and_scale_data
from .modeling import train_models, evaluate_model, save_model
from .visualization import (
    plot_correlation_matrix, 
    plot_feature_importance, 
    plot_confusion_matrix, 
    plot_class_distribution,
    plot_numeric_features_distributions,
    create_visualizations
)

__all__ = [
    'load_config',
    'load_data',
    'preprocess_data',
    'split_and_scale_data',
    'train_models',
    'evaluate_model',
    'save_model',
    'plot_correlation_matrix',
    'plot_feature_importance',
    'plot_confusion_matrix',
    'plot_class_distribution',
    'plot_numeric_features_distributions',
    'create_visualizations'
]


__description__ = "A machine learning package for predicting forest cover types using cartographic data"


__author__ = "Your Name"
__email__ = "your.email@example.com"

print(f"Initializing Forest Cover Prediction Package v{__version__}")