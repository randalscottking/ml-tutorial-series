"""
Model building utilities for ML tutorial series.

Functions for:
- Building classification models
- Training and prediction
- Feature importance
- Model persistence
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib


def build_logistic_model(C=1.0, random_state=42):
    """
    Create logistic regression model.
    
    Args:
        C: Regularization strength
        random_state: Random seed
        
    Returns:
        Untrained LogisticRegression model
    """
    return LogisticRegression(C=C, random_state=random_state, max_iter=1000)


def build_random_forest(n_estimators=100, max_depth=None, random_state=42):
    """
    Create random forest classifier.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        
    Returns:
        Untrained RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )


def train_model(model, X_train, y_train):
    """
    Train a model.
    
    Args:
        model: Scikit-learn model
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Args:
        model: Trained scikit-learn model
        filepath: Path to save model
    """
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def get_feature_importance(model, feature_names):
    """
    Get feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and importance scores
    """
    import pandas as pd
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    return importance_df.sort_values('importance', ascending=False)
