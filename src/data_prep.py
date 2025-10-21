"""
Data preparation utilities for ML tutorial series.

Functions for:
- Loading data from databases
- Cleaning and preprocessing
- Feature scaling
- Train/test splits
- Handling missing values
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_from_sql(query, connection_string):
    """
    Load data from SQL database.
    
    Args:
        query: SQL query string
        connection_string: Database connection string
        
    Returns:
        pandas DataFrame
    """
    # Implementation coming in Tutorial 2
    pass


def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in dataset.
    
    Args:
        df: pandas DataFrame
        strategy: 'median', 'mean', or 'drop'
        
    Returns:
        Cleaned DataFrame
    """
    # Implementation coming in Tutorial 2
    pass


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train/test split with validation set option.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Scaled X_train, X_test, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
