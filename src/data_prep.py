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
from sklearn.impute import SimpleImputer
import sqlalchemy


def load_from_sql(query, connection_string):
    """
    Load data from SQL database.
    
    Args:
        query: SQL query string
        connection_string: Database connection string
        
    Returns:
        pandas DataFrame
    
    Example:
        # PostgreSQL
        query = "SELECT * FROM customers WHERE active = 1"
        conn_str = "postgresql://user:password@localhost:5432/database"
        df = load_from_sql(query, conn_str)
        
        # MySQL
        conn_str = "mysql+pymysql://user:password@localhost:3306/database"
        
        # SQLite
        conn_str = "sqlite:///path/to/database.db"
    """
    engine = sqlalchemy.create_engine(connection_string)
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


def handle_missing_values(df, strategy='median', categorical_strategy='mode'):
    """
    Handle missing values in dataset.
    
    Args:
        df: pandas DataFrame
        strategy: 'median', 'mean', or 'drop' for numerical columns
        categorical_strategy: 'mode' or 'drop' for categorical columns
        
    Returns:
        Cleaned DataFrame
        
    Example:
        df_clean = handle_missing_values(df, strategy='median')
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        # Only drop if very few missing values
        missing_pct = df_clean.isnull().sum().sum() / (df_clean.shape[0] * df_clean.shape[1])
        if missing_pct < 0.05:
            df_clean = df_clean.dropna()
            print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        else:
            print(f"Warning: Too much missing data ({missing_pct:.2%}). Use imputation instead.")
            return df_clean
    
    else:
        # Handle numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0 and df_clean[numerical_cols].isnull().any().any():
            imputer = SimpleImputer(strategy=strategy)
            df_clean[numerical_cols] = imputer.fit_transform(df_clean[numerical_cols])
            print(f"Imputed {df_clean[numerical_cols].isnull().sum().sum()} numerical values with {strategy}")
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        if categorical_strategy == 'mode':
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    mode_value = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_value, inplace=True)
                    print(f"Imputed {col} with mode: {mode_value}")
    
    return df_clean


def create_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Create train/test split with optional stratification.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set (default 0.2)
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution (recommended for imbalanced data)
        
    Returns:
        X_train, X_test, y_train, y_test
        
    Example:
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=True
        )
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X):.1%})")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X):.1%})")
    
    if stratify:
        print(f"Training target distribution: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
        print(f"Test target distribution: {pd.Series(y_test).value_counts(normalize=True).to_dict()}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, return_dataframe=True):
    """
    Scale features using StandardScaler.
    
    CRITICAL: Always fit on training data only!
    
    Args:
        X_train: Training features
        X_test: Test features
        return_dataframe: Whether to return DataFrames (preserves column names)
        
    Returns:
        X_train_scaled, X_test_scaled, fitted_scaler
        
    Example:
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Save scaler for production
        import joblib
        joblib.dump(scaler, 'scaler.pkl')
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    if return_dataframe and isinstance(X_train, pd.DataFrame):
        # Convert back to DataFrame to keep column names
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index
        )
    
    print("Features scaled using StandardScaler")
    print(f"Training mean: {scaler.mean_[:3]}...")
    print(f"Training std: {scaler.scale_[:3]}...")
    
    return X_train_scaled, X_test_scaled, scaler


def encode_categorical_features(df, columns=None, drop_first=True):
    """
    One-hot encode categorical features.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to encode (if None, encodes all object columns)
        drop_first: Drop first dummy category to avoid multicollinearity
        
    Returns:
        DataFrame with encoded features
        
    Example:
        df_encoded = encode_categorical_features(
            df, 
            columns=['contract_type', 'payment_method'],
            drop_first=True
        )
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    
    print(f"Encoded {len(columns)} categorical columns")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {df_encoded.shape}")
    print(f"Added {df_encoded.shape[1] - df.shape[1]} dummy columns")
    
    return df_encoded


def check_data_leakage(X_train, X_test):
    """
    Check for potential data leakage between train and test sets.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Dictionary with leakage checks
        
    Example:
        leakage_report = check_data_leakage(X_train, X_test)
    """
    report = {
        'index_overlap': len(set(X_train.index).intersection(set(X_test.index))),
        'identical_rows': 0,
        'feature_distribution_similarity': {}
    }
    
    # Check for identical rows
    if isinstance(X_train, pd.DataFrame):
        train_tuples = set(map(tuple, X_train.values))
        test_tuples = set(map(tuple, X_test.values))
        report['identical_rows'] = len(train_tuples.intersection(test_tuples))
    
    # Check feature distributions
    for col in X_train.columns:
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()
        similarity = 1 - abs(train_mean - test_mean) / (abs(train_mean) + 1e-10)
        report['feature_distribution_similarity'][col] = similarity
    
    # Print warnings
    if report['index_overlap'] > 0:
        print(f"WARNING: {report['index_overlap']} overlapping indices between train and test!")
    
    if report['identical_rows'] > 0:
        print(f"WARNING: {report['identical_rows']} identical rows found in both sets!")
    
    print(f"Average feature distribution similarity: {np.mean(list(report['feature_distribution_similarity'].values())):.3f}")
    
    return report


def create_data_summary(df):
    """
    Create a comprehensive data summary report.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Summary statistics dictionary
        
    Example:
        summary = create_data_summary(df)
        print(summary)
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'data_types': df.dtypes.value_counts().to_dict(),
        'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'duplicated_rows': df.duplicated().sum()
    }
    
    print("Data Summary Report")
    print("=" * 50)
    print(f"Shape: {summary['shape']}")
    print(f"Memory usage: {summary['memory_usage_mb']:.2f} MB")
    print(f"Duplicated rows: {summary['duplicated_rows']}")
    print(f"Missing values: {sum(summary['missing_values'].values())} total")
    print(f"Data types: {summary['data_types']}")
    
    return summary
