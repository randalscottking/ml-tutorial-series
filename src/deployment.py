"""
Deployment utilities for ML tutorial series.

Functions for:
- Model API endpoints
- Prediction logging
- Performance monitoring
- Model versioning
"""

import pandas as pd
from datetime import datetime
import json


def make_prediction(model, input_data, scaler=None):
    """
    Make prediction on new data.
    
    Args:
        model: Trained model
        input_data: New data (dict or DataFrame)
        scaler: Fitted scaler (optional)
        
    Returns:
        Prediction result
    """
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    
    # Scale if scaler provided
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df
    
    # Get prediction and probability
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability[1]),  # Probability of positive class
        'timestamp': datetime.now().isoformat()
    }


def log_prediction(prediction_result, log_file='predictions.jsonl'):
    """
    Log prediction to file for monitoring.
    
    Args:
        prediction_result: Dictionary with prediction details
        log_file: Path to log file
    """
    with open(log_file, 'a') as f:
        f.write(json.dumps(prediction_result) + '\n')


def batch_predict(model, input_data, scaler=None, log=True):
    """
    Make predictions on batch of data.
    
    Args:
        model: Trained model
        input_data: DataFrame of new data
        scaler: Fitted scaler (optional)
        log: Whether to log predictions
        
    Returns:
        DataFrame with predictions
    """
    # Scale if needed
    if scaler is not None:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data
    
    # Make predictions
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[:, 1]
    
    # Create results dataframe
    results = input_data.copy()
    results['prediction'] = predictions
    results['probability'] = probabilities
    results['timestamp'] = datetime.now().isoformat()
    
    # Log if requested
    if log:
        for _, row in results.iterrows():
            log_prediction(row.to_dict())
    
    return results


def monitor_model_performance(predictions_df, actuals_df):
    """
    Calculate model performance metrics on production data.
    
    Args:
        predictions_df: DataFrame with predictions
        actuals_df: DataFrame with actual outcomes
        
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Merge predictions and actuals
    merged = predictions_df.merge(actuals_df, on='id')
    
    metrics = {
        'accuracy': accuracy_score(merged['actual'], merged['prediction']),
        'roc_auc': roc_auc_score(merged['actual'], merged['probability']),
        'total_predictions': len(merged),
        'timestamp': datetime.now().isoformat()
    }
    
    return metrics
