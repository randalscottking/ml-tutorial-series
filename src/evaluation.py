"""
Model evaluation utilities for ML tutorial series.

Functions for:
- Computing evaluation metrics
- Creating confusion matrices
- ROC curves and AUC
- Precision-recall analysis
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_metrics_report(metrics):
    """
    Print formatted metrics report.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    print("Model Performance Metrics")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper():.<20} {value:.4f}")
    print("-" * 40)
