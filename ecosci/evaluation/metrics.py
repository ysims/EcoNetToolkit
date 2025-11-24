"""Metrics computation for classification and regression tasks.

This module contains functions for computing various performance metrics
for both single-output and multi-output problems.
"""

from typing import Dict, Any
import numpy as np


def safe_std(vals):
    """Calculate std, returning 0.0 instead of nan when there's only 1 value."""
    if len(vals) <= 1:
        return 0.0
    return vals.std()


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    """Compute classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities (if available).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy, balanced_accuracy, precision, recall,
        f1, cohen_kappa, confusion_matrix, and optionally roc_auc and average_precision.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        balanced_accuracy_score,
        average_precision_score,
        cohen_kappa_score,
    )

    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(
        y_true,
        y_pred,
        average="binary" if len(np.unique(y_true)) == 2 else "macro",
        zero_division=0,
    )
    out["recall"] = recall_score(
        y_true,
        y_pred,
        average="binary" if len(np.unique(y_true)) == 2 else "macro",
        zero_division=0,
    )
    out["f1"] = f1_score(
        y_true,
        y_pred,
        average="binary" if len(np.unique(y_true)) == 2 else "macro",
        zero_division=0,
    )
    out["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if y_proba is not None:
        try:
            # Binary classification: use positive class probabilities
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                out["average_precision"] = average_precision_score(
                    y_true, y_proba[:, 1]
                )
            # Multiclass: use OVR strategy
            elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                out["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                out["average_precision"] = average_precision_score(
                    y_true, y_proba, average="macro"
                )
            # 1D probabilities (rare, but handle it)
            else:
                out["roc_auc"] = roc_auc_score(y_true, y_proba)
                out["average_precision"] = average_precision_score(y_true, y_proba)
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC/AP: {e}")
            out["roc_auc"] = None
            out["average_precision"] = None
    else:
        out["roc_auc"] = None
        out["average_precision"] = None

    return out


def compute_multi_output_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    """Compute classification metrics for multi-output predictions.
    
    For multi-output, we compute metrics per output and aggregate.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Ground truth labels
    y_pred : array-like, shape (n_samples, n_outputs)
        Predicted labels
    y_proba : array-like, shape (n_samples, n_outputs, n_classes) or None
        Predicted probabilities (if available)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with average metrics across outputs and per-output metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        balanced_accuracy_score,
        cohen_kappa_score,
    )
    
    n_outputs = y_true.shape[1]
    out = {}
    
    # Compute per-output metrics
    per_output_metrics = []
    for i in range(n_outputs):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        y_proba_i = y_proba[i] if y_proba is not None and isinstance(y_proba, list) else None
        
        metrics_i = compute_classification_metrics(y_true_i, y_pred_i, y_proba_i)
        per_output_metrics.append(metrics_i)
    
    # Aggregate metrics (average across outputs)
    for key in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'cohen_kappa']:
        values = [m[key] for m in per_output_metrics if key in m and m[key] is not None]
        if values:
            out[f"{key}_mean"] = np.mean(values)
            out[f"{key}_std"] = np.std(values)
    
    # Store per-output metrics
    out["per_output"] = per_output_metrics
    out["n_outputs"] = n_outputs
    
    return out


def compute_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Compute regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - r2: R-squared (coefficient of determination)
        - mape: Mean Absolute Percentage Error (if no zeros in y_true)
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    out = {}
    out["mse"] = mean_squared_error(y_true, y_pred)
    out["rmse"] = np.sqrt(out["mse"])
    out["mae"] = mean_absolute_error(y_true, y_pred)
    out["r2"] = r2_score(y_true, y_pred)

    # MAPE - only compute if no zeros in y_true
    try:
        if not np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            out["mape"] = mape
        else:
            out["mape"] = None
    except Exception:
        out["mape"] = None

    return out


def compute_multi_output_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Compute regression metrics for multi-output predictions.
    
    For multi-output regression, we compute metrics per output and aggregate.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_outputs)
        Ground truth values
    y_pred : array-like, shape (n_samples, n_outputs)
        Predicted values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with average metrics across outputs and per-output metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    n_outputs = y_true.shape[1]
    out = {}
    
    # Compute per-output metrics
    per_output_metrics = []
    for i in range(n_outputs):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        
        metrics_i = compute_regression_metrics(y_true_i, y_pred_i)
        per_output_metrics.append(metrics_i)
    
    # Aggregate metrics (average across outputs)
    for key in ['mse', 'rmse', 'mae', 'r2']:
        values = [m[key] for m in per_output_metrics if key in m and m[key] is not None]
        if values:
            out[f"{key}_mean"] = np.mean(values)
            out[f"{key}_std"] = np.std(values)
    
    # Store per-output metrics
    out["per_output"] = per_output_metrics
    out["n_outputs"] = n_outputs
    
    return out
