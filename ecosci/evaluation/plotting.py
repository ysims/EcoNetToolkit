"""Plotting and visualisation utilities for evaluation results.

This module contains functions for creating various plots and visualisations
for model evaluation, including confusion matrices, PR curves, residual plots,
and feature importance plots.
"""

from typing import Dict, Any, Optional, List
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_metric_comparisons(
    combined_df: pd.DataFrame,
    plot_metrics: List[str],
    output_dir: str,
    all_dfs: Optional[Dict] = None
):
    """Create boxplot comparisons for metrics across models or seeds.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined DataFrame with all results
    plot_metrics : List[str]
        List of metric names to plot
    output_dir : str
        Directory to save plots
    all_dfs : Optional[Dict]
        Dictionary of DataFrames per model (for multi-model comparison)
    """
    if all_dfs and len(all_dfs) > 1:
        # Multi-model comparison plots
        for m in plot_metrics:
            if m in combined_df.columns:
                plt.figure(figsize=(max(6, len(all_dfs) * 1.5), 4))
                sns.boxplot(data=combined_df, x="model", y=m)
                plt.title(f"{m.replace('_', ' ').title()} - Model Comparison")
                plt.xlabel("Model")
                plt.ylabel(m.replace("_", " ").title())
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"comparison_{m}.png"))
                plt.close()
    else:
        # Single model plots (backward compatibility)
        for m in plot_metrics:
            if m in combined_df.columns:
                plt.figure(figsize=(4, 3))
                sns.boxplot(data=combined_df, y=m)
                plt.title(m.replace("_", " ").title())
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"metric_{m}.png"))
                plt.close()


def plot_confusion_matrices(
    results: Dict,
    y_test,
    output_dir: str
):
    """Create confusion matrix heatmaps for classification models.
    
    Parameters
    ----------
    results : Dict
        Results dictionary {model_name: [run_results]}
    y_test : array-like
        Test labels
    output_dir : str
        Directory to save plots
    """
    from sklearn.metrics import confusion_matrix
    
    for model_name, model_results in results.items():
        try:
            last = model_results[-1]
            cm = confusion_matrix(y_test, last.get("y_pred"))
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name.upper()} (last run)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            plt.savefig(
                os.path.join(model_dir, f"confusion_matrix_{model_name}.png")
            )
            plt.close()
        except Exception as e:
            print(
                f"Warning: Could not create confusion matrix for {model_name}: {e}"
            )


def plot_pr_curves(
    results: Dict,
    y_test,
    output_dir: str
):
    """Create precision-recall curves for classification models.
    
    Parameters
    ----------
    results : Dict
        Results dictionary {model_name: [run_results]}
    y_test : array-like
        Test labels
    output_dir : str
        Directory to save plots
    """
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
    )
    
    # Individual PR curves for each model
    for model_name, model_results in results.items():
        try:
            last = model_results[-1]
            y_proba = last.get("y_proba")
            if y_proba is not None:
                # if proba has shape (n,2), take positive class
                if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
                    pos_scores = y_proba[:, 1]
                else:
                    pos_scores = y_proba
                precision, recall, _ = precision_recall_curve(y_test, pos_scores)
                ap = average_precision_score(y_test, pos_scores)
                plt.figure(figsize=(5, 4))
                plt.plot(recall, precision, label=f"AP={ap:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall - {model_name.upper()} (last run)")
                plt.legend()
                plt.tight_layout()
                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                plt.savefig(os.path.join(model_dir, f"pr_curve_{model_name}.png"))
                plt.close()
        except Exception as e:
            pass  # Silently skip if PR curve can't be generated

    # Combined PR curve for model comparison (binary classification only)
    if len(results) > 1:
        try:
            plt.figure(figsize=(7, 5))
            for model_name, model_results in results.items():
                last = model_results[-1]
                y_proba = last.get("y_proba")
                if y_proba is not None:
                    if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
                        pos_scores = y_proba[:, 1]
                    else:
                        pos_scores = y_proba
                    precision, recall, _ = precision_recall_curve(
                        y_test, pos_scores
                    )
                    ap = average_precision_score(y_test, pos_scores)
                    plt.plot(
                        recall,
                        precision,
                        label=f"{model_name} (AP={ap:.3f})",
                        linewidth=2,
                    )

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Comparison (last run)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pr_curve_comparison.png"))
            plt.close()
        except Exception as e:
            pass  # Silently skip if comparison can't be generated


def plot_residuals(
    results: Dict,
    y_test,
    output_dir: str
):
    """Create residual plots for regression models.
    
    Parameters
    ----------
    results : Dict
        Results dictionary {model_name: [run_results]}
    y_test : array-like
        Test values
    output_dir : str
        Directory to save plots
    """
    for model_name, model_results in results.items():
        try:
            last = model_results[-1]
            y_pred = last.get("y_pred")
            residuals = y_test - y_pred

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Predicted vs Actual
            ax1.scatter(y_test, y_pred, alpha=0.6)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
            ax1.set_xlabel("Actual")
            ax1.set_ylabel("Predicted")
            ax1.set_title(f"Predicted vs Actual - {model_name.upper()}")
            ax1.grid(True, alpha=0.3)

            # Residual plot
            ax2.scatter(y_pred, residuals, alpha=0.6)
            ax2.axhline(y=0, color="r", linestyle="--", lw=2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Residuals")
            ax2.set_title(f"Residual Plot - {model_name.upper()}")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            plt.savefig(os.path.join(model_dir, f"residual_plot_{model_name}.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create residual plot for {model_name}: {e}")


def plot_feature_importance(
    feature_importances: np.ndarray,
    feature_std: np.ndarray,
    feature_names: Optional[List[str]],
    model_name: str,
    output_dir: str,
    importance_method: str,
    top_n: int = 20
):
    """Create feature importance bar plot.
    
    Parameters
    ----------
    feature_importances : np.ndarray
        Array of feature importance values
    feature_std : np.ndarray
        Array of standard deviations for feature importances
    feature_names : Optional[List[str]]
        Names of features
    model_name : str
        Name of the model
    output_dir : str
        Directory to save plot
    importance_method : str
        Description of the method used to compute importance
    top_n : int
        Number of top features to display
    """
    try:
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1]
        
        # Helper to get feature label
        def get_feature_label(idx):
            if feature_names and idx < len(feature_names):
                return feature_names[idx]
            return f"Feature {idx}"
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot top N features
        top_n = min(top_n, len(feature_importances))
        top_indices = indices[:top_n]
        top_importances = feature_importances[top_indices]
        top_stds = feature_std[top_indices]
        
        y_pos = np.arange(top_n)
        ax.barh(y_pos, top_importances, xerr=top_stds, align='centre', alpha=0.7, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([get_feature_label(idx) for idx in top_indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances - {model_name.upper()}\n({importance_method})')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        plt.savefig(os.path.join(model_dir, f"feature_importance_{model_name}.png"), dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Feature importance plot saved to: {model_dir}/feature_importance_{model_name}.png")
    except Exception as e:
        print(f"  Warning: Could not create feature importance plot: {e}")


def plot_cv_comparison(
    all_dfs: Dict[str, pd.DataFrame],
    output_dir: str,
    problem_type: str,
    is_multi_output: bool
):
    """Create comparison plots across folds and models for cross-validation.
    
    Parameters
    ----------
    all_dfs : Dict[str, pd.DataFrame]
        Dictionary of DataFrames per model
    output_dir : str
        Directory to save plots
    problem_type : str
        "classification" or "regression"
    is_multi_output : bool
        Whether this is multi-output problem
    """
    if len(all_dfs) <= 1:
        return
    
    print(f"\nGenerating comparison plots...")
    
    # Determine which metrics to plot
    if problem_type == "classification":
        if is_multi_output:
            metrics_to_plot = {"accuracy_mean": "Accuracy", "balanced_accuracy_mean": "Balanced Accuracy", 
                             "f1_mean": "F1 Score", "cohen_kappa_mean": "Cohen's Kappa"}
        else:
            metrics_to_plot = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced Accuracy",
                             "f1": "F1 Score", "cohen_kappa": "Cohen's Kappa"}
    else:
        if is_multi_output:
            metrics_to_plot = {"r2_mean": "R²", "rmse_mean": "RMSE", "mae_mean": "MAE", "mse_mean": "MSE"}
        else:
            metrics_to_plot = {"r2": "R²", "rmse": "RMSE", "mae": "MAE", "mse": "MSE"}
    
    for metric_key, metric_label in metrics_to_plot.items():
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_for_plot = []
            labels_for_plot = []
            
            for model_name, df in all_dfs.items():
                if metric_key in df.columns:
                    data_for_plot.append(df[metric_key].values)
                    labels_for_plot.append(model_name.upper())
            
            if data_for_plot:
                bp = ax.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")
                
                ax.set_ylabel(metric_label)
                ax.set_title(f"{metric_label} Comparison - K-Fold CV")
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"cv_comparison_{metric_key}.png"))
                plt.close()
        except Exception as e:
            print(f"  Warning: Could not create comparison plot for {metric_key}: {e}")


def plot_validation_vs_test(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    problem_type: str
):
    """Create validation vs test comparison plots for hyperparameter tuning.
    
    Parameters
    ----------
    val_df : pd.DataFrame
        Validation metrics DataFrame
    test_df : pd.DataFrame
        Test metrics DataFrame
    model_name : str
        Name of the model
    output_dir : str
        Directory to save plots
    problem_type : str
        "classification" or "regression"
    """
    try:
        # Select key metrics based on problem type
        if problem_type == "regression":
            metrics_to_plot = ['r2', 'rmse', 'mae']
        else:
            metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Prepare data for boxplot
            data = [val_df[metric].values, test_df[metric].values]
            positions = [1, 2]
            labels = ['Validation', 'Test']
            
            bp = ax.boxplot(data, positions=positions, labels=labels, 
                           patch_artist=True, widths=0.6)
            
            # Colour boxes
            colours = ['lightblue', 'lightcoral']
            for patch, colour in zip(bp['boxes'], colours):
                patch.set_facecolor(colour)
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} - {model_name.upper()}')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        plot_path = os.path.join(model_dir, f"val_vs_test_{model_name}.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Comparison plot saved to: {plot_path}")
    except Exception as e:
        print(f"  Warning: Could not create comparison plot: {e}")
