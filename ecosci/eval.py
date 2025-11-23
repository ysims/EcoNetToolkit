"""Evaluation and reporting utilities.

Focus on metrics that make sense for unbalanced datasets:
- Balanced accuracy (accounts for class imbalance)
- Precision/Recall/F1 (macro for multi-class)
- ROC-AUC and Average Precision (PR AUC) when probabilities are available

Outputs:
- `report.json`: list of per-seed metrics
- `metric_*.png`: quick boxplots across seeds
- `confusion_matrix.png`: confusion matrix heatmap (last run)
- `pr_curve.png`: precision-recall curve if binary and probabilities exist
"""

from typing import Dict, Any, Optional
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def safe_std(vals):
    """Calculate std, returning 0.0 instead of nan when there's only 1 value."""
    if len(vals) <= 1:
        return 0.0
    return vals.std()


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
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


def evaluate_and_report(
    results, y_test, output_dir: str = "outputs", problem_type: str = "classification", label_names: list = None, feature_names: list = None, X_test=None
):
    """Compute metrics across runs and save a compact report and plots.

    Parameters
    ----------
    results : dict or list
        If dict: Output from Trainer.run() with multiple models {model_name: [run_results]}
        If list: Single model results (backward compatibility)
    y_test : array-like
        Ground-truth labels (classification) or values (regression) for the test set.
    output_dir : str
        Where to save the report and plots.
    problem_type : str
        "classification" or "regression". Determines which metrics to compute.
    label_names : list, optional
        Names of the target variables (for multi-output). If None, will use Output 1, Output 2, etc.
    feature_names : list, optional
        Names of the input features. If None, will use Feature 0, Feature 1, etc.
    X_test : array-like, optional
        Test features for permutation importance. Required for models without built-in importance (e.g., MLPs).
    """
    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd

    # Handle backward compatibility: convert list to dict
    if isinstance(results, list):
        # Assume single model (old format)
        model_name = results[0].get("model_name", "model") if results else "model"
        results = {model_name: results}

    # Determine if we have multi-output
    is_multi_output = len(y_test.shape) > 1 and y_test.shape[1] > 1
    
    # Compute metrics for each model
    all_summaries = {}
    all_dfs = {}

    for model_name, model_results in results.items():
        summary = []
        for r in model_results:
            seed = r.get("seed")
            y_pred = r.get("y_pred")
            y_proba = r.get("y_proba")

            if is_multi_output:
                # Multi-output case
                if problem_type == "regression":
                    metrics = compute_multi_output_regression_metrics(y_test, y_pred)
                else:
                    metrics = compute_multi_output_classification_metrics(y_test, y_pred, y_proba)
            else:
                # Single output case
                if problem_type == "regression":
                    metrics = compute_regression_metrics(y_test, y_pred)
                else:
                    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

            metrics["seed"] = seed
            metrics["model"] = model_name
            summary.append(metrics)

        all_summaries[model_name] = summary
        all_dfs[model_name] = pd.DataFrame(summary)

    # Save individual model reports in model-specific subfolders
    for model_name, summary in all_summaries.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        report_path = os.path.join(model_dir, f"report_{model_name}.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Combine all results for comparison
    combined_df = pd.concat(all_dfs.values(), ignore_index=True)
    combined_summary = [item for sublist in all_summaries.values() for item in sublist]

    # Save combined report
    with open(os.path.join(output_dir, "report_all_models.json"), "w") as f:
        json.dump(combined_summary, f, indent=2)

    # Print results for each model
    if is_multi_output:
        # For multi-output, show aggregated metrics
        if problem_type == "regression":
            display_cols = ["seed", "mse_mean", "rmse_mean", "mae_mean", "r2_mean"]
            comparison_metrics = ["mse_mean", "rmse_mean", "mae_mean", "r2_mean"]
            plot_metrics = ["mse_mean", "rmse_mean", "mae_mean", "r2_mean"]
        else:
            display_cols = [
                "seed",
                "accuracy_mean",
                "balanced_accuracy_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "cohen_kappa_mean",
            ]
            comparison_metrics = [
                "accuracy_mean",
                "balanced_accuracy_mean",
                "f1_mean",
                "cohen_kappa_mean",
            ]
            plot_metrics = ["accuracy_mean", "balanced_accuracy_mean", "f1_mean", "cohen_kappa_mean"]
    else:
        # Single output
        if problem_type == "regression":
            display_cols = ["seed", "mse", "rmse", "mae", "r2"]
            if "mape" in combined_df.columns:
                display_cols.append("mape")
            comparison_metrics = ["mse", "rmse", "mae", "r2"]
            plot_metrics = ["mse", "rmse", "mae", "r2"]
        else:
            display_cols = [
                "seed",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "cohen_kappa",
            ]
            if "roc_auc" in combined_df.columns:
                display_cols.append("roc_auc")
            if "average_precision" in combined_df.columns:
                display_cols.append("average_precision")
            comparison_metrics = [
                "accuracy",
                "balanced_accuracy",
                "f1",
                "cohen_kappa",
                "roc_auc",
            ]
            plot_metrics = ["accuracy", "balanced_accuracy", "f1", "cohen_kappa"]

    for model_name, df in all_dfs.items():
        print(f"\n{'='*80}")
        print(f"RESULTS FOR MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        print(
            df[display_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A",
            )
        )

        print(f"\n{'-'*80}")
        print(f"SUMMARY STATISTICS FOR {model_name.upper()} (mean ± std)")
        print(f"{'-'*80}")
        metric_cols = [c for c in display_cols if c != "seed"]
        for col in metric_cols:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"{col:20s}: {vals.mean():.4f} ± {safe_std(vals):.4f}")
        print(f"{'-'*80}\n")
        
        # For multi-output, print per-output breakdown
        if is_multi_output and "per_output" in df.columns[0:1] or (len(df) > 0 and "per_output" in df.iloc[0]):
            print(f"\n{'-'*80}")
            print(f"PER-OUTPUT BREAKDOWN FOR {model_name.upper()}")
            print(f"{'-'*80}")
            
            # Get per-output metrics from first seed as example structure
            first_row = df.iloc[0]
            if "per_output" in first_row and first_row["per_output"] is not None:
                per_output_list = first_row["per_output"]
                n_outputs = len(per_output_list)
                
                # Collect metrics across all seeds for each output
                for output_idx in range(n_outputs):
                    # Use label name if provided, otherwise use generic Output N
                    if label_names and output_idx < len(label_names):
                        output_label = f"{label_names[output_idx]}"
                    else:
                        output_label = f"Output {output_idx + 1}"
                    
                    print(f"\n{output_label}:")
                    
                    # Aggregate metrics across seeds for this output
                    output_metrics = {}
                    for _, row in df.iterrows():
                        if "per_output" in row and row["per_output"] is not None:
                            per_out = row["per_output"]
                            if output_idx < len(per_out):
                                for key, val in per_out[output_idx].items():
                                    if key not in ["confusion_matrix", "seed", "model"] and val is not None:
                                        if key not in output_metrics:
                                            output_metrics[key] = []
                                        output_metrics[key].append(val)
                    
                    # Print summary for this output
                    for metric_name, values in sorted(output_metrics.items()):
                        if len(values) > 0:
                            mean_val = np.mean(values)
                            std_val = safe_std(np.array(values))
                            print(f"  {metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")
            print(f"{'-'*80}\n")

    # Model comparison table
    if len(all_dfs) > 1:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON (mean ± std)")
        print(f"{'='*80}")
        comparison_data = []
        for model_name, df in all_dfs.items():
            row = {"Model": model_name}
            for col in comparison_metrics:
                if col in df.columns:
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        row[col] = f"{vals.mean():.4f} ± {safe_std(vals):.4f}"
                    else:
                        row[col] = "N/A"
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print(f"{'='*80}\n")

        # Determine best and second-best models
        # For regression: lower MSE is better
        # Select primary metric based on problem type
        # For classification: Cohen's kappa is more robust as it accounts for chance agreement
        # For regression: MSE is standard (lower is better)
        if is_multi_output:
            if problem_type == "regression":
                primary_metric = "mse_mean"
                lower_is_better = True
            else:
                primary_metric = "cohen_kappa_mean"
                lower_is_better = False
        else:
            if problem_type == "regression":
                primary_metric = "mse"
                lower_is_better = True
            else:
                primary_metric = "cohen_kappa"
                lower_is_better = False

        # Calculate mean primary metric for each model
        model_scores = []
        for model_name, df in all_dfs.items():
            if primary_metric in df.columns:
                vals = df[primary_metric].dropna()
                if len(vals) > 0:
                    model_scores.append(
                        {
                            "model": model_name,
                            "mean": vals.mean(),
                            "std": safe_std(vals),
                        }
                    )

        if len(model_scores) > 0:
            # Sort by primary metric
            model_scores.sort(key=lambda x: x["mean"], reverse=not lower_is_better)

            print(f"\n{'='*80}")
            print("MODEL RANKING")
            print(f"{'='*80}")
            print(
                f"Ranked by: {primary_metric.replace('_', ' ').upper()} ({'Lower is better' if lower_is_better else 'Higher is better'})"
            )
            print(f"{'-'*80}")

            # Create ranking table with all metrics
            ranking_data = []
            for rank, score in enumerate(model_scores, 1):
                model_name = score["model"]
                model_df = all_dfs[model_name]

                row = {
                    "Rank": f"{rank}.",
                    "Model": model_name.upper(),
                    primary_metric: f"{score['mean']:.4f} ± {score['std']:.4f}",
                }

                # Add other metrics
                metric_cols = [
                    c for c in display_cols if c != "seed" and c != primary_metric
                ]
                for col in metric_cols:
                    if col in model_df.columns:
                        vals = model_df[col].dropna()
                        if len(vals) > 0:
                            row[col] = f"{vals.mean():.4f} ± {safe_std(vals):.4f}"

                ranking_data.append(row)

            # Print ranking table
            ranking_df = pd.DataFrame(ranking_data)
            print(ranking_df.to_string(index=False))
            print(f"{'='*80}\n")

    # Plot comparison across models

    if len(all_dfs) > 1:
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

    # Confusion matrices for each model (last run) - classification only
    if problem_type == "classification":
        for model_name, model_results in results.items():
            try:
                from sklearn.metrics import confusion_matrix

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

    # Precision-Recall curves for each model - classification only
    if problem_type == "classification":
        for model_name, model_results in results.items():
            try:
                from sklearn.metrics import (
                    precision_recall_curve,
                    average_precision_score,
                )

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
                from sklearn.metrics import (
                    precision_recall_curve,
                    average_precision_score,
                )

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

    # Residual plots for regression
    elif problem_type == "regression" and not is_multi_output:
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

    # Feature importance analysis
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    for model_name, model_results in results.items():
        try:
            import joblib
            from sklearn.inspection import permutation_importance
            from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
            
            # Collect feature importances from all trained models
            all_importances = []
            
            for result in model_results:
                model_path = result.get("model_path")
                
                if not model_path or not os.path.exists(model_path):
                    continue
                    
                model = joblib.load(model_path)
                
                # Extract feature importances based on model type
                feature_importances = None
                importance_method = None
                
                # For multi-output models, need to handle differently
                if isinstance(model, (MultiOutputClassifier, MultiOutputRegressor)):
                    # Get the base estimator
                    if hasattr(model.estimators_[0], 'feature_importances_'):
                        # Average importances across outputs
                        importances_list = [est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')]
                        if importances_list:
                            feature_importances = np.mean(importances_list, axis=0)
                            importance_method = "built-in (averaged across outputs)"
                elif hasattr(model, 'feature_importances_'):
                    # Tree-based models (Random Forest, XGBoost)
                    feature_importances = model.feature_importances_
                    importance_method = "built-in (Gini/gain)"
                elif hasattr(model, 'coef_'):
                    # Linear models (use absolute coefficient values)
                    coef = model.coef_
                    if len(coef.shape) > 1:
                        # Multi-output or multi-class: average absolute coefficients
                        feature_importances = np.mean(np.abs(coef), axis=0)
                    else:
                        feature_importances = np.abs(coef)
                    importance_method = "coefficient magnitude"
                
                # If no built-in importance, use permutation importance (works for all models including MLPs)
                if feature_importances is None and X_test is not None:
                    # Only print once for the first model
                    if len(all_importances) == 0:
                        print(f"\n{model_name.upper()}: Computing permutation importance across all {len(model_results)} models...")
                        print(f"  (This measures performance drop when each feature is shuffled)")
                    
                    # Determine scoring metric based on problem type
                    if problem_type == "classification":
                        scoring = "accuracy"
                    else:  # regression
                        scoring = "r2"
                    
                    # Compute permutation importance
                    perm_importance = permutation_importance(
                        model, X_test, y_test, 
                        n_repeats=10,  # Number of times to permute each feature
                        random_state=42,
                        scoring=scoring,
                        n_jobs=-1  # Use all available cores
                    )
                    feature_importances = perm_importance.importances_mean
                    importance_method = f"permutation (based on {scoring}, averaged across seeds)"
                elif feature_importances is not None and importance_method:
                    # Update method to indicate averaging across seeds
                    if len(all_importances) == 0:
                        importance_method = importance_method + f", averaged across {len(model_results)} seeds"
                
                if feature_importances is not None:
                    all_importances.append(feature_importances)
            
            # Average importances across all models
            if all_importances:
                # Average across all seeds
                avg_feature_importances = np.mean(all_importances, axis=0)
                std_feature_importances = np.std(all_importances, axis=0)
                
                print(f"\n{model_name.upper()}:")
                print(f"{'-'*80}")
                print(f"Method: {importance_method}")
                print(f"Averaged across {len(all_importances)} model(s)")
                
                # Sort features by importance
                indices = np.argsort(avg_feature_importances)[::-1]
                
                # Helper to get feature label
                def get_feature_label(idx):
                    if feature_names and idx < len(feature_names):
                        return feature_names[idx]
                    return f"Feature {idx}"
                
                # Print top 20 features with mean ± std
                print(f"\nTop 20 most important features (mean ± std):")
                for i, idx in enumerate(indices[:20], 1):
                    importance = avg_feature_importances[idx]
                    std = std_feature_importances[idx]
                    feat_label = get_feature_label(idx)
                    print(f"  {i:2d}. {feat_label:40s}: {importance:.6f} ± {std:.6f}")
                
                # Create feature importance plot
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot top 20 features
                    top_n = min(20, len(avg_feature_importances))
                    top_indices = indices[:top_n]
                    top_importances = avg_feature_importances[top_indices]
                    top_stds = std_feature_importances[top_indices]
                    
                    y_pos = np.arange(top_n)
                    ax.barh(y_pos, top_importances, xerr=top_stds, align='center', alpha=0.7, capsize=3)
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
            else:
                print(f"\n{model_name.upper()}: Feature importance not available (no models or no test data provided)")
        except Exception as e:
            print(f"\nWarning: Could not extract feature importance for {model_name}: {e}")
    
    print(f"\n{'='*80}\n")

    return combined_summary


def evaluate_and_report_cv(
    results, output_dir: str = "outputs", problem_type: str = "classification", 
    label_names: list = None, feature_names: list = None, X_test=None
):
    """Compute metrics for k-fold cross-validation with per-fold breakdown.
    
    Parameters
    ----------
    results : dict
        Output from Trainer.run_cv() with format {model_name: [run_results_with_fold_info]}
        Each run_result contains: seed, fold, y_pred, y_test, etc.
    output_dir : str
        Where to save the report and plots.
    problem_type : str
        "classification" or "regression".
    label_names : list, optional
        Names of the target variables (for multi-output).
    feature_names : list, optional
        Names of the input features.
    X_test : array-like, optional
        Test features (note: in CV, each fold has different test set).
    """
    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd
    
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}\n")
    
    # Determine if multi-output by checking first result
    first_result = next(iter(results.values()))[0]
    y_test_sample = first_result["y_test"]
    is_multi_output = len(y_test_sample.shape) > 1 and y_test_sample.shape[1] > 1
    
    all_summaries = {}
    all_dfs = {}
    
    for model_name, model_results in results.items():
        # Group results by fold
        folds = {}
        for r in model_results:
            fold_id = r.get("fold")
            if fold_id not in folds:
                folds[fold_id] = []
            folds[fold_id].append(r)
        
        n_folds = len(folds)
        
        print(f"\n{model_name.upper()} - {n_folds} Folds")
        print(f"{'-'*80}")
        
        # Compute metrics for each fold
        fold_summaries = []
        for fold_id in sorted(folds.keys()):
            fold_results = folds[fold_id]
            fold_metrics_list = []
            
            for r in fold_results:
                seed = r.get("seed")
                y_pred = r.get("y_pred")
                y_test = r.get("y_test")
                y_proba = r.get("y_proba")
                
                if problem_type == "classification":
                    if is_multi_output:
                        metrics = compute_multi_output_classification_metrics(y_test, y_pred, y_proba)
                    else:
                        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
                else:  # regression
                    if is_multi_output:
                        metrics = compute_multi_output_regression_metrics(y_test, y_pred)
                    else:
                        metrics = compute_regression_metrics(y_test, y_pred)
                
                metrics["seed"] = seed
                metrics["fold"] = fold_id
                fold_metrics_list.append(metrics)
            
            fold_summaries.append(fold_metrics_list)
        
        # Print per-fold metrics (averaged across seeds within each fold)
        print(f"\nPer-Fold Metrics (averaged across {len(fold_results)} seeds per fold):")
        
        if problem_type == "classification":
            if is_multi_output:
                # Multi-output classification
                metric_keys = ["accuracy_mean", "balanced_accuracy_mean", "f1_mean"]
            else:
                metric_keys = ["accuracy", "balanced_accuracy", "f1"]
        else:
            if is_multi_output:
                # Multi-output regression
                metric_keys = ["r2_mean", "rmse_mean", "mae_mean"]
            else:
                metric_keys = ["r2", "rmse", "mae"]
        
        # Print table header
        print(f"\n  {'Fold':<6}", end="")
        for mk in metric_keys:
            print(f"{mk:<20}", end="")
        print()
        print(f"  {'-'*6}", end="")
        for _ in metric_keys:
            print(f"{'-'*20}", end="")
        print()
        
        # Print per-fold metrics
        for fold_id in sorted(folds.keys()):
            fold_results_list = fold_summaries[fold_id]
            fold_df = pd.DataFrame(fold_results_list)
            
            print(f"  {fold_id:<6}", end="")
            for mk in metric_keys:
                if mk in fold_df.columns:
                    mean_val = fold_df[mk].mean()
                    std_val = fold_df[mk].std()
                    print(f"{mean_val:.4f} ± {std_val:.4f}    ", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()
        
        # Compute overall metrics (averaged across all folds and seeds)
        all_fold_metrics = [m for fold_list in fold_summaries for m in fold_list]
        overall_df = pd.DataFrame(all_fold_metrics)
        
        print(f"\n  {'Overall':<6}", end="")
        for mk in metric_keys:
            if mk in overall_df.columns:
                mean_val = overall_df[mk].mean()
                std_val = overall_df[mk].std()
                print(f"{mean_val:.4f} ± {std_val:.4f}    ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print("\n")
        
        # Save detailed report
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        report_path = os.path.join(model_dir, f"report_{model_name}_cv.json")
        
        with open(report_path, "w") as f:
            json.dump(all_fold_metrics, f, indent=2, default=str)
        
        print(f"  Detailed JSON report saved to: {report_path}")
        
        # Save detailed metrics as CSV (all seeds and folds)
        csv_detailed_path = os.path.join(model_dir, f"metrics_{model_name}_cv_detailed.csv")
        overall_df.to_csv(csv_detailed_path, index=False)
        print(f"  Detailed CSV metrics saved to: {csv_detailed_path}")
        
        # Save per-fold summary (averaged across seeds)
        fold_summary_data = []
        for fold_id in sorted(folds.keys()):
            fold_results_list = fold_summaries[fold_id]
            fold_df = pd.DataFrame(fold_results_list)
            
            # Calculate mean and std for each metric
            fold_summary = {"fold": fold_id}
            for col in fold_df.columns:
                if col not in ["seed", "fold"]:
                    fold_summary[f"{col}_mean"] = fold_df[col].mean()
                    fold_summary[f"{col}_std"] = fold_df[col].std()
            fold_summary_data.append(fold_summary)
        
        # Add overall row
        overall_summary = {"fold": "overall"}
        for col in overall_df.columns:
            if col not in ["seed", "fold"]:
                overall_summary[f"{col}_mean"] = overall_df[col].mean()
                overall_summary[f"{col}_std"] = overall_df[col].std()
        fold_summary_data.append(overall_summary)
        
        fold_summary_df = pd.DataFrame(fold_summary_data)
        csv_summary_path = os.path.join(model_dir, f"metrics_{model_name}_cv_per_fold.csv")
        fold_summary_df.to_csv(csv_summary_path, index=False)
        print(f"  Per-fold summary CSV saved to: {csv_summary_path}")
        
        all_summaries[model_name] = overall_df
        all_dfs[model_name] = overall_df
    
    # Create comparison plots across folds and models
    if len(results) > 1:
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
    
    print(f"\n{'='*80}\n")
    
    # Feature importance for CV
    print(f"{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS (per fold and averaged)")
    print(f"{'='*80}")
    
    for model_name, model_results in results.items():
        try:
            import joblib
            from sklearn.inspection import permutation_importance
            from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
            
            # Group results by fold
            folds_for_fi = {}
            for r in model_results:
                fold_id = r.get("fold")
                if fold_id not in folds_for_fi:
                    folds_for_fi[fold_id] = []
                folds_for_fi[fold_id].append(r)
            
            print(f"\n{model_name.upper()}: Collecting feature importances from all {len(folds_for_fi)} folds...")
            
            # Collect feature importance for each fold
            fold_importances_dict = {}
            importance_method = None
            
            for fold_id in sorted(folds_for_fi.keys()):
                fold_results = folds_for_fi[fold_id]
                fold_importances_list = []
                
                for result in fold_results:
                    model_path = result.get("model_path")
                    
                    if not model_path or not os.path.exists(model_path):
                        continue
                    
                    model = joblib.load(model_path)
                    y_test_fold = result.get("y_test")
                    X_test_fold = result.get("X_test")
                    
                    # Extract feature importances based on model type
                    feature_importances = None
                    
                    # For multi-output models
                    if isinstance(model, (MultiOutputClassifier, MultiOutputRegressor)):
                        if hasattr(model.estimators_[0], 'feature_importances_'):
                            importances_list = [est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')]
                            if importances_list:
                                feature_importances = np.mean(importances_list, axis=0)
                                importance_method = "built-in (averaged across outputs, seeds, and folds)"
                    elif hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                        importance_method = "built-in (Gini/gain, averaged across seeds and folds)"
                    elif hasattr(model, 'coef_'):
                        coef = model.coef_
                        if len(coef.shape) > 1:
                            feature_importances = np.mean(np.abs(coef), axis=0)
                        else:
                            feature_importances = np.abs(coef)
                        importance_method = "coefficient magnitude (averaged across seeds and folds)"
                    
                    # If no built-in importance, use permutation importance (e.g., for MLPs)
                    if feature_importances is None and X_test_fold is not None and y_test_fold is not None:
                        # Only print once per fold
                        if len(fold_importances_list) == 0:
                            print(f"  Fold {fold_id}: Computing permutation importance (this may take a while)...")
                        
                        # Determine scoring metric based on problem type
                        if problem_type == "classification":
                            scoring = "accuracy"
                        else:  # regression
                            scoring = "r2"
                        
                        # Compute permutation importance
                        perm_importance = permutation_importance(
                            model, X_test_fold, y_test_fold,
                            n_repeats=10,  # Number of times to permute each feature
                            random_state=42,
                            scoring=scoring,
                            n_jobs=-1  # Use all available cores
                        )
                        feature_importances = perm_importance.importances_mean
                        importance_method = f"permutation (based on {scoring}, averaged across seeds and folds)"
                    
                    if feature_importances is not None:
                        fold_importances_list.append(feature_importances)
                
                # Average across seeds for this fold
                if fold_importances_list:
                    fold_avg_importance = np.mean(fold_importances_list, axis=0)
                    fold_importances_dict[fold_id] = fold_avg_importance
            
            # Average importances across all folds
            if fold_importances_dict:
                all_fold_importances = list(fold_importances_dict.values())
                avg_feature_importances = np.mean(all_fold_importances, axis=0)
                std_feature_importances = np.std(all_fold_importances, axis=0)
                
                print(f"\n{model_name.upper()}:")
                print(f"{'-'*80}")
                print(f"Method: {importance_method}")
                print(f"Computed from all {len(fold_importances_dict)} folds")
                
                # Sort features by importance
                indices = np.argsort(avg_feature_importances)[::-1]
                
                # Helper to get feature label
                def get_feature_label(idx):
                    if feature_names and idx < len(feature_names):
                        return feature_names[idx]
                    return f"Feature {idx}"
                
                # Print top 20 features
                print(f"\nTop 20 most important features (mean ± std):")
                for i, idx in enumerate(indices[:20], 1):
                    importance = avg_feature_importances[idx]
                    std = std_feature_importances[idx]
                    feat_label = get_feature_label(idx)
                    print(f"  {i:2d}. {feat_label:40s}: {importance:.6f} ± {std:.6f}")
                
                # Save feature importance to CSV (all features, sorted by importance)
                # Include per-fold values and overall average
                feature_importance_data = []
                for rank, idx in enumerate(indices, 1):
                    row = {
                        "rank": rank,
                        "feature": get_feature_label(idx),
                        "importance_mean_all_folds": avg_feature_importances[idx],
                        "importance_std_across_folds": std_feature_importances[idx],
                        "feature_index": idx
                    }
                    # Add per-fold importance
                    for fold_id in sorted(fold_importances_dict.keys()):
                        row[f"importance_fold_{fold_id}"] = fold_importances_dict[fold_id][idx]
                    feature_importance_data.append(row)
                
                fi_df = pd.DataFrame(feature_importance_data)
                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                fi_csv_path = os.path.join(model_dir, f"feature_importance_{model_name}_cv_all_folds.csv")
                fi_df.to_csv(fi_csv_path, index=False)
                print(f"\n  Feature importance CSV (all folds) saved to: {fi_csv_path}")
                
                # Create feature importance plot
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    top_n = min(20, len(avg_feature_importances))
                    top_indices = indices[:top_n]
                    top_importances = avg_feature_importances[top_indices]
                    top_stds = std_feature_importances[top_indices]
                    
                    y_pos = np.arange(top_n)
                    ax.barh(y_pos, top_importances, xerr=top_stds, align='center', alpha=0.7, capsize=3)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([get_feature_label(idx) for idx in top_indices])
                    ax.invert_yaxis()
                    ax.set_xlabel('Importance')
                    ax.set_title(f'Top {top_n} Feature Importances - {model_name.upper()} (All Folds)\n({importance_method})')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    model_dir = os.path.join(output_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    plt.savefig(os.path.join(model_dir, f"feature_importance_{model_name}_cv.png"), dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    print(f"\n  Feature importance plot saved to: {model_dir}/feature_importance_{model_name}_cv.png")
                except Exception as e:
                    print(f"  Warning: Could not create feature importance plot: {e}")
            else:
                print(f"\n{model_name.upper()}: Feature importance not available")
        except Exception as e:
            print(f"\nWarning: Could not extract feature importance for {model_name}: {e}")
    
    print(f"\n{'='*80}\n")
    
    return all_summaries


def evaluate_tuning_results(
    results: Dict,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "outputs",
    problem_type: str = "classification",
    label_names: Optional[list] = None,
    feature_names: Optional[list] = None,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
):
    """Evaluate and report results from hyperparameter tuning.
    
    This function evaluates model performance on both validation and test sets
    after hyperparameter tuning. It produces reports and visualizations comparing
    performance across seeds and between val/test sets.
    
    Parameters
    ----------
    results : Dict
        Results dictionary from trainer.run_with_tuning()
    y_val : np.ndarray
        Validation set labels
    y_test : np.ndarray
        Test set labels
    output_dir : str
        Directory to save outputs
    problem_type : str
        "classification" or "regression"
    label_names : list, optional
        Names of label columns
    feature_names : list, optional
        Names of features (for feature importance)
    X_val : np.ndarray, optional
        Validation features (for feature importance)
    X_test : np.ndarray, optional
        Test features (for feature importance)
    
    Returns
    -------
    dict
        Summary statistics for each model
    """
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING EVALUATION")
    print(f"{'='*80}\n")
    
    all_summaries = {}
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 60)
        
        # Collect metrics for each seed
        val_metrics_per_seed = []
        test_metrics_per_seed = []
        best_params_per_seed = []
        best_cv_scores = []
        
        for run in model_results:
            seed = run['seed']
            best_params = run['best_params']
            best_cv_score = run['best_cv_score']
            
            best_params_per_seed.append(best_params)
            best_cv_scores.append(best_cv_score)
            
            # Compute validation metrics
            if problem_type == "regression":
                val_metrics = compute_regression_metrics(
                    y_val, run['y_val_pred']
                )
            else:
                val_metrics = compute_classification_metrics(
                    y_val, run['y_val_pred'], run.get('y_val_proba')
                )
            val_metrics['seed'] = seed
            val_metrics_per_seed.append(val_metrics)
            
            # Compute test metrics
            if problem_type == "regression":
                test_metrics = compute_regression_metrics(
                    y_test, run['y_test_pred']
                )
            else:
                test_metrics = compute_classification_metrics(
                    y_test, run['y_test_pred'], run.get('y_test_proba')
                )
            test_metrics['seed'] = seed
            test_metrics_per_seed.append(test_metrics)
        
        # Create summary DataFrames
        val_df = pd.DataFrame(val_metrics_per_seed)
        test_df = pd.DataFrame(test_metrics_per_seed)
        
        # Print summary statistics
        print(f"\nValidation Set Performance (across {len(model_results)} seeds):")
        if problem_type == "regression":
            print(f"  R² = {val_df['r2'].mean():.4f} ± {safe_std(val_df['r2']):.4f}")
            print(f"  RMSE = {val_df['rmse'].mean():.4f} ± {safe_std(val_df['rmse']):.4f}")
            print(f"  MAE = {val_df['mae'].mean():.4f} ± {safe_std(val_df['mae']):.4f}")
        else:
            print(f"  Accuracy = {val_df['accuracy'].mean():.4f} ± {safe_std(val_df['accuracy']):.4f}")
            print(f"  Balanced Accuracy = {val_df['balanced_accuracy'].mean():.4f} ± {safe_std(val_df['balanced_accuracy']):.4f}")
            print(f"  F1 = {val_df['f1'].mean():.4f} ± {safe_std(val_df['f1']):.4f}")
        
        print(f"\nTest Set Performance (across {len(model_results)} seeds):")
        if problem_type == "regression":
            print(f"  R² = {test_df['r2'].mean():.4f} ± {safe_std(test_df['r2']):.4f}")
            print(f"  RMSE = {test_df['rmse'].mean():.4f} ± {safe_std(test_df['rmse']):.4f}")
            print(f"  MAE = {test_df['mae'].mean():.4f} ± {safe_std(test_df['mae']):.4f}")
        else:
            print(f"  Accuracy = {test_df['accuracy'].mean():.4f} ± {safe_std(test_df['accuracy']):.4f}")
            print(f"  Balanced Accuracy = {test_df['balanced_accuracy'].mean():.4f} ± {safe_std(test_df['balanced_accuracy']):.4f}")
            print(f"  F1 = {test_df['f1'].mean():.4f} ± {safe_std(test_df['f1']):.4f}")
        
        print(f"\nBest CV Score (during tuning): {np.mean(best_cv_scores):.4f} ± {safe_std(np.array(best_cv_scores)):.4f}")
        
        # Save detailed results
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save validation metrics
        val_csv = os.path.join(model_dir, f"metrics_{model_name}_validation.csv")
        val_df.to_csv(val_csv, index=False)
        print(f"\n  Validation metrics saved to: {val_csv}")
        
        # Save test metrics
        test_csv = os.path.join(model_dir, f"metrics_{model_name}_test.csv")
        test_df.to_csv(test_csv, index=False)
        print(f"  Test metrics saved to: {test_csv}")
        
        # Save hyperparameter summary
        params_df = pd.DataFrame(best_params_per_seed)
        params_csv = os.path.join(model_dir, f"best_params_{model_name}.csv")
        params_df.to_csv(params_csv, index=False)
        print(f"  Best parameters saved to: {params_csv}")
        
        # Create comparison plots
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
                
                # Color boxes
                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} - {model_name.upper()}')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_path = os.path.join(model_dir, f"val_vs_test_{model_name}.png")
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  Comparison plot saved to: {plot_path}")
        except Exception as e:
            print(f"  Warning: Could not create comparison plot: {e}")
        
        # Store summary
        all_summaries[model_name] = {
            'validation': val_df,
            'test': test_df,
            'best_params': best_params_per_seed,
            'best_cv_scores': best_cv_scores,
        }
    
    # Create summary report comparing all models
    summary_report = {}
    for model_name, summary in all_summaries.items():
        val_df = summary['validation']
        test_df = summary['test']
        
        if problem_type == "regression":
            summary_report[model_name] = {
                'val_r2_mean': float(val_df['r2'].mean()),
                'val_r2_std': float(safe_std(val_df['r2'])),
                'test_r2_mean': float(test_df['r2'].mean()),
                'test_r2_std': float(safe_std(test_df['r2'])),
                'val_rmse_mean': float(val_df['rmse'].mean()),
                'val_rmse_std': float(safe_std(val_df['rmse'])),
                'test_rmse_mean': float(test_df['rmse'].mean()),
                'test_rmse_std': float(safe_std(test_df['rmse'])),
            }
        else:
            summary_report[model_name] = {
                'val_accuracy_mean': float(val_df['accuracy'].mean()),
                'val_accuracy_std': float(safe_std(val_df['accuracy'])),
                'test_accuracy_mean': float(test_df['accuracy'].mean()),
                'test_accuracy_std': float(safe_std(test_df['accuracy'])),
                'val_f1_mean': float(val_df['f1'].mean()),
                'val_f1_std': float(safe_std(val_df['f1'])),
                'test_f1_mean': float(test_df['f1'].mean()),
                'test_f1_std': float(safe_std(test_df['f1'])),
            }
    
    # Save summary report
    summary_path = os.path.join(output_dir, "tuning_summary_all_models.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Summary report saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    return all_summaries
