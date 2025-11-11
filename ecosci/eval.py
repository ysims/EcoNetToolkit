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
    results, y_test, output_dir: str = "outputs", problem_type: str = "classification"
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
    """
    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd

    # Handle backward compatibility: convert list to dict
    if isinstance(results, list):
        # Assume single model (old format)
        model_name = results[0].get("model_name", "model") if results else "model"
        results = {model_name: results}

    # Compute metrics for each model
    all_summaries = {}
    all_dfs = {}

    for model_name, model_results in results.items():
        summary = []
        for r in model_results:
            seed = r.get("seed")
            y_pred = r.get("y_pred")
            y_proba = r.get("y_proba")

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
                    model_scores.append({
                        'model': model_name,
                        'mean': vals.mean(),
                        'std': safe_std(vals)
                    })
        
        if len(model_scores) > 0:
            # Sort by primary metric
            model_scores.sort(key=lambda x: x['mean'], reverse=not lower_is_better)
            
            print(f"\n{'='*80}")
            print("MODEL RANKING")
            print(f"{'='*80}")
            print(f"Ranked by: {primary_metric.replace('_', ' ').upper()} ({'Lower is better' if lower_is_better else 'Higher is better'})")
            print(f"{'-'*80}")
            
            # Create ranking table with all metrics
            ranking_data = []
            for rank, score in enumerate(model_scores, 1):
                model_name = score['model']
                model_df = all_dfs[model_name]
                
                row = {
                    "Rank": f"{rank}.",
                    "Model": model_name.upper(),
                    primary_metric: f"{score['mean']:.4f} ± {score['std']:.4f}"
                }
                
                # Add other metrics
                metric_cols = [c for c in display_cols if c != "seed" and c != primary_metric]
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
    elif problem_type == "regression":
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

    return combined_summary
