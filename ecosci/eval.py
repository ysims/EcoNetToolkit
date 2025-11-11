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


def evaluate_and_report(results, y_test, output_dir: str = "outputs"):
    """Compute metrics across runs and save a compact report and plots.

    Parameters
    ----------
    results : dict or list
        If dict: Output from Trainer.run() with multiple models {model_name: [run_results]}
        If list: Single model results (backward compatibility)
    y_test : array-like
        Ground-truth labels for the test set.
    output_dir : str
        Where to save the report and plots.
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

            metrics = compute_classification_metrics(y_test, y_pred, y_proba)
            metrics["seed"] = seed
            metrics["model"] = model_name
            summary.append(metrics)

        all_summaries[model_name] = summary
        all_dfs[model_name] = pd.DataFrame(summary)

    # Save individual model reports
    for model_name, summary in all_summaries.items():
        report_path = os.path.join(output_dir, f"report_{model_name}.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Combine all results for comparison
    combined_df = pd.concat(all_dfs.values(), ignore_index=True)
    combined_summary = [item for sublist in all_summaries.values() for item in sublist]

    # Save combined report
    with open(os.path.join(output_dir, "report_all_models.json"), "w") as f:
        json.dump(combined_summary, f, indent=2)

    # Print results for each model
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
                    print(f"{col:20s}: {vals.mean():.4f} ± {vals.std():.4f}")
        print(f"{'-'*80}\n")

    # Model comparison table
    if len(all_dfs) > 1:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON (mean ± std)")
        print(f"{'='*80}")
        comparison_data = []
        for model_name, df in all_dfs.items():
            row = {"Model": model_name}
            for col in [
                "accuracy",
                "balanced_accuracy",
                "f1",
                "cohen_kappa",
                "roc_auc",
            ]:
                if col in df.columns:
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        row[col] = f"{vals.mean():.4f} ± {vals.std():.4f}"
                    else:
                        row[col] = "N/A"
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        print(f"{'='*80}\n")

    # Plot comparison across models
    plot_metrics = ["accuracy", "balanced_accuracy", "f1", "cohen_kappa"]

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

    # Confusion matrices for each model (last run)
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
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create confusion matrix for {model_name}: {e}")

    # Precision-Recall curves for each model
    for model_name, model_results in results.items():
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score

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
                plt.savefig(os.path.join(output_dir, f"pr_curve_{model_name}.png"))
                plt.close()
        except Exception as e:
            pass  # Silently skip if PR curve can't be generated

    # Combined PR curve for model comparison (binary classification only)
    if len(results) > 1:
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score

            plt.figure(figsize=(7, 5))
            for model_name, model_results in results.items():
                last = model_results[-1]
                y_proba = last.get("y_proba")
                if y_proba is not None:
                    if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
                        pos_scores = y_proba[:, 1]
                    else:
                        pos_scores = y_proba
                    precision, recall, _ = precision_recall_curve(y_test, pos_scores)
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

    return combined_summary
