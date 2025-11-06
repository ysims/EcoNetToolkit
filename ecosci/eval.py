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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, average_precision_score

    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, average="binary" if len(np.unique(y_true))==2 else "macro", zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true))==2 else "macro", zero_division=0)
    out["f1"] = f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true))==2 else "macro", zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if y_proba is not None:
        try:
            # if multiclass, use macro AUC when possible
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                out["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                out["average_precision"] = average_precision_score(y_true, y_proba, average="macro")
            else:
                out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
                out["average_precision"] = average_precision_score(y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
        except Exception:
            out["roc_auc"] = None
            out["average_precision"] = None

    return out


def evaluate_and_report(results, y_test, output_dir: str = "outputs"):
    """Compute metrics across runs and save a compact report and plots.

    Parameters
    ----------
    results : list
        Output from Trainer.run() containing predictions and optional probabilities.
    y_test : array-like
        Ground-truth labels for the test set.
    output_dir : str
        Where to save the report and plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    for r in results:
        seed = r.get("seed")
        y_pred = r.get("y_pred")
        y_proba = r.get("y_proba")

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        metrics["seed"] = seed
        summary.append(metrics)

    # save JSON summary
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # simple plots: metric boxplot across runs
    import pandas as pd

    df = pd.DataFrame(summary)
    plot_metrics = ["accuracy", "balanced_accuracy", "f1"]
    for m in plot_metrics:
        if m in df.columns:
            plt.figure(figsize=(4, 3))
            sns.boxplot(y=df[m])
            plt.title(m)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"metric_{m}.png"))
            plt.close()

    # Confusion matrix for the last run (for a quick glance)
    try:
        from sklearn.metrics import confusion_matrix
        last = results[-1]
        cm = confusion_matrix(y_test, last.get("y_pred"))
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (last run)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
    except Exception:
        pass

    # Precision-Recall curve for binary classification if probabilities available
    try:
        import numpy as np
        from sklearn.metrics import precision_recall_curve, average_precision_score
        last = results[-1]
        y_proba = last.get("y_proba")
        if y_proba is not None:
            # if proba has shape (n,2), take positive class
            if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2:
                pos_scores = y_proba[:, 1]
            else:
                pos_scores = y_proba
            precision, recall, _ = precision_recall_curve(y_test, pos_scores)
            ap = average_precision_score(y_test, pos_scores)
            plt.figure(figsize=(4, 3))
            plt.plot(recall, precision, label=f"AP={ap:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall (last run)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pr_curve.png"))
            plt.close()
    except Exception:
        pass

    return summary
