"""Reporting and evaluation utilities.

This module contains the main evaluation and reporting functions that orchestrate
metrics computation, feature importance analysis, and plot generation.

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
import pandas as pd

from .metrics import (
    safe_std,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_multi_output_classification_metrics,
    compute_multi_output_regression_metrics,
)
from .plotting import (
    plot_metric_comparisons,
    plot_confusion_matrices,
    plot_pr_curves,
    plot_residuals,
    plot_feature_importance,
    plot_cv_comparison,
    plot_validation_vs_test,
)
from .feature_importance import (
    extract_feature_importance,
    extract_cv_feature_importance,
)


def evaluate_and_report(
    results, y_test, output_dir: str = "outputs", problem_type: str = "classification", 
    label_names: list = None, feature_names: list = None, X_test=None
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
    display_cols, comparison_metrics, plot_metrics = _get_metric_columns(
        problem_type, is_multi_output, combined_df
    )

    _print_model_results(all_dfs, display_cols, is_multi_output, label_names)
    
    if len(all_dfs) > 1:
        _print_model_comparison(all_dfs, comparison_metrics)
        _print_model_ranking(all_dfs, display_cols, problem_type, is_multi_output)

    # Create plots
    plot_metric_comparisons(combined_df, plot_metrics, output_dir, all_dfs)
    
    if problem_type == "classification":
        plot_confusion_matrices(results, y_test, output_dir)
        plot_pr_curves(results, y_test, output_dir)
    elif problem_type == "regression" and not is_multi_output:
        plot_residuals(results, y_test, output_dir)

    # Feature importance analysis
    extract_feature_importance(results, X_test, y_test, output_dir, problem_type, feature_names)

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
        
        # Print per-fold metrics
        metric_keys = _get_cv_metric_keys(problem_type, is_multi_output)
        _print_cv_fold_results(folds, fold_summaries, metric_keys)
        
        # Compute overall metrics (averaged across all folds and seeds)
        all_fold_metrics = [m for fold_list in fold_summaries for m in fold_list]
        overall_df = pd.DataFrame(all_fold_metrics)
        
        # Print overall metrics
        print(f"\n  {'Overall':<6}", end="")
        for mk in metric_keys:
            if mk in overall_df.columns:
                mean_val = overall_df[mk].mean()
                std_val = overall_df[mk].std()
                print(f"{mean_val:.4f} ± {std_val:.4f}    ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print("\n")
        
        # Save reports and metrics
        _save_cv_reports(model_name, all_fold_metrics, fold_summaries, folds, overall_df, output_dir)
        
        all_summaries[model_name] = overall_df
        all_dfs[model_name] = overall_df
    
    # Create comparison plots across folds and models
    plot_cv_comparison(all_dfs, output_dir, problem_type, is_multi_output)
    
    print(f"\n{'='*80}\n")
    
    # Feature importance for CV
    extract_cv_feature_importance(results, output_dir, problem_type, feature_names)
    
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
    after hyperparameter tuning. It produces reports and visualisations comparing
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
        _print_tuning_summary(val_df, test_df, best_cv_scores, problem_type, len(model_results))
        
        # Save detailed results
        _save_tuning_results(model_name, val_df, test_df, best_params_per_seed, output_dir)
        
        # Create comparison plots
        plot_validation_vs_test(val_df, test_df, model_name, output_dir, problem_type)
        
        # Store summary
        all_summaries[model_name] = {
            'validation': val_df,
            'test': test_df,
            'best_params': best_params_per_seed,
            'best_cv_scores': best_cv_scores,
        }
    
    # Create summary report comparing all models
    _save_tuning_summary_report(all_summaries, output_dir, problem_type)
    
    return all_summaries


# Helper functions

def _get_metric_columns(problem_type, is_multi_output, combined_df):
    """Determine which metric columns to display based on problem type."""
    if is_multi_output:
        if problem_type == "regression":
            display_cols = ["seed", "mse_mean", "rmse_mean", "mae_mean", "r2_mean"]
            comparison_metrics = ["mse_mean", "rmse_mean", "mae_mean", "r2_mean"]
            plot_metrics = ["mse_mean", "rmse_mean", "mae_mean", "r2_mean"]
        else:
            display_cols = [
                "seed", "accuracy_mean", "balanced_accuracy_mean",
                "precision_mean", "recall_mean", "f1_mean", "cohen_kappa_mean",
            ]
            comparison_metrics = [
                "accuracy_mean", "balanced_accuracy_mean", "f1_mean", "cohen_kappa_mean",
            ]
            plot_metrics = ["accuracy_mean", "balanced_accuracy_mean", "f1_mean", "cohen_kappa_mean"]
    else:
        if problem_type == "regression":
            display_cols = ["seed", "mse", "rmse", "mae", "r2"]
            if "mape" in combined_df.columns:
                display_cols.append("mape")
            comparison_metrics = ["mse", "rmse", "mae", "r2"]
            plot_metrics = ["mse", "rmse", "mae", "r2"]
        else:
            display_cols = [
                "seed", "accuracy", "balanced_accuracy",
                "precision", "recall", "f1", "cohen_kappa",
            ]
            if "roc_auc" in combined_df.columns:
                display_cols.append("roc_auc")
            if "average_precision" in combined_df.columns:
                display_cols.append("average_precision")
            comparison_metrics = [
                "accuracy", "balanced_accuracy", "f1", "cohen_kappa", "roc_auc",
            ]
            plot_metrics = ["accuracy", "balanced_accuracy", "f1", "cohen_kappa"]
    
    return display_cols, comparison_metrics, plot_metrics


def _print_model_results(all_dfs, display_cols, is_multi_output, label_names):
    """Print results for each model."""
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
        if is_multi_output and len(df) > 0 and "per_output" in df.iloc[0]:
            _print_per_output_breakdown(df, label_names, model_name)


def _print_per_output_breakdown(df, label_names, model_name):
    """Print per-output breakdown for multi-output problems."""
    print(f"\n{'-'*80}")
    print(f"PER-OUTPUT BREAKDOWN FOR {model_name.upper()}")
    print(f"{'-'*80}")
    
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


def _print_model_comparison(all_dfs, comparison_metrics):
    """Print model comparison table."""
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


def _print_model_ranking(all_dfs, display_cols, problem_type, is_multi_output):
    """Print model ranking based on primary metric."""
    # Determine primary metric
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
                model_scores.append({
                    "model": model_name,
                    "mean": vals.mean(),
                    "std": safe_std(vals),
                })

    if len(model_scores) > 0:
        # Sort by primary metric
        model_scores.sort(key=lambda x: x["mean"], reverse=not lower_is_better)

        print(f"\n{'='*80}")
        print("MODEL RANKING")
        print(f"{'='*80}")
        print(
            f"Ranked by: {primary_metric.replace('_', ' ').upper()} "
            f"({'Lower is better' if lower_is_better else 'Higher is better'})"
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


def _get_cv_metric_keys(problem_type, is_multi_output):
    """Get metric keys for CV results."""
    if problem_type == "classification":
        if is_multi_output:
            return ["accuracy_mean", "balanced_accuracy_mean", "f1_mean"]
        else:
            return ["accuracy", "balanced_accuracy", "f1"]
    else:
        if is_multi_output:
            return ["r2_mean", "rmse_mean", "mae_mean"]
        else:
            return ["r2", "rmse", "mae"]


def _print_cv_fold_results(folds, fold_summaries, metric_keys):
    """Print per-fold CV results."""
    print(f"\nPer-Fold Metrics (averaged across seeds per fold):")
    
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


def _save_cv_reports(model_name, all_fold_metrics, fold_summaries, folds, overall_df, output_dir):
    """Save CV reports and metrics to files."""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save detailed report
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


def _print_tuning_summary(val_df, test_df, best_cv_scores, problem_type, n_seeds):
    """Print summary of hyperparameter tuning results."""
    print(f"\nValidation Set Performance (across {n_seeds} seeds):")
    if problem_type == "regression":
        print(f"  R² = {val_df['r2'].mean():.4f} ± {safe_std(val_df['r2']):.4f}")
        print(f"  RMSE = {val_df['rmse'].mean():.4f} ± {safe_std(val_df['rmse']):.4f}")
        print(f"  MAE = {val_df['mae'].mean():.4f} ± {safe_std(val_df['mae']):.4f}")
    else:
        print(f"  Accuracy = {val_df['accuracy'].mean():.4f} ± {safe_std(val_df['accuracy']):.4f}")
        print(f"  Balanced Accuracy = {val_df['balanced_accuracy'].mean():.4f} ± {safe_std(val_df['balanced_accuracy']):.4f}")
        print(f"  F1 = {val_df['f1'].mean():.4f} ± {safe_std(val_df['f1']):.4f}")
    
    print(f"\nTest Set Performance (across {n_seeds} seeds):")
    if problem_type == "regression":
        print(f"  R² = {test_df['r2'].mean():.4f} ± {safe_std(test_df['r2']):.4f}")
        print(f"  RMSE = {test_df['rmse'].mean():.4f} ± {safe_std(test_df['rmse']):.4f}")
        print(f"  MAE = {test_df['mae'].mean():.4f} ± {safe_std(test_df['mae']):.4f}")
    else:
        print(f"  Accuracy = {test_df['accuracy'].mean():.4f} ± {safe_std(test_df['accuracy']):.4f}")
        print(f"  Balanced Accuracy = {test_df['balanced_accuracy'].mean():.4f} ± {safe_std(test_df['balanced_accuracy']):.4f}")
        print(f"  F1 = {test_df['f1'].mean():.4f} ± {safe_std(test_df['f1']):.4f}")
    
    print(f"\nBest CV Score (during tuning): {np.mean(best_cv_scores):.4f} ± {safe_std(np.array(best_cv_scores)):.4f}")


def _save_tuning_results(model_name, val_df, test_df, best_params_per_seed, output_dir):
    """Save hyperparameter tuning results to files."""
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


def _save_tuning_summary_report(all_summaries, output_dir, problem_type):
    """Save summary report comparing all models."""
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
