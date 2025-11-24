"""Display and formatting utilities for evaluation reports."""

import numpy as np
import pandas as pd
from ..metrics import safe_std


def get_metric_columns(problem_type, is_multi_output, combined_df):
    """Determine which metric columns to display based on problem type.
    
    Parameters
    ----------
    problem_type : str
        "classification" or "regression"
    is_multi_output : bool
        Whether the problem has multiple outputs
    combined_df : DataFrame
        Combined results DataFrame
        
    Returns
    -------
    tuple
        (display_cols, comparison_metrics, plot_metrics)
    """
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


def print_model_results(all_dfs, display_cols, is_multi_output, label_names):
    """Print results for each model.
    
    Parameters
    ----------
    all_dfs : dict
        Dictionary mapping model names to DataFrames with results
    display_cols : list
        Columns to display
    is_multi_output : bool
        Whether the problem has multiple outputs
    label_names : list or None
        Names of output labels
    """
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
            print_per_output_breakdown(df, label_names, model_name)


def print_per_output_breakdown(df, label_names, model_name):
    """Print per-output breakdown for multi-output problems.
    
    Parameters
    ----------
    df : DataFrame
        Results DataFrame
    label_names : list or None
        Names of output labels
    model_name : str
        Name of the model
    """
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


def print_model_comparison(all_dfs, comparison_metrics):
    """Print model comparison table.
    
    Parameters
    ----------
    all_dfs : dict
        Dictionary mapping model names to DataFrames
    comparison_metrics : list
        Metrics to compare
    """
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


def print_model_ranking(all_dfs, display_cols, problem_type, is_multi_output):
    """Print model ranking based on primary metric.
    
    Parameters
    ----------
    all_dfs : dict
        Dictionary mapping model names to DataFrames
    display_cols : list
        Columns to display
    problem_type : str
        "classification" or "regression"
    is_multi_output : bool
        Whether the problem has multiple outputs
    """
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


def get_cv_metric_keys(problem_type, is_multi_output):
    """Get metric keys for CV results.
    
    Parameters
    ----------
    problem_type : str
        "classification" or "regression"
    is_multi_output : bool
        Whether the problem has multiple outputs
        
    Returns
    -------
    list
        List of metric keys
    """
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


def print_cv_fold_results(folds, fold_summaries, metric_keys):
    """Print per-fold CV results.
    
    Parameters
    ----------
    folds : dict
        Dictionary of fold results
    fold_summaries : list
        List of fold summaries
    metric_keys : list
        Metric keys to display
    """
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


def print_tuning_summary(val_df, test_df, best_cv_scores, problem_type, n_seeds):
    """Print summary of hyperparameter tuning results.
    
    Parameters
    ----------
    val_df : DataFrame
        Validation metrics
    test_df : DataFrame
        Test metrics
    best_cv_scores : list
        Best CV scores across seeds
    problem_type : str
        "classification" or "regression"
    n_seeds : int
        Number of seeds
    """
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
