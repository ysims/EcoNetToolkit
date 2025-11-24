"""Main evaluation orchestration functions.

This module contains the high-level functions that orchestrate metrics computation,
feature importance analysis, and plot generation.
"""

from typing import Dict, Optional
import os
import json
import numpy as np
import pandas as pd

from ..metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_multi_output_classification_metrics,
    compute_multi_output_regression_metrics,
)
from ..plotting import (
    plot_metric_comparisons,
    plot_confusion_matrices,
    plot_pr_curves,
    plot_residuals,
    plot_cv_comparison,
    plot_validation_vs_test,
)
from ..feature_importance import (
    extract_feature_importance,
    extract_cv_feature_importance,
)
from .display import (
    get_metric_columns,
    print_model_results,
    print_model_comparison,
    print_model_ranking,
    get_cv_metric_keys,
    print_cv_fold_results,
    print_tuning_summary,
)
from .output import (
    save_cv_reports,
    save_tuning_results,
    save_tuning_summary_report,
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
        Names of the target variables (for multi-output).
    feature_names : list, optional
        Names of the input features.
    X_test : array-like, optional
        Test features for permutation importance.
    
    Returns
    -------
    list
        Combined summary of results across all models
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle backward compatibility: convert list to dict
    if isinstance(results, list):
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
                if problem_type == "regression":
                    metrics = compute_multi_output_regression_metrics(y_test, y_pred)
                else:
                    metrics = compute_multi_output_classification_metrics(y_test, y_pred, y_proba)
            else:
                if problem_type == "regression":
                    metrics = compute_regression_metrics(y_test, y_pred)
                else:
                    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

            metrics["seed"] = seed
            metrics["model"] = model_name
            summary.append(metrics)

        all_summaries[model_name] = summary
        all_dfs[model_name] = pd.DataFrame(summary)

    # Save individual model reports
    for model_name, summary in all_summaries.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        report_path = os.path.join(model_dir, f"report_{model_name}.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Combine all results
    combined_df = pd.concat(all_dfs.values(), ignore_index=True)
    combined_summary = [item for sublist in all_summaries.values() for item in sublist]

    # Save combined report
    with open(os.path.join(output_dir, "report_all_models.json"), "w") as f:
        json.dump(combined_summary, f, indent=2)

    # Print results
    display_cols, comparison_metrics, plot_metrics = get_metric_columns(
        problem_type, is_multi_output, combined_df
    )

    print_model_results(all_dfs, display_cols, is_multi_output, label_names)
    
    if len(all_dfs) > 1:
        print_model_comparison(all_dfs, comparison_metrics)
        print_model_ranking(all_dfs, display_cols, problem_type, is_multi_output)

    # Create plots
    plot_metric_comparisons(combined_df, plot_metrics, output_dir, all_dfs)
    
    if problem_type == "classification":
        plot_confusion_matrices(results, y_test, output_dir)
        plot_pr_curves(results, y_test, output_dir)
    elif problem_type == "regression" and not is_multi_output:
        plot_residuals(results, y_test, output_dir)

    # Feature importance
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
    output_dir : str
        Where to save the report and plots.
    problem_type : str
        "classification" or "regression".
    label_names : list, optional
        Names of the target variables.
    feature_names : list, optional
        Names of the input features.
    X_test : array-like, optional
        Test features (note: in CV, each fold has different test set).
        
    Returns
    -------
    dict
        Summaries for all models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}\n")
    
    # Determine if multi-output
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
                else:
                    if is_multi_output:
                        metrics = compute_multi_output_regression_metrics(y_test, y_pred)
                    else:
                        metrics = compute_regression_metrics(y_test, y_pred)
                
                metrics["seed"] = seed
                metrics["fold"] = fold_id
                fold_metrics_list.append(metrics)
            
            fold_summaries.append(fold_metrics_list)
        
        # Print per-fold metrics
        metric_keys = get_cv_metric_keys(problem_type, is_multi_output)
        print_cv_fold_results(folds, fold_summaries, metric_keys)
        
        # Compute overall metrics
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
        
        # Save reports
        save_cv_reports(model_name, all_fold_metrics, fold_summaries, folds, overall_df, output_dir)
        
        all_summaries[model_name] = overall_df
        all_dfs[model_name] = overall_df
    
    # Create plots
    plot_cv_comparison(all_dfs, output_dir, problem_type, is_multi_output)
    
    print(f"\n{'='*80}\n")
    
    # Feature importance
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
        Names of features
    X_val : np.ndarray, optional
        Validation features
    X_test : np.ndarray, optional
        Test features
    
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
                val_metrics = compute_regression_metrics(y_val, run['y_val_pred'])
            else:
                val_metrics = compute_classification_metrics(
                    y_val, run['y_val_pred'], run.get('y_val_proba')
                )
            val_metrics['seed'] = seed
            val_metrics_per_seed.append(val_metrics)
            
            # Compute test metrics
            if problem_type == "regression":
                test_metrics = compute_regression_metrics(y_test, run['y_test_pred'])
            else:
                test_metrics = compute_classification_metrics(
                    y_test, run['y_test_pred'], run.get('y_test_proba')
                )
            test_metrics['seed'] = seed
            test_metrics_per_seed.append(test_metrics)
        
        # Create summary DataFrames
        val_df = pd.DataFrame(val_metrics_per_seed)
        test_df = pd.DataFrame(test_metrics_per_seed)
        
        # Print summary
        print_tuning_summary(val_df, test_df, best_cv_scores, problem_type, len(model_results))
        
        # Save results
        save_tuning_results(model_name, val_df, test_df, best_params_per_seed, output_dir)
        
        # Create plots
        plot_validation_vs_test(val_df, test_df, model_name, output_dir, problem_type)
        
        # Store summary
        all_summaries[model_name] = {
            'validation': val_df,
            'test': test_df,
            'best_params': best_params_per_seed,
            'best_cv_scores': best_cv_scores,
        }
    
    # Save summary report
    save_tuning_summary_report(all_summaries, output_dir, problem_type)
    
    return all_summaries
