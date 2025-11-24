"""Output and file saving utilities for evaluation reports."""

import os
import json
import pandas as pd
from ..metrics import safe_std


def save_cv_reports(model_name, all_fold_metrics, fold_summaries, folds, overall_df, output_dir):
    """Save CV reports and metrics to files.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    all_fold_metrics : list
        All metrics across folds
    fold_summaries : list
        Per-fold summaries
    folds : dict
        Dictionary of fold results
    overall_df : DataFrame
        Overall results DataFrame
    output_dir : str
        Output directory
    """
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


def save_tuning_results(model_name, val_df, test_df, best_params_per_seed, output_dir):
    """Save hyperparameter tuning results to files.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    val_df : DataFrame
        Validation metrics
    test_df : DataFrame
        Test metrics
    best_params_per_seed : list
        Best parameters for each seed
    output_dir : str
        Output directory
    """
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


def save_tuning_summary_report(all_summaries, output_dir, problem_type):
    """Save summary report comparing all models.
    
    Parameters
    ----------
    all_summaries : dict
        Dictionary of summaries for all models
    output_dir : str
        Output directory
    problem_type : str
        "classification" or "regression"
    """
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
