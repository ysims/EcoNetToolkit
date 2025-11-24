#!/usr/bin/env python3
"""Run EcoNetToolkit: python run.py --config configs/example_config.yaml"""

import argparse
import numpy as np
import os

from ecosci.config import load_config
from ecosci.data import CSVDataLoader
from ecosci.models import ModelZoo
from ecosci.trainer import Trainer
from ecosci.evaluation import evaluate_and_report, evaluate_and_report_cv

parser = argparse.ArgumentParser(description="Train a model using a YAML config.")
parser.add_argument("--config", required=True, help="Path to YAML config file")
args = parser.parse_args()

cfg = load_config(args.config)

# Create output directory based on config name if not specified
if "dir" not in cfg.get("output", {}):
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg.setdefault("output", {})
    cfg["output"]["dir"] = os.path.join("outputs", config_name)

# Load and prepare data
data_cfg = cfg.get("data", {})
problem_type = cfg.get("problem_type", "classification")
cv_group_column = data_cfg.get("cv_group_column")

loader = CSVDataLoader(
    path=data_cfg.get("path"),
    features=data_cfg.get("features"),
    label=data_cfg.get("label"),
    labels=data_cfg.get("labels"),
    test_size=data_cfg.get("test_size", 0.2),
    val_size=data_cfg.get("val_size", 0.2),
    random_state=data_cfg.get("random_state", 0),
    scaling=data_cfg.get("scaling", "standard"),
    impute_strategy=data_cfg.get("impute_strategy", "mean"),
    problem_type=problem_type,
    cv_group_column=cv_group_column,
)

# Get output directory
output_dir = cfg.get("output", {}).get("dir", "outputs")

# Train
trainer = Trainer(
    ModelZoo.get_model,
    problem_type=cfg.get("problem_type", "classification"),
    output_dir=output_dir,
)

# Check if hyperparameter tuning is enabled
tuning_enabled = cfg.get("tuning", {}).get("enabled", False)

# Determine which mode to use
if tuning_enabled and cv_group_column is not None:
    # Hyperparameter tuning mode with grouped train/val/test splits
    n_train_groups = data_cfg.get("n_train_groups", 4)
    n_val_groups = data_cfg.get("n_val_groups", 2)
    n_test_groups = data_cfg.get("n_test_groups", 2)
    
    print(f"\n{'='*70}")
    print(f"Hyperparameter Tuning Mode")
    print(f"{'='*70}")
    print(f"Using grouped train/val/test splits with group column: {cv_group_column}")
    print(f"  Train groups: {n_train_groups}")
    print(f"  Val groups: {n_val_groups}")
    print(f"  Test groups: {n_test_groups}")
    print(f"{'='*70}\n")
    
    # Prepare grouped splits
    (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments, 
     groups_train, groups_val, groups_test) = \
        loader.prepare_grouped_splits(n_train_groups, n_val_groups, n_test_groups)
    
    # Run training with hyperparameter tuning
    results = trainer.run_with_tuning(
        cfg, X_train, X_val, X_test, y_train, y_val, y_test, group_assignments,
        groups_train, groups_val
    )
    
    # Evaluate on both validation and test sets
    # For tuning mode, we want to see performance on both val and test
    from ecosci.evaluation import evaluate_tuning_results
    summary = evaluate_tuning_results(
        results,
        y_val,
        y_test,
        output_dir=output_dir,
        problem_type=problem_type,
        label_names=loader.labels if hasattr(loader, 'labels') else None,
        feature_names=loader.processed_feature_names if hasattr(loader, 'processed_feature_names') else None,
        X_val=X_val,
        X_test=X_test,
    )
    
elif cv_group_column is not None:
    # K-fold cross-validation mode (no tuning)
    print(f"Running k-fold cross-validation using group column: {cv_group_column}")
    fold_data_list = loader.prepare_cv_folds()
    results = trainer.run_cv(cfg, fold_data_list)
    
    # Evaluate with CV-specific reporting
    summary = evaluate_and_report_cv(
        results,
        output_dir=output_dir,
        problem_type=problem_type,
        label_names=loader.labels if hasattr(loader, 'labels') else None,
        feature_names=loader.processed_feature_names if hasattr(loader, 'processed_feature_names') else None,
    )
else:
    # Regular train/test split (no tuning, no CV)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Evaluate
    summary = evaluate_and_report(
        results,
        y_test,
        output_dir=output_dir,
        problem_type=problem_type,
        label_names=loader.labels if hasattr(loader, 'labels') else None,
        feature_names=loader.processed_feature_names if hasattr(loader, 'processed_feature_names') else None,
        X_test=X_test,
    )

# Print quick summary
if tuning_enabled and cv_group_column is not None:
    # For tuning mode
    print(f"\n{'='*70}")
    print(f"Hyperparameter Tuning Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"  - Best hyperparameters per seed")
    print(f"  - Validation and test set predictions")
    print(f"  - Model checkpoints")
    print(f"{'='*70}\n")
elif cv_group_column is not None:
    # For CV, summary is a dict of DataFrames
    print(f"\nDone. See {output_dir}/ for full CV reports and plots.")
else:
    # For regular split, summary is a list of dicts
    if problem_type == "regression":
        r2s = [r.get("r2") for r in summary if "r2" in r]
        if r2s:
            print(f"\nMean R²: {np.mean(r2s):.3f}")
        rmses = [r.get("rmse") for r in summary if "rmse" in r]
        if rmses:
            print(f"Mean RMSE: {np.mean(rmses):.3f}")
    else:
        accs = [r.get("accuracy") for r in summary if "accuracy" in r]
        if accs:
            print(f"\nMean accuracy: {np.mean(accs):.3f}")
    print(f"Done. See {output_dir}/ for full report and plots.")
