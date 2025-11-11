#!/usr/bin/env python3
"""Run EcoNetToolkit: python run.py --config configs/example_config.yaml"""

import argparse
import numpy as np

from ecosci.config import load_config
from ecosci.data import CSVDataLoader
from ecosci.models import ModelZoo
from ecosci.trainer import Trainer
from ecosci.eval import evaluate_and_report

parser = argparse.ArgumentParser(description="Train a model using a YAML config.")
parser.add_argument("--config", required=True, help="Path to YAML config file")
args = parser.parse_args()

cfg = load_config(args.config)

# Create output directory based on config name if not specified
if "output_dir" not in cfg.get("output", {}):
    import os

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg.setdefault("output", {})
    cfg["output"]["dir"] = os.path.join("outputs", config_name)

# Load and prepare data
data_cfg = cfg.get("data", {})
problem_type = cfg.get("problem_type", "classification")
loader = CSVDataLoader(
    path=data_cfg.get("path"),
    features=data_cfg.get("features"),
    label=data_cfg.get("label"),
    test_size=data_cfg.get("test_size", 0.2),
    val_size=data_cfg.get("val_size", 0.2),
    random_state=data_cfg.get("random_state", 0),
    scaling=data_cfg.get("scaling", "standard"),
    impute_strategy=data_cfg.get("impute_strategy", "mean"),
    problem_type=problem_type,
)
X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()

# Get output directory
output_dir = cfg.get("output", {}).get("dir", "outputs")

# Train
trainer = Trainer(
    ModelZoo.get_model,
    problem_type=cfg.get("problem_type", "classification"),
    output_dir=output_dir,
)
results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)

# Evaluate
problem_type = cfg.get("problem_type", "classification")
summary = evaluate_and_report(
    results,
    y_test,
    output_dir=output_dir,
    problem_type=problem_type,
)

# Print quick summary
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
