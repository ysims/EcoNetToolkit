#!/usr/bin/env python3
"""Inspect saved model files (.joblib) and print a summary table.

Usage:
    python inspect_models.py outputs/model_*.joblib
    python inspect_models.py outputs/model_mlp_seed0.joblib outputs/model_mlp_seed1.joblib
"""
import argparse
import joblib
import os
import pandas as pd


def inspect_model(path):
    """Load a model and extract metadata."""
    model = joblib.load(path)
    info = {
        "file": os.path.basename(path),
        "type": type(model).__name__,
    }
    
    # Extract model-specific info
    if hasattr(model, "n_iter_"):
        info["n_iter"] = model.n_iter_
    if hasattr(model, "n_layers_"):
        info["n_layers"] = model.n_layers_
    if hasattr(model, "n_features_in_"):
        info["n_features_in"] = model.n_features_in_
    if hasattr(model, "classes_"):
        info["n_classes"] = len(model.classes_)
    if hasattr(model, "best_loss_") and model.best_loss_ is not None:
        info["best_loss"] = f"{model.best_loss_:.4f}"
    
    # For tree-based models
    if hasattr(model, "n_estimators"):
        info["n_estimators"] = model.n_estimators
    if hasattr(model, "max_depth") and model.max_depth is not None:
        info["max_depth"] = model.max_depth
    
    return info


parser = argparse.ArgumentParser(description="Inspect saved model files.")
parser.add_argument("models", nargs="+", help="Paths to .joblib model files")
args = parser.parse_args()

results = []
for path in args.models:
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue
    try:
        info = inspect_model(path)
        results.append(info)
    except Exception as e:
        print(f"Error loading {path}: {e}")

if results:
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
else:
    print("No models loaded.")
