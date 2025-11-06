"""Simple CLI to run the pipeline from a YAML config.

Usage
-----
python scripts/run.py --config configs/example_config.yaml
"""
import argparse
from ecosci.config import load_config
from ecosci.data import CSVDataLoader
from ecosci.models import ModelZoo
from ecosci.trainer import Trainer
from ecosci.eval import evaluate_and_report


def main():
    parser = argparse.ArgumentParser(description="Train a model using a YAML config (no coding required).")
    parser.add_argument("--config", required=True, help="Path to YAML config file (see configs/example_config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    loader = CSVDataLoader(
        path=data_cfg.get("path"),
        features=data_cfg.get("features"),
        label=data_cfg.get("label"),
        test_size=data_cfg.get("test_size", 0.2),
        val_size=data_cfg.get("val_size", 0.2),
        random_state=data_cfg.get("random_state", 0),
        scaling=data_cfg.get("scaling", "standard"),
        impute_strategy=data_cfg.get("impute_strategy", "mean"),
    )

    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()

    trainer = Trainer(ModelZoo.get_model, problem_type=cfg.get("problem_type", "classification"), output_dir=cfg.get("output_dir", "outputs"))

    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)

    summary = evaluate_and_report(results, y_test, output_dir=cfg.get("output_dir", "outputs"))

    # Print a tiny summary in the console (mean over seeds)
    try:
        import numpy as np
        accs = [r.get("accuracy") for r in summary if "accuracy" in r]
        bals = [r.get("balanced_accuracy") for r in summary if "balanced_accuracy" in r]
        f1s = [r.get("f1") for r in summary if "f1" in r]
        if accs:
            print(f"Mean accuracy over {len(accs)} runs: {np.mean(accs):.3f}")
        if bals:
            print(f"Mean balanced accuracy: {np.mean(bals):.3f}")
        if f1s:
            print(f"Mean F1: {np.mean(f1s):.3f}")
    except Exception:
        pass

    print("Finished. Outputs written to:", cfg.get("output_dir", "outputs"))


if __name__ == "__main__":
    main()
