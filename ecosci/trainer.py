"""Training loop for repeated runs and reporting.

What this does:
- Repeats training for a list of seeds (or a count), so you get stable results
- Uses scikit-learn's built-in early stopping for the MLP if enabled in config
- Saves each trained model as a `.joblib` file
- Returns predictions per run so the evaluator can compute metrics
"""

from typing import Any, Dict, List
import numpy as np
import random
import joblib
import os


class Trainer:
    """Orchestrates model training for one config.

    Parameters
    ----------
    model_factory : callable
        Something like `ModelZoo.get_model` that returns an sklearn-like model.
    problem_type : str
        "classification" or "regression".
    output_dir : str
        Where to save trained models and reports.
    """

    def __init__(
        self,
        model_factory,
        problem_type: str = "classification",
        output_dir: str = "outputs",
    ):
        self.model_factory = model_factory
        self.problem_type = problem_type
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _set_seed(self, s: int):
        np.random.seed(s)
        random.seed(s)

    def run(self, cfg: Dict[str, Any], X_train, X_val, X_test, y_train, y_val, y_test):
        """Run training according to cfg.

        Config keys used
        ----------------
        - models (list of model configs, each with name and params)
        - model (single model config, for backward compatibility)
        - training.seeds (optional, list)
        - training.repetitions (if `seeds` not provided)
        - training.random_seed (base for repetitions)
        """
        models_cfg = cfg.get("models", [])

        seeds = cfg.get("training", {}).get("seeds")
        repetitions = cfg.get("training", {}).get("repetitions", 1)
        base_seed = cfg.get("training", {}).get("random_seed", 0)

        seeds_list = []
        if seeds:
            seeds_list = seeds
        else:
            seeds_list = [base_seed + i for i in range(repetitions)]

        all_results = {}

        # Train each model type
        for model_idx, model_cfg in enumerate(models_cfg):
            mname = model_cfg.get("name", "mlp")
            mparams = model_cfg.get("params", {})

            print(f"\n{'='*80}")
            print(f"Training Model {model_idx+1}/{len(models_cfg)}: {mname.upper()}")
            print(f"{'='*80}")

            model_results = []

            for i, s in enumerate(seeds_list):
                print(f"  Run {i+1}/{len(seeds_list)} with seed={s}")
                self._set_seed(s)

                # ensure model has random_state where appropriate
                mparams_local = dict(mparams)
                if "random_state" in mparams_local or True:
                    mparams_local.setdefault("random_state", s)

                model = self.model_factory(mname, self.problem_type, mparams_local)

                # fit: sklearn models have fit(X, y). For MLP, early_stopping is
                # handled internally if enabled via params in the YAML config.
                if (
                    X_val is not None
                    and hasattr(model, "partial_fit")
                    and cfg.get("training", {}).get("use_partial_fit", False)
                ):
                    # not used by default; kept minimal
                    model.partial_fit(X_train, y_train)
                else:
                    # For classifiers that require y as 1d array, ensure shape
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = None
                # Only try to get probabilities for classification problems
                if self.problem_type == "classification" and hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception as e:
                        print(f"    Warning: predict_proba failed: {e}")
                        y_proba = None

                # save model for this run
                fname = os.path.join(self.output_dir, f"model_{mname}_seed{s}.joblib")
                joblib.dump(model, fname)

                model_results.append(
                    {
                        "seed": s,
                        "model_name": mname,
                        "model_path": fname,
                        "y_pred": y_pred,
                        "y_proba": y_proba,
                    }
                )

            all_results[mname] = model_results

        return all_results
