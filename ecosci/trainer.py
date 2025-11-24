"""Training loop for repeated runs and reporting.

What this does:
- Repeats training for a list of seeds (or a count), so you get stable results
- Uses scikit-learn's built-in early stopping for the MLP if enabled in config
- Saves each trained model as a `.joblib` file
- Returns predictions per run so the evaluator can compute metrics
- Supports hyperparameter tuning with train/val/test splits
"""

from typing import Any, Dict, Optional
from tqdm import tqdm
import numpy as np
import random
import joblib
import os
import json


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
        
        # Determine number of outputs from y_train shape
        if len(y_train.shape) == 1:
            n_outputs = 1
        else:
            n_outputs = y_train.shape[1]

        # Train each model type
        for model_idx, model_cfg in enumerate(models_cfg):
            mname = model_cfg.get("name", "mlp")
            mparams = model_cfg.get("params", {})

            model_results = []

            # Use tqdm for progress bar
            for s in tqdm(seeds_list, desc=f"{mname.upper()}", unit="seed"):
                self._set_seed(s)

                # ensure model has random_state where appropriate
                mparams_local = dict(mparams)
                if "random_state" in mparams_local or True:
                    mparams_local.setdefault("random_state", s)

                model = self.model_factory(mname, self.problem_type, mparams_local, n_outputs)

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
                if self.problem_type == "classification" and hasattr(
                    model, "predict_proba"
                ):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception as e:
                        print(f"    Warning: predict_proba failed: {e}")
                        y_proba = None

                # save model for this run in model-specific subfolder
                model_dir = os.path.join(self.output_dir, mname)
                os.makedirs(model_dir, exist_ok=True)
                fname = os.path.join(model_dir, f"model_{mname}_seed{s}.joblib")
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
    
    def run_cv(self, cfg: Dict[str, Any], fold_data_list):
        """Run k-fold cross-validation training.
        
        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary
        fold_data_list : list of tuples
            List of (X_train, X_val, X_test, y_train, y_val, y_test, fold_id) tuples
        
        Returns
        -------
        dict
            Results organised as {model_name: [run_results_with_fold_info]}
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
            
            model_results = []
            
            # Iterate over each fold
            for fold_idx, (X_train, X_val, X_test, y_train, y_val, y_test, fold_id) in enumerate(fold_data_list):
                # Determine number of outputs
                if len(y_train.shape) == 1:
                    n_outputs = 1
                else:
                    n_outputs = y_train.shape[1]
                
                # Run each seed for this fold
                for s in tqdm(seeds_list, desc=f"{mname.upper()} - Fold {fold_id+1}/{len(fold_data_list)}", unit="seed"):
                    self._set_seed(s)
                    
                    # ensure model has random_state where appropriate
                    mparams_local = dict(mparams)
                    if "random_state" in mparams_local or True:
                        mparams_local.setdefault("random_state", s)
                    
                    model = self.model_factory(mname, self.problem_type, mparams_local, n_outputs)
                    
                    # fit
                    if (
                        X_val is not None
                        and hasattr(model, "partial_fit")
                        and cfg.get("training", {}).get("use_partial_fit", False)
                    ):
                        model.partial_fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    y_proba = None
                    
                    # Only try to get probabilities for classification problems
                    if self.problem_type == "classification" and hasattr(model, "predict_proba"):
                        try:
                            y_proba = model.predict_proba(X_test)
                        except Exception as e:
                            # predict_proba may fail for some classifiers (e.g., SVC without probability=True)
                            # or when the model has issues with the data. This is not critical for training,
                            # but we silently continue without probabilities for this fold.
                            y_proba = None
                    
                    # save model for this run in model-specific subfolder
                    model_dir = os.path.join(self.output_dir, mname, f"fold{fold_id}")
                    os.makedirs(model_dir, exist_ok=True)
                    fname = os.path.join(model_dir, f"model_{mname}_fold{fold_id}_seed{s}.joblib")
                    joblib.dump(model, fname)
                    
                    model_results.append(
                        {
                            "seed": s,
                            "fold": fold_id,
                            "model_name": mname,
                            "model_path": fname,
                            "y_pred": y_pred,
                            "y_proba": y_proba,
                            "y_test": y_test,  # Include y_test for fold-specific evaluation
                            "X_test": X_test,  # Include X_test for permutation importance
                        }
                    )
            
            all_results[mname] = model_results
        
        return all_results
    
    def run_with_tuning(
        self,
        cfg: Dict[str, Any],
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        group_assignments: Optional[Dict] = None,
        groups_train: Optional[np.ndarray] = None,
        groups_val: Optional[np.ndarray] = None,
    ):
        """Run training with hyperparameter tuning.
        
        This method performs hyperparameter optimisation on the training + validation
        sets, then evaluates the best model on the held-out test set. Multiple seeds
        are used for stable results.
        
        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary with tuning settings
        X_train : np.ndarray
            Training features
        X_val : np.ndarray
            Validation features (used during hyperparameter tuning)
        X_test : np.ndarray
            Test features (held out for final evaluation)
        y_train : np.ndarray
            Training labels
        y_val : np.ndarray
            Validation labels
        y_test : np.ndarray
            Test labels
        group_assignments : Dict, optional
            Dictionary mapping split names to group IDs (for documentation)
        groups_train : np.ndarray, optional
            Group IDs for training samples
        groups_val : np.ndarray, optional
            Group IDs for validation samples
        
        Returns
        -------
        dict
            Results organised as {model_name: [run_results_with_tuning_info]}
        """
        from ecosci.hyperopt import HyperparameterTuner
        
        models_cfg = cfg.get("models", [])
        tuning_cfg = cfg.get("tuning", {})
        
        # Get seeds for stable results
        seeds = cfg.get("training", {}).get("seeds")
        repetitions = cfg.get("training", {}).get("repetitions", 1)
        base_seed = cfg.get("training", {}).get("random_seed", 0)
        
        if seeds:
            seeds_list = seeds
        else:
            seeds_list = [base_seed + i for i in range(repetitions)]
        
        # Tuning configuration
        search_method = tuning_cfg.get("search_method", "random")
        n_iter = tuning_cfg.get("n_iter", 50)
        cv_folds = tuning_cfg.get("cv_folds", 3)
        scoring = tuning_cfg.get("scoring")
        n_jobs = tuning_cfg.get("n_jobs", -1)
        verbose = tuning_cfg.get("verbose", 1)
        
        all_results = {}
        
        # Train each model type
        for model_idx, model_cfg in enumerate(models_cfg):
            mname = model_cfg.get("name", "mlp")
            
            # Get hyperparameter search space from config
            param_space = model_cfg.get("param_space")
            
            model_results = []
            
            print(f"\n{'='*60}")
            print(f"Tuning {mname.upper()}")
            print(f"{'='*60}")
            print(f"Metric: {scoring} ({'lower is better' if 'neg_' in str(scoring) else 'higher is better'})")
            print(f"Search method: {search_method}")
            print(f"CV folds: {cv_folds}")
            if search_method == "random":
                print(f"Iterations: {n_iter}")
            
            # Combine train and val for tuning (GridSearchCV will split internally)
            X_train_val = np.vstack([X_train, X_val])
            y_train_val = np.concatenate([y_train, y_val])
            
            # Combine groups for grouped CV during tuning (if groups are provided)
            groups_train_val = None
            if groups_train is not None and groups_val is not None:
                groups_train_val = np.concatenate([groups_train, groups_val])
            
            # Track best params across seeds
            best_params_per_seed = []
            
            for seed_idx, s in enumerate(tqdm(seeds_list, desc=f"Seeds", unit="seed")):
                self._set_seed(s)
                
                # Create tuner for this seed
                tuner = HyperparameterTuner(
                    problem_type=self.problem_type,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring=scoring,
                    search_method=search_method,
                    random_state=s,
                    n_jobs=n_jobs,
                    verbose=verbose if seed_idx == 0 else 0,  # Only verbose for first seed
                )
                
                # Tune hyperparameters
                print(f"\n  Seed {s}: Tuning hyperparameters...")
                best_model, tuning_results = tuner.tune_model(
                    mname, X_train_val, y_train_val, param_space, groups=groups_train_val
                )
                
                best_params = tuning_results['best_params']
                best_score = tuning_results['best_score']
                best_params_per_seed.append(best_params)
                
                # Make the score more interpretable
                if self.problem_type == "regression":
                    # For regression, the score is typically neg_mean_squared_error
                    if best_score < 0:
                        mse = -best_score
                        rmse = np.sqrt(mse)
                        print(f"  Seed {s}: Best CV score: MSE={mse:.4f}, RMSE={rmse:.4f} (lower is better)")
                    else:
                        # If positive (e.g., R²), show as-is
                        print(f"  Seed {s}: Best CV score: {best_score:.4f}")
                else:
                    # For classification, usually accuracy or similar (higher is better)
                    print(f"  Seed {s}: Best CV score: {best_score:.4f}")
                print(f"  Seed {s}: Best params: {best_params}")
                
                # Retrain on train set only (optional, but good practice)
                # The best_model is already trained on train_val, but we can
                # retrain on just train to match the typical workflow
                from ecosci.models import ModelZoo
                
                # Determine number of outputs
                if len(y_train.shape) == 1:
                    n_outputs = 1
                else:
                    n_outputs = y_train.shape[1]
                
                # Create new model with best params and train on train set
                final_model = ModelZoo.get_model(
                    mname,
                    self.problem_type,
                    best_params,
                    n_outputs
                )
                final_model.fit(X_train, y_train)
                
                # Evaluate on validation and test sets
                y_val_pred = final_model.predict(X_val)
                y_test_pred = final_model.predict(X_test)
                
                y_val_proba = None
                y_test_proba = None
                if self.problem_type == "classification" and hasattr(final_model, "predict_proba"):
                    try:
                        y_val_proba = final_model.predict_proba(X_val)
                        y_test_proba = final_model.predict_proba(X_test)
                    except Exception as e:
                        # predict_proba may fail for some classifiers (e.g., SVC without probability=True)
                        # or when the model has issues with the data. This is not critical for evaluation,
                        # but we silently continue without probabilities. Metrics like ROC-AUC won't be available.
                        y_val_proba = None
                        y_test_proba = None
                
                # Save model
                model_dir = os.path.join(self.output_dir, mname)
                os.makedirs(model_dir, exist_ok=True)
                model_fname = os.path.join(model_dir, f"model_{mname}_seed{s}.joblib")
                joblib.dump(final_model, model_fname)
                
                # Save tuning results
                tuning_fname = os.path.join(model_dir, f"tuning_results_{mname}_seed{s}.json")
                tuning_summary = {
                    'best_params': best_params,
                    'best_cv_score': float(best_score),
                    'seed': s,
                    'cv_folds': cv_folds,
                    'search_method': search_method,
                }
                with open(tuning_fname, 'w') as f:
                    json.dump(tuning_summary, f, indent=2)
                
                model_results.append({
                    'seed': s,
                    'model_name': mname,
                    'model_path': model_fname,
                    'best_params': best_params,
                    'best_cv_score': best_score,
                    'y_val_pred': y_val_pred,
                    'y_val_proba': y_val_proba,
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba,
                    'tuning_results_path': tuning_fname,
                })
            
            # Save summary of best params across all seeds
            summary_path = os.path.join(model_dir, f"tuning_summary_{mname}.json")
            summary = {
                'model_name': mname,
                'n_seeds': len(seeds_list),
                'seeds': seeds_list,
                'best_params_per_seed': best_params_per_seed,
                'search_method': search_method,
                'cv_folds': cv_folds,
                'group_assignments': group_assignments,
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n  Saved tuning summary to: {summary_path}")
            
            all_results[mname] = model_results
        
        return all_results
