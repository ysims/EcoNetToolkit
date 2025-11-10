"""Model zoo: quick access to common models with a unified interface.

Supported model names (set in YAML under `models[].name`):
- mlp          : shallow neural network (1 hidden layer by default)
- random_forest: tree ensemble, good baseline
- svm          : support vector machine (classification or regression)
- xgboost      : gradient boosting trees (requires `xgboost` package)
- logistic     : logistic regression (classification baseline)

All model hyperparameters come from `models[].params` in the YAML and are passed
through to the underlying scikit-learn/xgboost classes. This keeps the code
short and the config flexible.
"""
from typing import Any, Dict


class ModelZoo:
    @staticmethod
    def get_model(name: str, problem_type: str = "classification", params: Dict[str, Any] = None):
        """Return a model instance by name using params dict.

        Parameters
        ----------
        name : str
            Model name (see list above).
        problem_type : str
            "classification" or "regression". Determines which estimator is used.
        params : Dict[str, Any]
            Hyperparameters forwarded to the underlying estimator.
        """
        params = params or {}

        if name.lower() == "mlp":
            from sklearn.neural_network import MLPClassifier, MLPRegressor

            if problem_type == "classification":
                # shallow MLP: single hidden layer default
                return MLPClassifier(hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (32,))),
                                     max_iter=params.get("max_iter", 200),
                                     early_stopping=params.get("early_stopping", True),
                                     validation_fraction=params.get("validation_fraction", 0.1),
                                     n_iter_no_change=params.get("n_iter_no_change", 10),
                                     random_state=params.get("random_state", 0),
                                     **{k: v for k, v in params.items() if k not in ["hidden_layer_sizes", "max_iter", "early_stopping", "validation_fraction", "n_iter_no_change", "random_state"]})
            else:
                return MLPRegressor(hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (32,))),
                                     max_iter=params.get("max_iter", 200),
                                     early_stopping=params.get("early_stopping", True),
                                     validation_fraction=params.get("validation_fraction", 0.1),
                                     n_iter_no_change=params.get("n_iter_no_change", 10),
                                     random_state=params.get("random_state", 0),
                                     **{k: v for k, v in params.items() if k not in ["hidden_layer_sizes", "max_iter", "early_stopping", "validation_fraction", "n_iter_no_change", "random_state"]})

        if name.lower() == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            if problem_type == "classification":
                return RandomForestClassifier(random_state=params.get("random_state", 0), 
                                             **{k: v for k, v in params.items() if k != "random_state"})
            else:
                return RandomForestRegressor(random_state=params.get("random_state", 0), 
                                            **{k: v for k, v in params.items() if k != "random_state"})

        if name.lower() == "svm":
            from sklearn.svm import SVC, SVR

            if problem_type == "classification":
                return SVC(probability=True, random_state=params.get("random_state", 0), 
                          **{k: v for k, v in params.items() if k != "random_state"})
            else:
                return SVR(**{k: v for k, v in params.items()})

        if name.lower() == "xgboost":
            try:
                from xgboost import XGBClassifier, XGBRegressor
            except Exception as e:
                raise ImportError("xgboost is required for the xgboost model: install via pip install xgboost")

            if problem_type == "classification":
                # use_label_encoder is deprecated and no longer needed
                # Labels are already encoded by the data loader
                eval_metric = params.get("eval_metric", "logloss")
                params_filtered = {k: v for k, v in params.items() if k != "eval_metric"}
                return XGBClassifier(eval_metric=eval_metric, **params_filtered)
            else:
                return XGBRegressor(**params)

        if name.lower() == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(random_state=params.get("random_state", 0), 
                                    max_iter=params.get("max_iter", 1000), 
                                    **{k: v for k, v in params.items() if k not in ["random_state", "max_iter"]})

        raise ValueError(f"Unknown model name: {name}")
