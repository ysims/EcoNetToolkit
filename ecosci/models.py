"""Model zoo: quick access to common models with a unified interface.

Supported model names (set in YAML under `models[].name`):
- mlp          : shallow neural network (1 hidden layer by default)
- random_forest: tree ensemble, good baseline
- svm          : support vector machine (classification or regression)
- xgboost      : gradient boosting trees (requires `xgboost` package)
- logistic     : logistic regression (classification baseline)
- linear       : linear regression (regression only)

All model hyperparameters come from `models[].params` in the YAML and are passed
through to the underlying scikit-learn/xgboost classes. This keeps the code
short and the config flexible.
"""

from typing import Any, Dict


class ModelZoo:
    @staticmethod
    def get_model(
        name: str, 
        problem_type: str = "classification", 
        params: Dict[str, Any] = None,
        n_outputs: int = 1
    ):
        """Return a model instance by name using params dict.

        Parameters
        ----------
        name : str
            Model name (see list above).
        problem_type : str
            "classification" or "regression". Determines which estimator is used.
        params : Dict[str, Any]
            Hyperparameters forwarded to the underlying estimator.
        n_outputs : int
            Number of output targets. If > 1, wraps the model in MultiOutput wrapper.
        """
        params = params or {}

        if name.lower() == "mlp":
            from sklearn.neural_network import MLPClassifier, MLPRegressor

            if problem_type == "classification":
                # shallow MLP: single hidden layer default
                base_model = MLPClassifier(
                    hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (32,))),
                    max_iter=params.get("max_iter", 200),
                    early_stopping=params.get("early_stopping", True),
                    validation_fraction=params.get("validation_fraction", 0.1),
                    n_iter_no_change=params.get("n_iter_no_change", 10),
                    random_state=params.get("random_state", 0),
                    verbose=params.get("verbose", False),  # Disable verbose output by default
                    **{
                        k: v
                        for k, v in params.items()
                        if k
                        not in [
                            "hidden_layer_sizes",
                            "max_iter",
                            "early_stopping",
                            "validation_fraction",
                            "n_iter_no_change",
                            "random_state",
                            "verbose",
                        ]
                    },
                )
                return ModelZoo.wrap_for_multioutput(base_model, problem_type, n_outputs)
            else:
                # MLPRegressor natively supports multi-output
                return MLPRegressor(
                    hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (32,))),
                    max_iter=params.get("max_iter", 200),
                    early_stopping=params.get("early_stopping", True),
                    validation_fraction=params.get("validation_fraction", 0.1),
                    n_iter_no_change=params.get("n_iter_no_change", 10),
                    random_state=params.get("random_state", 0),
                    verbose=params.get("verbose", False),  # Disable verbose output by default
                    **{
                        k: v
                        for k, v in params.items()
                        if k
                        not in [
                            "hidden_layer_sizes",
                            "max_iter",
                            "early_stopping",
                            "validation_fraction",
                            "n_iter_no_change",
                            "random_state",
                            "verbose",
                        ]
                    },
                )

        if name.lower() == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Random Forest natively supports multi-output
            if problem_type == "classification":
                return RandomForestClassifier(
                    random_state=params.get("random_state", 0),
                    **{k: v for k, v in params.items() if k != "random_state"},
                )
            else:
                return RandomForestRegressor(
                    random_state=params.get("random_state", 0),
                    **{k: v for k, v in params.items() if k != "random_state"},
                )

        if name.lower() == "svm":
            from sklearn.svm import SVC, SVR

            if problem_type == "classification":
                base_model = SVC(
                    probability=True,
                    random_state=params.get("random_state", 0),
                    **{k: v for k, v in params.items() if k != "random_state"},
                )
                return ModelZoo.wrap_for_multioutput(base_model, problem_type, n_outputs)
            else:
                # SVR doesn't support random_state parameter
                base_model = SVR(**{k: v for k, v in params.items() if k != "random_state"})
                return ModelZoo.wrap_for_multioutput(base_model, problem_type, n_outputs)

        if name.lower() == "xgboost":
            try:
                from xgboost import XGBClassifier, XGBRegressor
            except Exception as e:
                raise ImportError(
                    "xgboost is required for the xgboost model: install via pip install xgboost"
                )

            if problem_type == "classification":
                # use_label_encoder is deprecated and no longer needed
                # Labels are already encoded by the data loader
                eval_metric = params.get("eval_metric", "logloss")
                params_filtered = {
                    k: v for k, v in params.items() if k != "eval_metric"
                }
                base_model = XGBClassifier(eval_metric=eval_metric, **params_filtered)
                return ModelZoo.wrap_for_multioutput(base_model, problem_type, n_outputs)
            else:
                base_model = XGBRegressor(**params)
                return ModelZoo.wrap_for_multioutput(base_model, problem_type, n_outputs)

        if name.lower() == "logistic":
            from sklearn.linear_model import LogisticRegression

            base_model = LogisticRegression(
                random_state=params.get("random_state", 0),
                max_iter=params.get("max_iter", 1000),
                **{
                    k: v
                    for k, v in params.items()
                    if k not in ["random_state", "max_iter"]
                },
            )
            return ModelZoo.wrap_for_multioutput(base_model, problem_type, n_outputs)

        if name.lower() == "linear":
            from sklearn.linear_model import LinearRegression, Ridge

            if problem_type == "classification":
                raise ValueError(
                    "Linear model is only for regression. Use 'logistic' for classification."
                )

            # If alpha is specified, use Ridge regression (supports regularisation)
            # Otherwise use plain LinearRegression
            if "alpha" in params:
                base_model = Ridge(
                    random_state=params.get("random_state", 0),
                    **{k: v for k, v in params.items() if k != "random_state"}
                )
            else:
                # LinearRegression doesn't support random_state or alpha
                base_model = LinearRegression(
                    **{k: v for k, v in params.items() if k not in ["random_state", "alpha"]}
                )
            
            # Wrap in MultiOutputRegressor if needed
            if n_outputs > 1:
                from sklearn.multioutput import MultiOutputRegressor
                return MultiOutputRegressor(base_model)
            return base_model

        raise ValueError(f"Unknown model name: {name}")
    
    @staticmethod
    def wrap_for_multioutput(model, problem_type: str, n_outputs: int):
        """Wrap a model for multi-output prediction if necessary.
        
        Parameters
        ----------
        model : estimator
            Base sklearn-compatible model.
        problem_type : str
            "classification" or "regression".
        n_outputs : int
            Number of output targets.
            
        Returns
        -------
        estimator
            The model, wrapped in MultiOutput if n_outputs > 1.
        """
        if n_outputs <= 1:
            return model
            
        # Some models natively support multi-output
        native_multioutput = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor',
            'KNeighborsClassifier', 'KNeighborsRegressor',
            'MLPRegressor',  # MLP Regressor supports multi-output
        ]
        
        model_class = model.__class__.__name__
        if model_class in native_multioutput:
            return model
            
        # Wrap in MultiOutput wrapper
        if problem_type == "classification":
            from sklearn.multioutput import MultiOutputClassifier
            return MultiOutputClassifier(model)
        else:
            from sklearn.multioutput import MultiOutputRegressor
            return MultiOutputRegressor(model)
