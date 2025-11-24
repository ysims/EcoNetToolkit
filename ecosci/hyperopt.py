"""Hyperparameter optimisation for different model types.

This module provides hyperparameter tuning functionality using scikit-learn's
search methods (GridSearchCV, RandomizedSearchCV). Each model type has a dedicated
tuning function with sensible default search spaces that can be overridden via config.

Key features:
- Separate tuning functions for each model type (MLP, RandomForest, XGBoost, SVM, Linear)
- Uses validation set for tuning, test set held out for final evaluation
- Supports both exhaustive grid search and randomized search
- Returns best model and tuning results
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
import joblib
import os


class HyperparameterTuner:
    """Hyperparameter tuning for various model types.
    
    Parameters
    ----------
    problem_type : str
        "classification" or "regression"
    n_iter : int, optional
        Number of iterations for RandomizedSearchCV (default: 50)
    cv : int, optional
        Number of cross-validation folds to use during tuning (default: 3)
    scoring : str, optional
        Scoring metric for tuning (default: auto-selected based on problem_type)
    search_method : str, optional
        Either "grid" for GridSearchCV or "random" for RandomizedSearchCV (default: "random")
    random_state : int, optional
        Random state for reproducibility (default: 42)
    n_jobs : int, optional
        Number of parallel jobs (default: -1, use all cores)
    verbose : int, optional
        Verbosity level (default: 1)
    """
    
    def __init__(
        self,
        problem_type: str = "regression",
        n_iter: int = 50,
        cv: int = 3,
        scoring: Optional[str] = None,
        search_method: str = "random",
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1,
    ):
        self.problem_type = problem_type
        self.n_iter = n_iter
        self.cv = cv
        self.search_method = search_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Auto-select scoring metric if not provided
        if scoring is None:
            if problem_type == "regression":
                self.scoring = "neg_mean_squared_error"
            else:
                self.scoring = "accuracy"
        else:
            self.scoring = scoring
    
    def _get_search_estimator(self, model: BaseEstimator, param_space: Dict, groups: Optional[np.ndarray] = None):
        """Create appropriate search estimator (Grid or Randomized).
        
        Parameters
        ----------
        model : BaseEstimator
            Base model to tune
        param_space : Dict
            Parameter search space
        groups : np.ndarray, optional
            Group labels for GroupKFold. If provided, uses GroupKFold instead of regular KFold.
        
        Returns
        -------
        search estimator
            GridSearchCV or RandomizedSearchCV configured appropriately
        """
        from sklearn.model_selection import GroupKFold
        
        # Determine CV strategy
        if groups is not None:
            # Use GroupKFold to respect group structure
            cv_strategy = GroupKFold(n_splits=self.cv)
        else:
            # Use default KFold (integer cv value)
            cv_strategy = self.cv
        
        if self.search_method == "grid":
            return GridSearchCV(
                model,
                param_space,
                cv=cv_strategy,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True,
            )
        else:
            return RandomizedSearchCV(
                model,
                param_space,
                n_iter=self.n_iter,
                cv=cv_strategy,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True,
                random_state=self.random_state,
            )
    
    def tune_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[BaseEstimator, Dict]:
        """Tune MLP hyperparameters.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_space : Dict, optional
            Custom parameter search space. If None, uses sensible defaults.
        groups : np.ndarray, optional
            Group labels for GroupKFold cross-validation
        
        Returns
        -------
        best_model : BaseEstimator
            Trained model with best hyperparameters
        results : Dict
            Dictionary containing best parameters, best score, and CV results
        """
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        
        # Default parameter space
        if param_space is None:
            param_space = {
                'hidden_layer_sizes': [
                    (16,), (32,), (64,),
                    (16, 8), (32, 16), (64, 32),
                    (32, 16, 8), (64, 32, 16),
                ],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate_init': [0.0001, 0.001, 0.01],
                'batch_size': [8, 16, 32, 64],
                'early_stopping': [True],
                'validation_fraction': [0.1],
                'n_iter_no_change': [10, 20],
                'max_iter': [1000],
            }
        
        # Create base model
        if self.problem_type == "regression":
            base_model = MLPRegressor(random_state=self.random_state, verbose=False)
        else:
            base_model = MLPClassifier(random_state=self.random_state, verbose=False)
        
        # Perform search
        search = self._get_search_estimator(base_model, param_space, groups=groups)
        search.fit(X_train, y_train, groups=groups)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
        }
        
        return search.best_estimator_, results
    
    def tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[BaseEstimator, Dict]:
        """Tune Random Forest hyperparameters.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_space : Dict, optional
            Custom parameter search space. If None, uses sensible defaults.
        groups : np.ndarray, optional
            Group labels for GroupKFold cross-validation
        
        Returns
        -------
        best_model : BaseEstimator
            Trained model with best hyperparameters
        results : Dict
            Dictionary containing best parameters, best score, and CV results
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Default parameter space
        if param_space is None:
            param_space = {
                'n_estimators': [100, 200, 500, 1000],
                'max_depth': [5, 10, 20, 30, 50, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.5],
                'bootstrap': [True, False],
            }
        
        # Create base model
        if self.problem_type == "regression":
            base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=1)
        else:
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=1)
        
        # Perform search
        search = self._get_search_estimator(base_model, param_space, groups=groups)
        search.fit(X_train, y_train, groups=groups)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
        }
        
        return search.best_estimator_, results
    
    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[BaseEstimator, Dict]:
        """Tune XGBoost hyperparameters.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_space : Dict, optional
            Custom parameter search space. If None, uses sensible defaults.
        groups : np.ndarray, optional
            Group labels for GroupKFold cross-validation
        
        Returns
        -------
        best_model : BaseEstimator
            Trained model with best hyperparameters
        results : Dict
            Dictionary containing best parameters, best score, and CV results
        """
        try:
            from xgboost import XGBRegressor, XGBClassifier
        except ImportError:
            raise ImportError("xgboost is required: install via pip install xgboost")
        
        # Default parameter space
        if param_space is None:
            param_space = {
                'n_estimators': [100, 200, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0, 0.01, 0.1, 1.0],
                'reg_lambda': [0.1, 1.0, 10.0],
            }
        
        # Create base model
        if self.problem_type == "regression":
            base_model = XGBRegressor(random_state=self.random_state, n_jobs=1)
        else:
            base_model = XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=1
            )
        
        # Perform search
        search = self._get_search_estimator(base_model, param_space, groups=groups)
        search.fit(X_train, y_train, groups=groups)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
        }
        
        return search.best_estimator_, results
    
    def tune_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[BaseEstimator, Dict]:
        """Tune SVM hyperparameters.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_space : Dict, optional
            Custom parameter search space. If None, uses sensible defaults.
        groups : np.ndarray, optional
            Group labels for GroupKFold cross-validation
        
        Returns
        -------
        best_model : BaseEstimator
            Trained model with best hyperparameters
        results : Dict
            Dictionary containing best parameters, best score, and CV results
        """
        from sklearn.svm import SVR, SVC
        
        # Default parameter space
        if param_space is None:
            if self.problem_type == "regression":
                param_space = {
                    'kernel': ['rbf', 'poly', 'linear'],
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.5, 1.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                }
            else:
                param_space = {
                    'kernel': ['rbf', 'poly', 'linear'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                }
        
        # Create base model
        if self.problem_type == "regression":
            base_model = SVR()
        else:
            base_model = SVC(probability=True, random_state=self.random_state)
        
        # Perform search
        search = self._get_search_estimator(base_model, param_space, groups=groups)
        search.fit(X_train, y_train, groups=groups)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
        }
        
        return search.best_estimator_, results
    
    def tune_linear(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[BaseEstimator, Dict]:
        """Tune Linear model hyperparameters.
        
        For linear regression, there are few hyperparameters to tune,
        but we can tune regularization if using Ridge/Lasso.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_space : Dict, optional
            Custom parameter search space. If None, uses Ridge regression with alpha tuning.
        groups : np.ndarray, optional
            Group labels for GroupKFold cross-validation
        
        Returns
        -------
        best_model : BaseEstimator
            Trained model with best hyperparameters
        results : Dict
            Dictionary containing best parameters, best score, and CV results
        """
        from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
        
        # Default parameter space - use Ridge for regression, LogisticRegression for classification
        if param_space is None:
            if self.problem_type == "regression":
                # Use Ridge with alpha tuning
                param_space = {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                }
                base_model = Ridge(random_state=self.random_state)
            else:
                # Use LogisticRegression with C and penalty tuning
                param_space = {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['saga'],  # saga supports all penalties
                    'l1_ratio': [0.1, 0.5, 0.9],  # for elasticnet
                }
                base_model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                )
        else:
            # Use provided param_space and determine model type from it
            if self.problem_type == "regression":
                base_model = Ridge(random_state=self.random_state)
            else:
                base_model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                )
        
        # Perform search
        search = self._get_search_estimator(base_model, param_space, groups=groups)
        search.fit(X_train, y_train, groups=groups)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
        }
        
        return search.best_estimator_, results
    
    def tune_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_space: Optional[Dict] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[BaseEstimator, Dict]:
        """Tune hyperparameters for specified model type.
        
        Parameters
        ----------
        model_name : str
            Name of model to tune ('mlp', 'random_forest', 'xgboost', 'svm', 'linear')
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_space : Dict, optional
            Custom parameter search space. If None, uses sensible defaults.
        groups : np.ndarray, optional
            Group labels for GroupKFold cross-validation
        
        Returns
        -------
        best_model : BaseEstimator
            Trained model with best hyperparameters
        results : Dict
            Dictionary containing best parameters, best score, and CV results
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower == "mlp":
            return self.tune_mlp(X_train, y_train, param_space, groups=groups)
        elif model_name_lower == "random_forest":
            return self.tune_random_forest(X_train, y_train, param_space, groups=groups)
        elif model_name_lower == "xgboost":
            return self.tune_xgboost(X_train, y_train, param_space, groups=groups)
        elif model_name_lower == "svm":
            return self.tune_svm(X_train, y_train, param_space, groups=groups)
        elif model_name_lower in ["linear", "logistic"]:
            return self.tune_linear(X_train, y_train, param_space, groups=groups)
        else:
            raise ValueError(f"Unknown model name for tuning: {model_name}")
    
    def save_tuning_results(self, results: Dict, output_path: str):
        """Save tuning results to disk.
        
        Parameters
        ----------
        results : Dict
            Results dictionary from tuning
        output_path : str
            Path to save results (as joblib file)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(results, output_path)
