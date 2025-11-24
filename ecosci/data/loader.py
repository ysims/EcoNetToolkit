"""CSV data loader with preprocessing pipeline."""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from .preprocessing import (
    get_feature_names_after_transform,
    build_preprocessing_pipeline,
    encode_labels,
)
from .splitting import prepare_cv_folds, prepare_grouped_splits


class CSVDataLoader:
    """Load a CSV and build a simple preprocessing pipeline.

    How to use (via YAML): specify the CSV path, list the feature columns and the
    label column, and choose options for imputation and scaling.

    Notes
    -----
    - Scaling: "standard" makes numbers with mean=0 and std=1 (good default).
        "minmax" squashes numbers to 0..1.
    - Categorical text columns are one-hot encoded automatically.
    - Missing numbers use the mean (by default). Categorical uses the most
        frequent value.
    
    Parameters
    ----------
    path : str
        Path to CSV file
    features : list of str, optional
        Column names to use as features. If None, all columns except labels are used.
    label : str, optional
        Single target column name (for backward compatibility)
    labels : list of str, optional
        Multiple target column names (for multi-output)
    test_size : float
        Fraction of data for test set
    val_size : float
        Fraction of data for validation set
    random_state : int
        Random seed for reproducibility
    scaling : str
        "standard" or "minmax" scaling
    impute_strategy : str
        Imputation strategy: "mean", "median", or "most_frequent"
    problem_type : str
        "classification" or "regression"
    cv_group_column : str, optional
        Column name for grouping in cross-validation
    """

    def __init__(
        self,
        path: str,
        features: Optional[List[str]] = None,
        label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 0,
        scaling: str = "standard",
        impute_strategy: str = "mean",
        problem_type: str = "classification",
        cv_group_column: Optional[str] = None,
    ):
        self.path = path
        self.features = features
        # Support both single label and multiple labels
        if labels is not None:
            self.labels = labels if isinstance(labels, list) else [labels]
            self.label = None
        elif label is not None:
            self.labels = [label]
            self.label = label
        else:
            self.labels = None
            self.label = None
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaling = scaling
        self.impute_strategy = impute_strategy
        self.problem_type = problem_type
        self.cv_group_column = cv_group_column

    def load(self) -> pd.DataFrame:
        """Read the CSV into a DataFrame."""
        df = pd.read_csv(self.path)
        return df
    
    def _get_feature_names_after_transform(self, preprocessor, numeric_cols, cat_cols):
        """Get feature names after preprocessing (including one-hot encoding)."""
        return get_feature_names_after_transform(preprocessor, numeric_cols, cat_cols)

    def prepare(self) -> Tuple:
        """Prepare arrays for modeling.

        Returns
        -------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)

        Under the hood we use scikit-learn pipelines for imputation, encoding
        and scaling to keep things robust and simple.
        """
        from sklearn.model_selection import train_test_split

        df = self.load()

        if self.labels is None:
            raise ValueError("Label column(s) must be provided in config")

        if self.features is None:
            # default: all columns except labels
            self.features = [c for c in df.columns if c not in self.labels]

        X = df[self.features].copy()
        
        # When single label, DataFrame.iloc returns Series; when multiple, returns DataFrame
        if len(self.labels) == 1:
            y = df[self.labels[0]].copy()  # Get as Series
            mask_target = ~y.isna()
        else:
            y = df[self.labels].copy()  # Get as DataFrame
            mask_target = ~y.isna().any(axis=1)
        
        if mask_target.any() and not mask_target.all():
            n_dropped_target = (~mask_target).sum()
            print(f"Warning: Dropping {n_dropped_target} rows with NaN in target {self.labels}")
            X = X[mask_target]
            y = y[mask_target]

        # Drop rows with NaN in any features
        mask_features = ~X.isna().any(axis=1)
        if not mask_features.all():
            n_dropped_features = (~mask_features).sum()
            cols_with_nan = X.columns[X.isna().any()].tolist()
            print(f"Warning: Dropping {n_dropped_features} additional rows with NaN in features: {cols_with_nan}")
            X = X[mask_features]
            y = y[mask_features]

        # Reset indices after dropping rows
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Encode labels if they are strings/categorical
        y, self.label_encoders, self.label_classes_dict, self.label_encoder, self.label_classes = encode_labels(
            y, self.labels, self.problem_type
        )

        # Identify numeric vs categorical (strings, categories, etc.)
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        # Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(
            numeric_cols, cat_cols, self.scaling, self.impute_strategy
        )

        X_proc = preprocessor.fit_transform(X)
        
        # Store preprocessor and generate feature names after transformation
        self.preprocessor = preprocessor
        self.processed_feature_names = self._get_feature_names_after_transform(
            preprocessor, numeric_cols, cat_cols
        )

        # Check number of unique classes for stratification
        if self.problem_type == "classification" and len(self.labels) == 1:
            n_unique = len(np.unique(y))
            stratify_y = y if n_unique <= 20 else None
        else:
            # Cannot stratify on multi-output or regression
            stratify_y = None

        # Split: first test, then validation from remaining
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_proc,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        # compute val fraction relative to train_val
        if self.val_size <= 0:
            X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None
        else:
            val_fraction = self.val_size / (1.0 - self.test_size)
            # Only stratify for classification problems with reasonable number of classes
            if self.problem_type == "classification" and len(self.labels) == 1:
                n_unique_train = len(np.unique(y_train_val))
                stratify_train = y_train_val if n_unique_train <= 20 else None
            else:
                stratify_train = None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_fraction,
                random_state=self.random_state,
                stratify=stratify_train,
            )

        return (
            X_train,
            X_val,
            X_test,
            np.array(y_train),
            (None if y_val is None else np.array(y_val)),
            np.array(y_test),
        )
    
    def prepare_cv_folds(self):
        """Prepare k-fold cross-validation splits using group-based splitting.
        
        Returns
        -------
        list of tuples
            Each tuple contains (X_train, X_val, X_test, y_train, y_val, y_test, fold_id)
        
        Notes
        -----
        - Uses GroupKFold to keep all samples from the same group together
        - Number of folds = number of unique groups
        """
        if self.cv_group_column is None:
            raise ValueError("cv_group_column must be specified for k-fold cross-validation")
        
        df = self.load()
        
        if self.labels is None:
            raise ValueError("Label column(s) must be provided in config")
        
        if self.features is None:
            self.features = [c for c in df.columns if c not in self.labels and c != self.cv_group_column]
        
        # Get group IDs
        if self.cv_group_column not in df.columns:
            raise ValueError(f"Group column '{self.cv_group_column}' not found in data")
        
        groups = df[self.cv_group_column].copy()
        X = df[self.features].copy()
        
        # Handle single vs multi-output
        if len(self.labels) == 1:
            y = df[self.labels[0]].copy()
            mask_target = ~y.isna()
        else:
            y = df[self.labels].copy()
            mask_target = ~y.isna().any(axis=1)
        
        # Drop NaN rows
        if mask_target.any() and not mask_target.all():
            n_dropped_target = (~mask_target).sum()
            print(f"Warning: Dropping {n_dropped_target} rows with NaN in target {self.labels}")
            X, y, groups = X[mask_target], y[mask_target], groups[mask_target]
        
        mask_features = ~X.isna().any(axis=1)
        if not mask_features.all():
            n_dropped_features = (~mask_features).sum()
            cols_with_nan = X.columns[X.isna().any()].tolist()
            print(f"Warning: Dropping {n_dropped_features} additional rows with NaN in features: {cols_with_nan}")
            X, y, groups = X[mask_features], y[mask_features], groups[mask_features]
        
        # Reset indices
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        groups = groups.reset_index(drop=True)
        
        # Encode labels
        y, self.label_encoders, self.label_classes_dict, self.label_encoder, self.label_classes = encode_labels(
            y, self.labels, self.problem_type
        )
        
        # Identify numeric vs categorical
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]
        
        # Store feature names from a sample fit
        sample_preprocessor = build_preprocessing_pipeline(
            numeric_cols, cat_cols, self.scaling, self.impute_strategy
        )
        sample_preprocessor.fit(X)
        self.processed_feature_names = self._get_feature_names_after_transform(
            sample_preprocessor, numeric_cols, cat_cols
        )
        
        # Prepare folds
        return prepare_cv_folds(
            X, y, groups, self.labels, numeric_cols, cat_cols,
            self.val_size, self.random_state, self.scaling, self.impute_strategy,
            self.problem_type, self.cv_group_column
        )
    
    def prepare_grouped_splits(
        self,
        n_train_groups: int = 4,
        n_val_groups: int = 2,
        n_test_groups: int = 2,
    ):
        """Prepare train/val/test splits by assigning groups to each split.
        
        This method ensures no data leakage by keeping all samples from the same
        group together in one split.
        
        Parameters
        ----------
        n_train_groups : int
            Number of groups to assign to training set
        n_val_groups : int
            Number of groups to assign to validation set
        n_test_groups : int
            Number of groups to assign to test set
        
        Returns
        -------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments,
             groups_train, groups_val, groups_test)
        """
        if self.cv_group_column is None:
            raise ValueError("cv_group_column must be specified for grouped train/val/test splits")
        
        df = self.load()
        
        if self.labels is None:
            raise ValueError("Label column(s) must be provided in config")
        
        if self.features is None:
            self.features = [c for c in df.columns if c not in self.labels and c != self.cv_group_column]
        
        if self.cv_group_column not in df.columns:
            raise ValueError(f"Group column '{self.cv_group_column}' not found in data")
        
        groups = df[self.cv_group_column].copy()
        X = df[self.features].copy()
        
        # Handle single vs multi-output
        if len(self.labels) == 1:
            y = df[self.labels[0]].copy()
            mask_target = ~y.isna()
        else:
            y = df[self.labels].copy()
            mask_target = ~y.isna().any(axis=1)
        
        # Drop NaN rows
        if mask_target.any() and not mask_target.all():
            n_dropped_target = (~mask_target).sum()
            print(f"Warning: Dropping {n_dropped_target} rows with NaN in target {self.labels}")
            X, y, groups = X[mask_target], y[mask_target], groups[mask_target]
        
        mask_features = ~X.isna().any(axis=1)
        if not mask_features.all():
            n_dropped_features = (~mask_features).sum()
            cols_with_nan = X.columns[X.isna().any()].tolist()
            print(f"Warning: Dropping {n_dropped_features} additional rows with NaN in features: {cols_with_nan}")
            X, y, groups = X[mask_features], y[mask_features], groups[mask_features]
        
        # Reset indices
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        groups = groups.reset_index(drop=True)
        
        # Encode labels
        y, self.label_encoders, self.label_classes_dict, self.label_encoder, self.label_classes = encode_labels(
            y, self.labels, self.problem_type
        )
        
        # Identify numeric vs categorical
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]
        
        # Prepare grouped splits
        result = prepare_grouped_splits(
            X, y, groups, self.labels, numeric_cols, cat_cols,
            self.random_state, self.scaling, self.impute_strategy,
            n_train_groups, n_val_groups, n_test_groups, self.cv_group_column
        )
        
        # Unpack result
        (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments,
         groups_train, groups_val, groups_test, preprocessor) = result
        
        # Store preprocessor and feature names
        self.preprocessor = preprocessor
        self.processed_feature_names = self._get_feature_names_after_transform(
            preprocessor, numeric_cols, cat_cols
        )
        
        return (
            X_train, X_val, X_test, y_train, y_val, y_test, group_assignments,
            groups_train, groups_val, groups_test
        )
