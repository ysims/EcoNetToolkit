"""Data loading and preprocessing utilities.

What this does for you (no coding required):
- Reads your CSV file
- Picks which columns are inputs (features) and which is the label (what you want to predict)
- Handles missing values (imputation)
- Scales numeric columns so models train better
- Encodes text/categorical columns into numbers automatically
- Splits your data into train/validation/test sets
"""

from typing import List, Optional, Tuple
import pandas as pd


class CSVDataLoader:
    """Load a CSV and build a simple preprocessing pipeline.

    How to use (via YAML): specify the CSV path, list the feature columns and the
    label column, and choose options for imputation and scaling.

    Notes
    - Scaling: "standard" makes numbers with mean=0 and std=1 (good default).
        "minmax" squashes numbers to 0..1.
    - Categorical text columns are one-hot encoded automatically.
    - Missing numbers use the mean (by default). Categorical uses the most
        frequent value.
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
        feature_names = []
        
        # Get feature names from each transformer
        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                # Numeric columns keep their names
                feature_names.extend(columns)
            elif name == "cat":
                # For categorical columns, get one-hot encoded names
                try:
                    # Get the OneHotEncoder from the pipeline
                    onehot = transformer.named_steps.get("onehot")
                    if onehot is not None and hasattr(onehot, "get_feature_names_out"):
                        # Get one-hot feature names
                        cat_features = onehot.get_feature_names_out(columns)
                        feature_names.extend(cat_features)
                    else:
                        # Fallback: just use column names
                        feature_names.extend(columns)
                except Exception:
                    # Fallback if something goes wrong
                    feature_names.extend(columns)
        
        return feature_names

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
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            StandardScaler,
            MinMaxScaler,
            OneHotEncoder,
            LabelEncoder,
        )
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import numpy as np

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
            print(
                f"Warning: Dropping {n_dropped_target} rows with NaN in target {self.labels}"
            )
            X = X[mask_target]
            y = y[mask_target]

        # Drop rows with NaN in any features
        mask_features = ~X.isna().any(axis=1)
        if not mask_features.all():
            n_dropped_features = (~mask_features).sum()
            cols_with_nan = X.columns[X.isna().any()].tolist()
            print(
                f"Warning: Dropping {n_dropped_features} additional rows with NaN in features: {cols_with_nan}"
            )
            X = X[mask_features]
            y = y[mask_features]

        # Reset indices after dropping rows
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Encode labels if they are strings/categorical (needed for XGBoost and others)
        # For multi-output, encode each output separately
        self.label_encoders = {}
        self.label_classes_dict = {}
        
        if len(self.labels) == 1:
            # Single output: maintain backward compatibility
            if y.dtype == "object" or y.dtype.name == "category":
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                self.label_encoders[self.labels[0]] = label_encoder
                self.label_classes_dict[self.labels[0]] = label_encoder.classes_
                # Keep backward compatibility
                self.label_encoder = label_encoder
                self.label_classes = label_encoder.classes_
            else:
                self.label_encoder = None
                self.label_classes = None
        else:
            # Multi-output: encode each column separately
            y_encoded = []
            for col in self.labels:
                if y[col].dtype == "object" or y[col].dtype.name == "category":
                    label_encoder = LabelEncoder()
                    y_col_encoded = label_encoder.fit_transform(y[col])
                    self.label_encoders[col] = label_encoder
                    self.label_classes_dict[col] = label_encoder.classes_
                else:
                    y_col_encoded = y[col].values
                    self.label_encoders[col] = None
                    self.label_classes_dict[col] = None
                y_encoded.append(y_col_encoded)
            y = np.column_stack(y_encoded)
            self.label_encoder = None
            self.label_classes = None

        # Identify numeric vs categorical (strings, categories, etc.)
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        transformers = []

        if numeric_cols:
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy=self.impute_strategy)),
                    (
                        "scaler",
                        (
                            StandardScaler()
                            if self.scaling == "standard"
                            else MinMaxScaler()
                        ),
                    ),
                ]
            )
            transformers.append(("num", num_pipeline, numeric_cols))

        if cat_cols:
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipeline, cat_cols))

        preprocessor = ColumnTransformer(transformers, remainder="drop")

        X_proc = preprocessor.fit_transform(X)
        
        # Store preprocessor and generate feature names after transformation
        self.preprocessor = preprocessor
        self.processed_feature_names = self._get_feature_names_after_transform(
            preprocessor, numeric_cols, cat_cols
        )

        # Check number of unique classes for stratification
        # Only stratify for classification problems with reasonable number of classes
        # For multi-output, we cannot use stratification directly
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
            where fold_id is the test fold number.
        
        Notes
        -----
        - Uses GroupKFold to keep all samples from the same group together in the same fold
        - The group is specified by cv_group_column (e.g., 'patch_id')
        - Each unique group value becomes a unit for splitting
        - Number of folds = number of unique groups
        """
        from sklearn.model_selection import GroupKFold
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            StandardScaler,
            MinMaxScaler,
            OneHotEncoder,
            LabelEncoder,
        )
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import numpy as np
        
        if self.cv_group_column is None:
            raise ValueError("cv_group_column must be specified for k-fold cross-validation")
        
        df = self.load()
        
        if self.labels is None:
            raise ValueError("Label column(s) must be provided in config")
        
        if self.features is None:
            # default: all columns except labels and group column
            self.features = [c for c in df.columns if c not in self.labels and c != self.cv_group_column]
        
        # Get group IDs
        if self.cv_group_column not in df.columns:
            raise ValueError(f"Group column '{self.cv_group_column}' not found in data")
        
        groups = df[self.cv_group_column].copy()
        
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
            print(
                f"Warning: Dropping {n_dropped_target} rows with NaN in target {self.labels}"
            )
            X = X[mask_target]
            y = y[mask_target]
            groups = groups[mask_target]
        
        # Drop rows with NaN in any features
        mask_features = ~X.isna().any(axis=1)
        if not mask_features.all():
            n_dropped_features = (~mask_features).sum()
            cols_with_nan = X.columns[X.isna().any()].tolist()
            print(
                f"Warning: Dropping {n_dropped_features} additional rows with NaN in features: {cols_with_nan}"
            )
            X = X[mask_features]
            y = y[mask_features]
            groups = groups[mask_features]
        
        # Reset indices after dropping rows
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        groups = groups.reset_index(drop=True)
        
        # Encode labels if they are strings/categorical
        self.label_encoders = {}
        self.label_classes_dict = {}
        
        if len(self.labels) == 1:
            # Single output
            if y.dtype == "object" or y.dtype.name == "category":
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                self.label_encoders[self.labels[0]] = label_encoder
                self.label_classes_dict[self.labels[0]] = label_encoder.classes_
                self.label_encoder = label_encoder
                self.label_classes = label_encoder.classes_
            else:
                self.label_encoder = None
                self.label_classes = None
        else:
            # Multi-output
            y_encoded = []
            for col in self.labels:
                if y[col].dtype == "object" or y[col].dtype.name == "category":
                    label_encoder = LabelEncoder()
                    y_col_encoded = label_encoder.fit_transform(y[col])
                    self.label_encoders[col] = label_encoder
                    self.label_classes_dict[col] = label_encoder.classes_
                else:
                    y_col_encoded = y[col].values
                    self.label_encoders[col] = None
                    self.label_classes_dict[col] = None
                y_encoded.append(y_col_encoded)
            y = np.column_stack(y_encoded)
            self.label_encoder = None
            self.label_classes = None
        
        # Identify numeric vs categorical
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]
        
        transformers = []
        
        if numeric_cols:
            scaler = StandardScaler() if self.scaling == "standard" else MinMaxScaler()
            num_pipeline = Pipeline([("imputer", SimpleImputer(strategy=self.impute_strategy)), ("scaler", scaler)])
            transformers.append(("num", num_pipeline, numeric_cols))
        
        if cat_cols:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, cat_cols))
        
        # Store the preprocessor template and feature names (but don't fit on all data)
        self.preprocessor_template = ColumnTransformer(transformers=transformers, remainder="drop")
        
        # We need to fit preprocessor on a sample to get feature names
        # Use just the first group to avoid leakage
        sample_preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        sample_preprocessor.fit(X)
        self.processed_feature_names = self._get_feature_names_after_transform(sample_preprocessor, numeric_cols, cat_cols)
        
        # Determine number of folds based on unique groups
        unique_groups = groups.unique()
        n_folds = len(unique_groups)
        
        print(f"\nK-Fold Cross-Validation Setup:")
        print(f"  Group column: {self.cv_group_column}")
        print(f"  Number of unique groups: {n_folds}")
        print(f"  Number of folds: {n_folds}")
        print(f"  Samples per group: {groups.value_counts().to_dict()}\n")
        
        # Create k-fold splits
        group_kfold = GroupKFold(n_splits=n_folds)
        
        fold_data = []
        # NOTE: Keep X as DataFrame for ColumnTransformer which uses column names
        for fold_idx, (train_val_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
            # Get raw (unscaled) train_val and test data - keep as DataFrames
            X_train_val_raw = X.iloc[train_val_idx]
            y_train_val = y[train_val_idx] if isinstance(y, np.ndarray) else y.iloc[train_val_idx]
            X_test_raw = X.iloc[test_idx]
            y_test = y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx]
            
            # Create a NEW preprocessor for this fold (to avoid data leakage)
            fold_preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            
            # Fit preprocessor ONLY on train_val data (not on test!)
            X_train_val_proc = fold_preprocessor.fit_transform(X_train_val_raw)
            
            # Transform test data using the fitted preprocessor
            X_test_proc = fold_preprocessor.transform(X_test_raw)
            
            # Further split train_val into train and val
            if self.val_size <= 0:
                X_train, X_val, y_train, y_val = X_train_val_proc, None, y_train_val, None
            else:
                # Use a fraction of train_val for validation
                from sklearn.model_selection import train_test_split
                val_fraction = self.val_size / (1.0 - (1.0 / n_folds))
                
                # For stratification
                if self.problem_type == "classification" and len(self.labels) == 1:
                    n_unique_train = len(np.unique(y_train_val))
                    stratify_train = y_train_val if n_unique_train <= 20 else None
                else:
                    stratify_train = None
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val_proc,
                    y_train_val,
                    test_size=val_fraction,
                    random_state=self.random_state,
                    stratify=stratify_train,
                )
            
            fold_data.append((
                X_train,
                X_val,
                X_test_proc,
                np.array(y_train),
                (None if y_val is None else np.array(y_val)),
                np.array(y_test),
                fold_idx
            ))
        
        return fold_data
    
    def prepare_grouped_splits(
        self,
        n_train_groups: int = 4,
        n_val_groups: int = 2,
        n_test_groups: int = 2,
    ):
        """Prepare train/val/test splits by assigning groups to each split.
        
        This method ensures no data leakage by keeping all samples from the same
        group together in one split (train, val, or test). Useful for spatial or
        temporal data where groups represent locations, patches, or time periods.
        
        Parameters
        ----------
        n_train_groups : int
            Number of groups to assign to training set
        n_val_groups : int
            Number of groups to assign to validation set (for hyperparameter tuning)
        n_test_groups : int
            Number of groups to assign to test set (for final evaluation)
        
        Returns
        -------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments)
            where group_assignments is a dict mapping split names to group IDs
        
        Raises
        ------
        ValueError
            If cv_group_column is not specified or if there aren't enough groups
        
        Notes
        -----
        - Requires cv_group_column to be set in the data config
        - Groups are shuffled before assignment to ensure randomness
        - Total groups needed = n_train_groups + n_val_groups + n_test_groups
        """
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import (
            StandardScaler,
            MinMaxScaler,
            OneHotEncoder,
            LabelEncoder,
        )
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import numpy as np
        
        if self.cv_group_column is None:
            raise ValueError(
                "cv_group_column must be specified for grouped train/val/test splits"
            )
        
        df = self.load()
        
        if self.labels is None:
            raise ValueError("Label column(s) must be provided in config")
        
        if self.features is None:
            # default: all columns except labels and group column
            self.features = [
                c for c in df.columns
                if c not in self.labels and c != self.cv_group_column
            ]
        
        # Get group IDs
        if self.cv_group_column not in df.columns:
            raise ValueError(f"Group column '{self.cv_group_column}' not found in data")
        
        groups = df[self.cv_group_column].copy()
        unique_groups = groups.unique()
        n_groups = len(unique_groups)
        
        # Validate we have enough groups
        total_groups_needed = n_train_groups + n_val_groups + n_test_groups
        if n_groups < total_groups_needed:
            raise ValueError(
                f"Not enough groups: need {total_groups_needed} "
                f"({n_train_groups} train + {n_val_groups} val + {n_test_groups} test), "
                f"but only have {n_groups} unique groups"
            )
        
        # Shuffle groups for random assignment
        rng = np.random.RandomState(self.random_state)
        shuffled_groups = rng.permutation(unique_groups)
        
        # Assign groups to splits
        train_groups = shuffled_groups[:n_train_groups]
        val_groups = shuffled_groups[n_train_groups:n_train_groups + n_val_groups]
        test_groups = shuffled_groups[n_train_groups + n_val_groups:n_train_groups + n_val_groups + n_test_groups]
        
        group_assignments = {
            'train': train_groups.tolist(),
            'val': val_groups.tolist(),
            'test': test_groups.tolist(),
        }
        
        print(f"\nGrouped Train/Val/Test Split Setup:")
        print(f"  Group column: {self.cv_group_column}")
        print(f"  Total unique groups: {n_groups}")
        print(f"  Train groups ({n_train_groups}): {train_groups.tolist()}")
        print(f"  Val groups ({n_val_groups}): {val_groups.tolist()}")
        print(f"  Test groups ({n_test_groups}): {test_groups.tolist()}")
        
        # Create masks for each split
        train_mask = groups.isin(train_groups)
        val_mask = groups.isin(val_groups)
        test_mask = groups.isin(test_groups)
        
        print(f"\n  Train samples: {train_mask.sum()}")
        print(f"  Val samples: {val_mask.sum()}")
        print(f"  Test samples: {test_mask.sum()}\n")
        
        # Extract features and labels
        X = df[self.features].copy()
        
        if len(self.labels) == 1:
            y = df[self.labels[0]].copy()
            mask_target = ~y.isna()
        else:
            y = df[self.labels].copy()
            mask_target = ~y.isna().any(axis=1)
        
        # Drop rows with NaN in target
        if mask_target.any() and not mask_target.all():
            n_dropped_target = (~mask_target).sum()
            print(
                f"Warning: Dropping {n_dropped_target} rows with NaN in target {self.labels}"
            )
            X = X[mask_target]
            y = y[mask_target]
            groups = groups[mask_target]
            train_mask = train_mask[mask_target]
            val_mask = val_mask[mask_target]
            test_mask = test_mask[mask_target]
        
        # Drop rows with NaN in features
        mask_features = ~X.isna().any(axis=1)
        if not mask_features.all():
            n_dropped_features = (~mask_features).sum()
            cols_with_nan = X.columns[X.isna().any()].tolist()
            print(
                f"Warning: Dropping {n_dropped_features} additional rows with NaN in features: {cols_with_nan}"
            )
            X = X[mask_features]
            y = y[mask_features]
            groups = groups[mask_features]
            train_mask = train_mask[mask_features]
            val_mask = val_mask[mask_features]
            test_mask = test_mask[mask_features]
        
        # Reset indices
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        groups = groups.reset_index(drop=True)
        train_mask = train_mask.reset_index(drop=True)
        val_mask = val_mask.reset_index(drop=True)
        test_mask = test_mask.reset_index(drop=True)
        
        # Encode labels
        self.label_encoders = {}
        self.label_classes_dict = {}
        
        if len(self.labels) == 1:
            if y.dtype == "object" or y.dtype.name == "category":
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                self.label_encoders[self.labels[0]] = label_encoder
                self.label_classes_dict[self.labels[0]] = label_encoder.classes_
                self.label_encoder = label_encoder
                self.label_classes = label_encoder.classes_
            else:
                self.label_encoder = None
                self.label_classes = None
        else:
            y_encoded = []
            for col in self.labels:
                if y[col].dtype == "object" or y[col].dtype.name == "category":
                    label_encoder = LabelEncoder()
                    y_col_encoded = label_encoder.fit_transform(y[col])
                    self.label_encoders[col] = label_encoder
                    self.label_classes_dict[col] = label_encoder.classes_
                else:
                    y_col_encoded = y[col].values
                    self.label_encoders[col] = None
                    self.label_classes_dict[col] = None
                y_encoded.append(y_col_encoded)
            y = np.column_stack(y_encoded)
            self.label_encoder = None
            self.label_classes = None
        
        # Build preprocessing pipeline
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]
        
        transformers = []
        
        if numeric_cols:
            scaler = StandardScaler() if self.scaling == "standard" else MinMaxScaler()
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy=self.impute_strategy)),
                ("scaler", scaler)
            ])
            transformers.append(("num", num_pipeline, numeric_cols))
        
        if cat_cols:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, cat_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        
        # Fit preprocessor on training data only (important!)
        X_train_raw = X[train_mask]
        preprocessor.fit(X_train_raw)
        
        # Transform all splits
        X_proc = preprocessor.transform(X)
        
        # Store preprocessor and feature names
        self.preprocessor = preprocessor
        self.processed_feature_names = self._get_feature_names_after_transform(
            preprocessor, numeric_cols, cat_cols
        )
        
        # Split the data
        X_train = X_proc[train_mask]
        X_val = X_proc[val_mask]
        X_test = X_proc[test_mask]
        
        # y should be numpy array at this point
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        
        # Extract group IDs for each split
        groups_train = groups[train_mask].values
        groups_val = groups[val_mask].values
        groups_test = groups[test_mask].values
        
        return (
            X_train,
            X_val,
            X_test,
            np.array(y_train),
            np.array(y_val),
            np.array(y_test),
            group_assignments,
            groups_train,
            groups_val,
            groups_test,
        )
