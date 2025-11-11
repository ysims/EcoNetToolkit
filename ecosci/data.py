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
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 0,
        scaling: str = "standard",
        impute_strategy: str = "mean",
    ):
        self.path = path
        self.features = features
        self.label = label
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaling = scaling
        self.impute_strategy = impute_strategy

    def load(self) -> pd.DataFrame:
        """Read the CSV into a DataFrame."""
        df = pd.read_csv(self.path)
        return df

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

        if self.label is None:
            raise ValueError("Label column must be provided in config")

        if self.features is None:
            # default: all columns except label
            self.features = [c for c in df.columns if c != self.label]

        X = df[self.features].copy()
        y = df[self.label].copy()

        # Encode labels if they are strings/categorical (needed for XGBoost and others)
        if y.dtype == "object" or y.dtype.name == "category":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            # Store the encoder for later use if needed
            self.label_encoder = label_encoder
            self.label_classes = label_encoder.classes_
        else:
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

        # Check number of unique classes for stratification
        n_unique = len(np.unique(y))
        stratify_y = y if n_unique <= 20 else None

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
            n_unique_train = len(np.unique(y_train_val))
            stratify_train = y_train_val if n_unique_train <= 20 else None
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
