"""Preprocessing utilities for feature extraction and transformation."""

from typing import List


def get_feature_names_after_transform(preprocessor, numeric_cols: List[str], cat_cols: List[str]) -> List[str]:
    """Get feature names after preprocessing (including one-hot encoding).
    
    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted sklearn ColumnTransformer with numeric and categorical pipelines
    numeric_cols : list of str
        Names of numeric columns
    cat_cols : list of str
        Names of categorical columns
        
    Returns
    -------
    list of str
        Feature names after transformation
    """
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


def build_preprocessing_pipeline(numeric_cols: List[str], cat_cols: List[str], 
                                 scaling: str = "standard", impute_strategy: str = "mean"):
    """Build a scikit-learn preprocessing pipeline.
    
    Parameters
    ----------
    numeric_cols : list of str
        Names of numeric columns
    cat_cols : list of str
        Names of categorical columns
    scaling : str
        "standard" or "minmax" scaling for numeric features
    impute_strategy : str
        Imputation strategy for numeric features ("mean", "median", "most_frequent")
        
    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline ready to fit
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    transformers = []
    
    if numeric_cols:
        scaler = StandardScaler() if scaling == "standard" else MinMaxScaler()
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", scaler)
        ])
        transformers.append(("num", num_pipeline, numeric_cols))
    
    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))
    
    return ColumnTransformer(transformers=transformers, remainder="drop")


def encode_labels(y, labels: List[str], problem_type: str = "classification"):
    """Encode labels if they are categorical strings.
    
    Parameters
    ----------
    y : pandas Series or DataFrame
        Target variable(s)
    labels : list of str
        Names of target variable(s)
    problem_type : str
        "classification" or "regression"
        
    Returns
    -------
    tuple
        (encoded_y, label_encoders_dict, label_classes_dict, label_encoder, label_classes)
        Last two maintain backward compatibility for single output
    """
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import pandas as pd
    
    label_encoders = {}
    label_classes_dict = {}
    
    if len(labels) == 1:
        # Single output: maintain backward compatibility
        if y.dtype == "object" or y.dtype.name == "category":
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            label_encoders[labels[0]] = label_encoder
            label_classes_dict[labels[0]] = label_encoder.classes_
            return y_encoded, label_encoders, label_classes_dict, label_encoder, label_encoder.classes_
        else:
            return y, label_encoders, label_classes_dict, None, None
    else:
        # Multi-output: encode each column separately
        y_encoded = []
        for col in labels:
            if y[col].dtype == "object" or y[col].dtype.name == "category":
                label_encoder = LabelEncoder()
                y_col_encoded = label_encoder.fit_transform(y[col])
                label_encoders[col] = label_encoder
                label_classes_dict[col] = label_encoder.classes_
            else:
                y_col_encoded = y[col].values
                label_encoders[col] = None
                label_classes_dict[col] = None
            y_encoded.append(y_col_encoded)
        y_encoded = np.column_stack(y_encoded)
        return y_encoded, label_encoders, label_classes_dict, None, None
