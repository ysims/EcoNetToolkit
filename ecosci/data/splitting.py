"""Data splitting utilities for cross-validation and grouped splits."""

from typing import List, Tuple
import numpy as np
import pandas as pd


def prepare_cv_folds(
    X: pd.DataFrame,
    y,
    groups: pd.Series,
    labels: List[str],
    numeric_cols: List[str],
    cat_cols: List[str],
    val_size: float,
    random_state: int,
    scaling: str,
    impute_strategy: str,
    problem_type: str,
    cv_group_column: str,
) -> List[Tuple]:
    """Prepare k-fold cross-validation splits using group-based splitting.
    
    Parameters
    ----------
    X : DataFrame
        Feature data
    y : array-like
        Target data (already encoded if categorical)
    groups : Series
        Group IDs for GroupKFold
    labels : list of str
        Names of target variables
    numeric_cols : list of str
        Names of numeric columns
    cat_cols : list of str
        Names of categorical columns
    val_size : float
        Validation set size as fraction
    random_state : int
        Random seed
    scaling : str
        "standard" or "minmax"
    impute_strategy : str
        Imputation strategy
    problem_type : str
        "classification" or "regression"
    cv_group_column : str
        Name of the group column
        
    Returns
    -------
    list of tuples
        Each tuple contains (X_train, X_val, X_test, y_train, y_val, y_test, fold_id)
    """
    from sklearn.model_selection import GroupKFold, train_test_split
    from .preprocessing import build_preprocessing_pipeline
    
    unique_groups = groups.unique()
    n_folds = len(unique_groups)
    
    print(f"\nK-Fold Cross-Validation Setup:")
    print(f"  Group column: {cv_group_column}")
    print(f"  Number of unique groups: {n_folds}")
    print(f"  Number of folds: {n_folds}")
    print(f"  Samples per group: {groups.value_counts().to_dict()}\n")
    
    # Create k-fold splits
    group_kfold = GroupKFold(n_splits=n_folds)
    
    fold_data = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
        # Get raw train_val and test data
        X_train_val_raw = X.iloc[train_val_idx]
        y_train_val = y[train_val_idx] if isinstance(y, np.ndarray) else y.iloc[train_val_idx]
        X_test_raw = X.iloc[test_idx]
        y_test = y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx]
        
        # Create a NEW preprocessor for this fold (to avoid data leakage)
        fold_preprocessor = build_preprocessing_pipeline(
            numeric_cols, cat_cols, scaling, impute_strategy
        )
        
        # Fit preprocessor ONLY on train_val data (not on test!)
        X_train_val_proc = fold_preprocessor.fit_transform(X_train_val_raw)
        X_test_proc = fold_preprocessor.transform(X_test_raw)
        
        # Further split train_val into train and val
        if val_size <= 0:
            X_train, X_val, y_train, y_val = X_train_val_proc, None, y_train_val, None
        else:
            val_fraction = val_size / (1.0 - (1.0 / n_folds))
            
            # For stratification
            if problem_type == "classification" and len(labels) == 1:
                n_unique_train = len(np.unique(y_train_val))
                stratify_train = y_train_val if n_unique_train <= 20 else None
            else:
                stratify_train = None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val_proc,
                y_train_val,
                test_size=val_fraction,
                random_state=random_state,
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
    X: pd.DataFrame,
    y,
    groups: pd.Series,
    labels: List[str],
    numeric_cols: List[str],
    cat_cols: List[str],
    random_state: int,
    scaling: str,
    impute_strategy: str,
    n_train_groups: int,
    n_val_groups: int,
    n_test_groups: int,
    cv_group_column: str,
) -> Tuple:
    """Prepare train/val/test splits by assigning groups to each split.
    
    This function ensures no data leakage by keeping all samples from the same
    group together in one split (train, val, or test).
    
    Parameters
    ----------
    X : DataFrame
        Feature data
    y : array-like
        Target data (already encoded if categorical)
    groups : Series
        Group IDs
    labels : list of str
        Names of target variables
    numeric_cols : list of str
        Names of numeric columns
    cat_cols : list of str
        Names of categorical columns
    random_state : int
        Random seed
    scaling : str
        "standard" or "minmax"
    impute_strategy : str
        Imputation strategy
    n_train_groups : int
        Number of groups for training
    n_val_groups : int
        Number of groups for validation
    n_test_groups : int
        Number of groups for testing
    cv_group_column : str
        Name of the group column
        
    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments,
         groups_train, groups_val, groups_test, preprocessor)
        
    X_train : ndarray
        Preprocessed training features
    X_val : ndarray
        Preprocessed validation features
    X_test : ndarray
        Preprocessed testing features
    y_train : ndarray
        Training target values
    y_val : ndarray
        Validation target values
    y_test : ndarray
        Testing target values
    group_assignments : dict
        Dictionary mapping split names to group IDs
    groups_train : ndarray
        Group IDs for training samples
    groups_val : ndarray
        Group IDs for validation samples
    groups_test : ndarray
        Group IDs for testing samples
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    """
    from .preprocessing import build_preprocessing_pipeline
    
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
    rng = np.random.RandomState(random_state)
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
    print(f"  Group column: {cv_group_column}")
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
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_cols, cat_cols, scaling, impute_strategy
    )
    
    # Fit preprocessor on training data only (important!)
    X_train_raw = X[train_mask]
    preprocessor.fit(X_train_raw)
    
    # Transform all splits
    X_proc = preprocessor.transform(X)
    
    # Split the data
    X_train = X_proc[train_mask]
    X_val = X_proc[val_mask]
    X_test = X_proc[test_mask]
    
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
        preprocessor,  # Return for feature name extraction
    )
