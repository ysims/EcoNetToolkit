"""Feature importance extraction and analysis.

This module contains functions for extracting and analyzing feature importance
from trained models, including built-in importance (tree models, linear models)
and permutation importance (for models like MLPs).
"""

from typing import Dict, Optional, List
import os
import numpy as np
import pandas as pd

from .metrics import safe_std
from .plotting import plot_feature_importance


def extract_feature_importance(
    results: Dict,
    X_test,
    y_test,
    output_dir: str,
    problem_type: str,
    feature_names: Optional[List[str]] = None
):
    """Extract and report feature importance for all models.
    
    Parameters
    ----------
    results : Dict
        Results dictionary {model_name: [run_results]}
    X_test : array-like
        Test features
    y_test : array-like
        Test labels/values
    output_dir : str
        Directory to save outputs
    problem_type : str
        "classification" or "regression"
    feature_names : Optional[List[str]]
        Names of features
    """
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
    
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    for model_name, model_results in results.items():
        try:
            # Collect feature importances from all trained models
            all_importances = []
            
            for result in model_results:
                model_path = result.get("model_path")
                
                if not model_path or not os.path.exists(model_path):
                    continue
                    
                model = joblib.load(model_path)
                
                # Extract feature importances based on model type
                feature_importances = None
                importance_method = None
                
                # For multi-output models, need to handle differently
                if isinstance(model, (MultiOutputClassifier, MultiOutputRegressor)):
                    # Get the base estimator
                    if hasattr(model.estimators_[0], 'feature_importances_'):
                        # Average importances across outputs
                        importances_list = [est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')]
                        if importances_list:
                            feature_importances = np.mean(importances_list, axis=0)
                            importance_method = "built-in (averaged across outputs)"
                elif hasattr(model, 'feature_importances_'):
                    # Tree-based models (Random Forest, XGBoost)
                    feature_importances = model.feature_importances_
                    importance_method = "built-in (Gini/gain)"
                elif hasattr(model, 'coef_'):
                    # Linear models (use absolute coefficient values)
                    coef = model.coef_
                    if len(coef.shape) > 1:
                        # Multi-output or multi-class: average absolute coefficients
                        feature_importances = np.mean(np.abs(coef), axis=0)
                    else:
                        feature_importances = np.abs(coef)
                    importance_method = "coefficient magnitude"
                
                # If no built-in importance, use permutation importance (works for all models including MLPs)
                if feature_importances is None and X_test is not None:
                    # Only print once for the first model
                    if len(all_importances) == 0:
                        print(f"\n{model_name.upper()}: Computing permutation importance across all {len(model_results)} models...")
                        print(f"  (This measures performance drop when each feature is shuffled)")
                    
                    # Determine scoring metric based on problem type
                    if problem_type == "classification":
                        scoring = "accuracy"
                    else:  # regression
                        scoring = "r2"
                    
                    # Compute permutation importance
                    perm_importance = permutation_importance(
                        model, X_test, y_test, 
                        n_repeats=10,  # Number of times to permute each feature
                        random_state=42,
                        scoring=scoring,
                        n_jobs=-1  # Use all available cores
                    )
                    feature_importances = perm_importance.importances_mean
                    importance_method = f"permutation (based on {scoring}, averaged across seeds)"
                elif feature_importances is not None and importance_method:
                    # Update method to indicate averaging across seeds
                    if len(all_importances) == 0:
                        importance_method = importance_method + f", averaged across {len(model_results)} seeds"
                
                if feature_importances is not None:
                    all_importances.append(feature_importances)
            
            # Average importances across all models
            if all_importances:
                # Average across all seeds
                avg_feature_importances = np.mean(all_importances, axis=0)
                std_feature_importances = np.std(all_importances, axis=0)
                
                print(f"\n{model_name.upper()}:")
                print(f"{'-'*80}")
                print(f"Method: {importance_method}")
                print(f"Averaged across {len(all_importances)} model(s)")
                
                # Sort features by importance
                indices = np.argsort(avg_feature_importances)[::-1]
                
                # Helper to get feature label
                def get_feature_label(idx):
                    if feature_names and idx < len(feature_names):
                        return feature_names[idx]
                    return f"Feature {idx}"
                
                # Print top 20 features with mean ± std
                print(f"\nTop 20 most important features (mean ± std):")
                for i, idx in enumerate(indices[:20], 1):
                    importance = avg_feature_importances[idx]
                    std = std_feature_importances[idx]
                    feat_label = get_feature_label(idx)
                    print(f"  {i:2d}. {feat_label:40s}: {importance:.6f} ± {std:.6f}")
                
                # Create feature importance plot
                plot_feature_importance(
                    avg_feature_importances,
                    std_feature_importances,
                    feature_names,
                    model_name,
                    output_dir,
                    importance_method,
                    top_n=20
                )
            else:
                print(f"\n{model_name.upper()}: Feature importance not available (no models or no test data provided)")
        except Exception as e:
            print(f"\nWarning: Could not extract feature importance for {model_name}: {e}")
    
    print(f"\n{'='*80}\n")


def extract_cv_feature_importance(
    results: Dict,
    output_dir: str,
    problem_type: str,
    feature_names: Optional[List[str]] = None
):
    """Extract and report feature importance for cross-validation results.
    
    Parameters
    ----------
    results : Dict
        Results dictionary {model_name: [run_results_with_fold_info]}
    output_dir : str
        Directory to save outputs
    problem_type : str
        "classification" or "regression"
    feature_names : Optional[List[str]]
        Names of features
    """
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
    
    print(f"{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS (per fold and averaged)")
    print(f"{'='*80}")
    
    for model_name, model_results in results.items():
        try:
            # Group results by fold
            folds_for_fi = {}
            for r in model_results:
                fold_id = r.get("fold")
                if fold_id not in folds_for_fi:
                    folds_for_fi[fold_id] = []
                folds_for_fi[fold_id].append(r)
            
            print(f"\n{model_name.upper()}: Collecting feature importances from all {len(folds_for_fi)} folds...")
            
            # Collect feature importance for each fold
            fold_importances_dict = {}
            importance_method = None
            
            for fold_id in sorted(folds_for_fi.keys()):
                fold_results = folds_for_fi[fold_id]
                fold_importances_list = []
                
                for result in fold_results:
                    model_path = result.get("model_path")
                    
                    if not model_path or not os.path.exists(model_path):
                        continue
                    
                    model = joblib.load(model_path)
                    y_test_fold = result.get("y_test")
                    X_test_fold = result.get("X_test")
                    
                    # Extract feature importances based on model type
                    feature_importances = None
                    
                    # For multi-output models
                    if isinstance(model, (MultiOutputClassifier, MultiOutputRegressor)):
                        if hasattr(model.estimators_[0], 'feature_importances_'):
                            importances_list = [est.feature_importances_ for est in model.estimators_ if hasattr(est, 'feature_importances_')]
                            if importances_list:
                                feature_importances = np.mean(importances_list, axis=0)
                                importance_method = "built-in (averaged across outputs, seeds, and folds)"
                    elif hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                        importance_method = "built-in (Gini/gain, averaged across seeds and folds)"
                    elif hasattr(model, 'coef_'):
                        coef = model.coef_
                        if len(coef.shape) > 1:
                            feature_importances = np.mean(np.abs(coef), axis=0)
                        else:
                            feature_importances = np.abs(coef)
                        importance_method = "coefficient magnitude (averaged across seeds and folds)"
                    
                    # If no built-in importance, use permutation importance (e.g., for MLPs)
                    if feature_importances is None and X_test_fold is not None and y_test_fold is not None:
                        # Only print once per fold
                        if len(fold_importances_list) == 0:
                            print(f"  Fold {fold_id}: Computing permutation importance (this may take a while)...")
                        
                        # Determine scoring metric based on problem type
                        if problem_type == "classification":
                            scoring = "accuracy"
                        else:  # regression
                            scoring = "r2"
                        
                        # Compute permutation importance
                        perm_importance = permutation_importance(
                            model, X_test_fold, y_test_fold,
                            n_repeats=10,  # Number of times to permute each feature
                            random_state=42,
                            scoring=scoring,
                            n_jobs=-1  # Use all available cores
                        )
                        feature_importances = perm_importance.importances_mean
                        importance_method = f"permutation (based on {scoring}, averaged across seeds and folds)"
                    
                    if feature_importances is not None:
                        fold_importances_list.append(feature_importances)
                
                # Average across seeds for this fold
                if fold_importances_list:
                    fold_avg_importance = np.mean(fold_importances_list, axis=0)
                    fold_importances_dict[fold_id] = fold_avg_importance
            
            # Average importances across all folds
            if fold_importances_dict:
                all_fold_importances = list(fold_importances_dict.values())
                avg_feature_importances = np.mean(all_fold_importances, axis=0)
                std_feature_importances = np.std(all_fold_importances, axis=0)
                
                print(f"\n{model_name.upper()}:")
                print(f"{'-'*80}")
                print(f"Method: {importance_method}")
                print(f"Computed from all {len(fold_importances_dict)} folds")
                
                # Sort features by importance
                indices = np.argsort(avg_feature_importances)[::-1]
                
                # Helper to get feature label
                def get_feature_label(idx):
                    if feature_names and idx < len(feature_names):
                        return feature_names[idx]
                    return f"Feature {idx}"
                
                # Print top 20 features
                print(f"\nTop 20 most important features (mean ± std):")
                for i, idx in enumerate(indices[:20], 1):
                    importance = avg_feature_importances[idx]
                    std = std_feature_importances[idx]
                    feat_label = get_feature_label(idx)
                    print(f"  {i:2d}. {feat_label:40s}: {importance:.6f} ± {std:.6f}")
                
                # Save feature importance to CSV (all features, sorted by importance)
                feature_importance_data = []
                for rank, idx in enumerate(indices, 1):
                    row = {
                        "rank": rank,
                        "feature": get_feature_label(idx),
                        "importance_mean_all_folds": avg_feature_importances[idx],
                        "importance_std_across_folds": std_feature_importances[idx],
                        "feature_index": idx
                    }
                    # Add per-fold importance
                    for fold_id in sorted(fold_importances_dict.keys()):
                        row[f"importance_fold_{fold_id}"] = fold_importances_dict[fold_id][idx]
                    feature_importance_data.append(row)
                
                fi_df = pd.DataFrame(feature_importance_data)
                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                fi_csv_path = os.path.join(model_dir, f"feature_importance_{model_name}_cv_all_folds.csv")
                fi_df.to_csv(fi_csv_path, index=False)
                print(f"\n  Feature importance CSV (all folds) saved to: {fi_csv_path}")
                
                # Create feature importance plot
                plot_feature_importance(
                    avg_feature_importances,
                    std_feature_importances,
                    feature_names,
                    f"{model_name}_cv",
                    output_dir,
                    importance_method,
                    top_n=20
                )
            else:
                print(f"\n{model_name.upper()}: Feature importance not available")
        except Exception as e:
            print(f"\nWarning: Could not extract feature importance for {model_name}: {e}")
    
    print(f"\n{'='*80}\n")
