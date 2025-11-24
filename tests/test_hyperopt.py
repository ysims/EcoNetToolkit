#!/usr/bin/env python3
"""Tests for hyperparameter tuning functionality.

This module tests the hyperparameter optimisation features including:
- Grouped train/validation/test splits
- Hyperparameter tuning for different model types (MLP, Random Forest)
- Full integration of tuning with the training pipeline
"""

import numpy as np
import pandas as pd
import tempfile
import os
import sys

from ecosci.data import CSVDataLoader

def create_test_data():
    """Create a small synthetic dataset with groups for testing."""
    np.random.seed(42)
    
    n_samples = 200
    n_features = 5
    n_groups = 8
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create groups (e.g., patches)
    groups = np.repeat(range(n_groups), n_samples // n_groups)
    
    # Create target with some signal
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.5
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    df['group_id'] = groups
    
    return df

def test_grouped_splits():
    """Test grouped train/val/test splits."""
    # Create test data
    df = create_test_data()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Create loader
        loader = CSVDataLoader(
            path=temp_file,
            features=[f'feature_{i}' for i in range(5)],
            labels=['target'],
            cv_group_column='group_id',
            problem_type='regression',
        )
        
        # Test grouped splits
        (X_train, X_val, X_test, _, _, _, group_assignments, _, _, _) = \
            loader.prepare_grouped_splits(n_train_groups=4, n_val_groups=2, n_test_groups=2)
        
        # Verify no overlap
        assert X_train.shape[0] > 0, "Train set is empty"
        assert X_val.shape[0] > 0, "Val set is empty"
        assert X_test.shape[0] > 0, "Test set is empty"
        assert len(group_assignments) == 3, "Should have 3 group assignments (train, val, test)"

    finally:
        os.unlink(temp_file)

def test_hyperparameter_tuner_mlp():
    """Test hyperparameter tuner with MLP."""
    from ecosci.hyperopt import HyperparameterTuner
    
    # Create simple test data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = X_train[:, 0] * 2 + np.random.randn(100) * 0.5
    
    # Test MLP tuning
    tuner = HyperparameterTuner(
        problem_type='regression',
        n_iter=3,  # Small for testing
        cv=2,      # Small for testing
        verbose=0,
    )
    
    param_space = {
        'hidden_layer_sizes': [(16,), (32,)],
        'alpha': [0.001, 0.01],
        'max_iter': [100],
    }
    
    best_model, results = tuner.tune_mlp(X_train, y_train, param_space)
    
    assert best_model is not None, "Best model should not be None"
    assert 'best_params' in results, "Results should contain best_params"
    assert 'best_score' in results, "Results should contain best_score"


def test_hyperparameter_tuner_random_forest():
    """Test hyperparameter tuner with Random Forest."""
    from ecosci.hyperopt import HyperparameterTuner
    
    # Create simple test data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = X_train[:, 0] * 2 + np.random.randn(100) * 0.5
    
    # Test Random Forest tuning
    tuner = HyperparameterTuner(
        problem_type='regression',
        n_iter=3,  # Small for testing
        cv=2,      # Small for testing
        verbose=0,
    )
    
    param_space = {
        'n_estimators': [10, 20],
        'max_depth': [3, 5],
    }
    
    best_model, results = tuner.tune_random_forest(X_train, y_train, param_space)
    
    assert best_model is not None, "Best model should not be None"
    assert 'best_params' in results, "Results should contain best_params"
    assert 'best_score' in results, "Results should contain best_score"
def test_tuning_integration():
    """Test full integration with config and hyperparameter tuning."""
    # Create test data
    df = create_test_data()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    # Create temporary output directory
    temp_output_dir = tempfile.mkdtemp()
    
    try:
        # Create a minimal config
        config = {
            'problem_type': 'regression',
            'data': {
                'path': temp_file,
                'features': [f'feature_{i}' for i in range(5)],
                'labels': ['target'],
                'cv_group_column': 'group_id',
                'n_train_groups': 4,
                'n_val_groups': 2,
                'n_test_groups': 2,
                'scaling': 'standard',
            },
            'tuning': {
                'enabled': True,
                'search_method': 'random',
                'n_iter': 2,  # Very small for testing
                'cv_folds': 2,
                'verbose': 0,
            },
            'models': [
                {
                    'name': 'linear',
                    'param_space': {
                        'alpha': [0.1, 1.0],
                    }
                }
            ],
            'training': {
                'repetitions': 2,  # Small for testing
                'random_seed': 42,
            },
            'output': {
                'dir': temp_output_dir,
            }
        }
        
        from ecosci.data import CSVDataLoader
        from ecosci.trainer import Trainer
        from ecosci.models import ModelZoo
        
        # Load data - extract n_*_groups before passing to loader
        n_train_groups = config['data'].pop('n_train_groups')
        n_val_groups = config['data'].pop('n_val_groups')
        n_test_groups = config['data'].pop('n_test_groups')
        
        loader = CSVDataLoader(**config['data'], problem_type=config['problem_type'])
        (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments, 
         groups_train, groups_val, groups_test) = \
            loader.prepare_grouped_splits(
                n_train_groups=n_train_groups,
                n_val_groups=n_val_groups,
                n_test_groups=n_test_groups
            )
        
        # Create trainer
        trainer = Trainer(
            ModelZoo.get_model,
            problem_type=config['problem_type'],
            output_dir=config['output']['dir'],
        )
        
        # Run training with tuning
        results = trainer.run_with_tuning(
            config, X_train, X_val, X_test, y_train, y_val, y_test, group_assignments
        )
        
        # Check outputs
        assert len(results) > 0, "No models were trained"
        
        for model_name in results.keys():
            model_dir = os.path.join(temp_output_dir, model_name)
            assert os.path.exists(model_dir), f"Model directory not found: {model_dir}"
            
            # Check for key files
            files = os.listdir(model_dir)
            assert any('model_' in f for f in files), "No model files found"
            assert any('tuning_results_' in f for f in files), "No tuning results found"
        
    finally:
        os.unlink(temp_file)
        # Cleanup temp output dir
        import shutil
        shutil.rmtree(temp_output_dir, ignore_errors=True)


if __name__ == "__main__":
    # This file can now be run with pytest
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))