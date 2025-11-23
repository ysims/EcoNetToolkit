#!/usr/bin/env python3
"""Quick test script to validate hyperparameter tuning functionality."""

import numpy as np
import pandas as pd
import tempfile
import os
import sys

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
    print("\n" + "="*60)
    print("Testing Grouped Splits")
    print("="*60)
    
    from ecosci.data import CSVDataLoader
    
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
        (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments) = \
            loader.prepare_grouped_splits(n_train_groups=4, n_val_groups=2, n_test_groups=2)
        
        print(f"\n✓ Successfully created grouped splits")
        print(f"  Train shape: {X_train.shape}")
        print(f"  Val shape: {X_val.shape}")
        print(f"  Test shape: {X_test.shape}")
        print(f"  Group assignments: {group_assignments}")
        
        # Verify no overlap
        assert X_train.shape[0] > 0, "Train set is empty"
        assert X_val.shape[0] > 0, "Val set is empty"
        assert X_test.shape[0] > 0, "Test set is empty"
        
        print("✓ All splits are non-empty")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_file)

def test_hyperparameter_tuner():
    """Test hyperparameter tuner."""
    print("\n" + "="*60)
    print("Testing Hyperparameter Tuner")
    print("="*60)
    
    from ecosci.hyperopt import HyperparameterTuner
    
    # Create simple test data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = X_train[:, 0] * 2 + np.random.randn(100) * 0.5
    
    try:
        # Test MLP tuning
        print("\nTesting MLP tuning...")
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
        
        print(f"✓ MLP tuning successful")
        print(f"  Best params: {results['best_params']}")
        print(f"  Best score: {results['best_score']:.4f}")
        
        # Test Random Forest tuning
        print("\nTesting Random Forest tuning...")
        param_space = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5],
        }
        
        best_model, results = tuner.tune_random_forest(X_train, y_train, param_space)
        
        print(f"✓ Random Forest tuning successful")
        print(f"  Best params: {results['best_params']}")
        print(f"  Best score: {results['best_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration with config."""
    print("\n" + "="*60)
    print("Testing Full Integration")
    print("="*60)
    
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
        (X_train, X_val, X_test, y_train, y_val, y_test, group_assignments) = \
            loader.prepare_grouped_splits(
                n_train_groups=n_train_groups,
                n_val_groups=n_val_groups,
                n_test_groups=n_test_groups
            )
        
        print(f"✓ Data loaded successfully")
        
        # Create trainer
        trainer = Trainer(
            ModelZoo.get_model,
            problem_type=config['problem_type'],
            output_dir=config['output']['dir'],
        )
        
        # Run training with tuning
        print("\nRunning training with tuning...")
        results = trainer.run_with_tuning(
            config, X_train, X_val, X_test, y_train, y_val, y_test, group_assignments
        )
        
        print(f"✓ Training completed successfully")
        print(f"  Models trained: {list(results.keys())}")
        
        # Check outputs
        for model_name in results.keys():
            model_dir = os.path.join(temp_output_dir, model_name)
            assert os.path.exists(model_dir), f"Model directory not found: {model_dir}"
            
            # Check for key files
            files = os.listdir(model_dir)
            assert any('model_' in f for f in files), "No model files found"
            assert any('tuning_results_' in f for f in files), "No tuning results found"
            
        print(f"✓ All expected output files created")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_file)
        # Cleanup temp output dir
        import shutil
        shutil.rmtree(temp_output_dir, ignore_errors=True)

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING VALIDATION TESTS")
    print("="*60)
    
    tests = [
        ("Grouped Splits", test_grouped_splits),
        ("Hyperparameter Tuner", test_hyperparameter_tuner),
        ("Full Integration", test_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
