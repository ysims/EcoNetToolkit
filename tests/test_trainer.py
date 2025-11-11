"""Tests for training loop and seed handling."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
import joblib

from ecosci.data import CSVDataLoader
from ecosci.models import ModelZoo
from ecosci.trainer import Trainer


@pytest.fixture
def sample_csv():
    """Create a temporary CSV file with sample data."""
    np.random.seed(42)
    n = 50
    data = {
        'num1': np.random.randn(n),
        'num2': np.random.randn(n) + 1,
        'label': np.random.choice([0, 1], n)
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        return f.name


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_trainer_runs_and_saves_models(sample_csv, temp_output_dir):
    """Test that trainer completes and saves model files."""
    loader = CSVDataLoader(path=sample_csv, features=['num1', 'num2'], label='label')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    cfg = {
        'model': {'name': 'logistic', 'params': {'random_state': 42}},
        'training': {'repetitions': 1, 'random_seed': 42},
        'output_dir': temp_output_dir
    }
    
    trainer = Trainer(ModelZoo.get_model, problem_type='classification', output_dir=temp_output_dir)
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    assert 'logistic' in results
    assert len(results['logistic']) == 1
    assert 'y_pred' in results['logistic'][0]
    assert 'seed' in results['logistic'][0]
    assert 'model_path' in results['logistic'][0]
    assert os.path.exists(results['logistic'][0]['model_path'])


def test_trainer_multiple_seeds_produces_multiple_results(sample_csv, temp_output_dir):
    """Test that multiple seeds produce multiple result dictionaries."""
    loader = CSVDataLoader(path=sample_csv, features=['num1', 'num2'], label='label')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    cfg = {
        'model': {'name': 'random_forest', 'params': {'n_estimators': 10}},
        'training': {'repetitions': 3, 'random_seed': 0},
        'output_dir': temp_output_dir
    }
    
    trainer = Trainer(ModelZoo.get_model, problem_type='classification', output_dir=temp_output_dir)
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    assert 'random_forest' in results
    assert len(results['random_forest']) == 3
    assert len(results['random_forest'][0]['y_pred']) == len(y_test)



def test_trainer_saves_joblib_files(sample_csv, temp_output_dir):
    """Test that trained models are saved as joblib files."""
    loader = CSVDataLoader(path=sample_csv, features=['num1', 'num2'], label='label')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    cfg = {
        'model': {'name': 'logistic', 'params': {}},
        'training': {'repetitions': 2, 'random_seed': 0},
        'output_dir': temp_output_dir
    }
    
    trainer = Trainer(ModelZoo.get_model, problem_type='classification', output_dir=temp_output_dir)
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    for result in results['logistic']:
        assert os.path.exists(result['model_path'])
        assert result['model_path'].endswith('.joblib')


def test_trainer_captures_probabilities(sample_csv, temp_output_dir):
    """Test that probabilities are captured when model supports them."""
    loader = CSVDataLoader(path=sample_csv, features=['num1', 'num2'], label='label')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    cfg = {
        'model': {'name': 'random_forest', 'params': {'n_estimators': 10, 'random_state': 42}},
        'training': {'repetitions': 1, 'random_seed': 0},
        'output_dir': temp_output_dir
    }
    
    trainer = Trainer(ModelZoo.get_model, problem_type='classification', output_dir=temp_output_dir)
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    assert 'y_proba' in results['random_forest'][0]
    assert results['random_forest'][0]['y_proba'] is not None
    assert results['random_forest'][0]['y_proba'].shape[0] == len(y_test)


# Regression tests
@pytest.fixture
def regression_csv():
    """Create a temporary CSV file with regression data."""
    np.random.seed(42)
    n = 50
    data = {
        'num1': np.random.randn(n),
        'num2': np.random.randn(n) + 1,
        'target': np.random.randn(n) * 10 + 50  # Continuous target
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        return f.name


def test_trainer_runs_regression(regression_csv, temp_output_dir):
    """Test that trainer works for regression problems."""
    loader = CSVDataLoader(path=regression_csv, features=['num1', 'num2'], label='target')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    cfg = {
        'model': {'name': 'random_forest', 'params': {'n_estimators': 10, 'random_state': 42}},
        'training': {'repetitions': 1, 'random_seed': 42},
        'output_dir': temp_output_dir
    }
    
    trainer = Trainer(ModelZoo.get_model, problem_type='regression', output_dir=temp_output_dir)
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    assert 'random_forest' in results
    assert len(results['random_forest']) == 1
    assert 'y_pred' in results['random_forest'][0]
    assert 'seed' in results['random_forest'][0]
    assert 'model_path' in results['random_forest'][0]
    assert os.path.exists(results['random_forest'][0]['model_path'])
    # Regression models should not have probabilities
    assert results['random_forest'][0].get('y_proba') is None


def test_trainer_regression_multiple_seeds(regression_csv, temp_output_dir):
    """Test that multiple seeds produce multiple regression results."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    
    loader = CSVDataLoader(path=regression_csv, features=['num1', 'num2'], label='target')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    cfg = {
        'model': {'name': 'mlp', 'params': {'hidden_layer_sizes': [16], 'max_iter': 100}},
        'training': {'repetitions': 3, 'random_seed': 0},
        'output_dir': temp_output_dir
    }
    
    trainer = Trainer(ModelZoo.get_model, problem_type='regression', output_dir=temp_output_dir)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    assert 'mlp' in results
    assert len(results['mlp']) == 3
    assert len(results['mlp'][0]['y_pred']) == len(y_test)
    
    # Check that predictions are continuous values
    for result in results['mlp']:
        y_pred = result['y_pred']
        assert isinstance(y_pred[0], (np.floating, float))


def test_trainer_regression_saves_models(regression_csv, temp_output_dir):
    """Test that regression models are saved properly."""
    loader = CSVDataLoader(path=regression_csv, features=['num1', 'num2'], label='target')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    cfg = {
        'model': {'name': 'svm', 'params': {'C': 1.0}},
        'training': {'repetitions': 2, 'random_seed': 0},
        'output_dir': temp_output_dir
    }
    
    trainer = Trainer(ModelZoo.get_model, problem_type='regression', output_dir=temp_output_dir)
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    for result in results['svm']:
        assert os.path.exists(result['model_path'])
        assert result['model_path'].endswith('.joblib')
        
        # Verify we can load the model
        loaded_model = joblib.load(result['model_path'])
        assert hasattr(loaded_model, 'predict')
