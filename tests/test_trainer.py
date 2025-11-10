"""Tests for training loop and seed handling."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil

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
    
    assert len(results) == 1
    assert 'y_pred' in results[0]
    assert 'seed' in results[0]
    assert 'model_path' in results[0]
    assert os.path.exists(results[0]['model_path'])


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
        
        assert len(results) == 3
        assert all('y_pred' in r for r in results)
        assert [r['seed'] for r in results] == [0, 1, 2]
    
    def test_trainer_explicit_seeds(self, temp_csv, temp_output_dir):
        """Test trainer with explicitly provided seeds."""
        loader = CSVDataLoader(path=temp_csv, features=['feature1', 'feature2'], label='label')
    
    trainer = Trainer(ModelZoo.get_model, problem_type='classification', output_dir=temp_output_dir)
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    assert len(results) == 3
    assert len(results[0]['y_pred']) == len(y_test)


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
    
    for result in results:
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
    
    assert 'y_proba' in results[0]
    assert results[0]['y_proba'] is not None
    assert results[0]['y_proba'].shape[0] == len(y_test)
