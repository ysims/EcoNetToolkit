"""Tests for data loading and preprocessing."""
import pytest
import numpy as np
import pandas as pd
import tempfile

from ecosci.data import CSVDataLoader


@pytest.fixture
def sample_csv():
    """Create a temporary CSV with mixed data types."""
    data = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5,
        'num2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 5,
        'cat': ['A', 'B', 'C'] * 16 + ['A', 'B'],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f, index=False)
        return f.name


def test_data_loader_creates_splits(sample_csv):
    """Test that data loader creates train/val/test splits."""
    loader = CSVDataLoader(
        path=sample_csv,
        features=['num1', 'num2', 'cat'],
        label='label',
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()
    
    # Verify we got all the data split properly
    total_samples = len(y_train) + len(y_val) + len(y_test)
    assert total_samples == 50
    assert len(y_train) > len(y_test)  # train should be largest
    assert len(X_train) == len(y_train)


def test_categorical_encoding_increases_features(sample_csv):
    """Test that categorical variables get one-hot encoded."""
    loader = CSVDataLoader(
        path=sample_csv,
        features=['num1', 'num2', 'cat'],
        label='label'
    )
    X_train, _, _, _, _, _ = loader.prepare()
    
    # 2 numeric + 3 categories one-hot encoded = 5 features
    assert X_train.shape[1] == 5


def test_scaling_produces_clean_data(sample_csv):
    """Test that scaling doesn't introduce NaN or inf."""
    for scaling in ['standard', 'minmax']:
        loader = CSVDataLoader(
            path=sample_csv,
            features=['num1', 'num2'],
            label='label',
            scaling=scaling
        )
        X_train, _, _, _, _, _ = loader.prepare()
        
        assert not np.isnan(X_train).any()
        assert not np.isinf(X_train).any()


def test_random_state_makes_splits_reproducible(sample_csv):
    """Test that same random_state gives same splits."""
    loader1 = CSVDataLoader(path=sample_csv, features=['num1'], label='label', random_state=42)
    X_train1, _, _, y_train1, _, _ = loader1.prepare()
    
    loader2 = CSVDataLoader(path=sample_csv, features=['num1'], label='label', random_state=42)
    X_train2, _, _, y_train2, _, _ = loader2.prepare()
    
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(y_train1, y_train2)
