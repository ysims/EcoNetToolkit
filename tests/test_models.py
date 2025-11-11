"""Tests for model instantiation and basic functionality."""

import pytest
from ecosci.models import ModelZoo


@pytest.mark.parametrize("model_name", ["mlp", "random_forest", "svm", "logistic"])
def test_all_models_can_be_instantiated(model_name):
    """Test that all supported models can be created."""
    model = ModelZoo.get_model(
        model_name, problem_type="classification", params={"random_state": 42}
    )
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_xgboost_creation_if_available():
    """Test XGBoost instantiation (may not be installed)."""
    try:
        model = ModelZoo.get_model(
            "xgboost", problem_type="classification", params={"random_state": 42}
        )
        assert model is not None
    except ImportError:
        pytest.skip("xgboost not installed")


def test_invalid_model_name_raises_error():
    """Test that invalid model names raise errors."""
    with pytest.raises(ValueError):
        ModelZoo.get_model("invalid_model", problem_type="classification")


def test_mlp_custom_params_are_applied():
    """Test that MLP accepts custom parameters."""
    model = ModelZoo.get_model(
        "mlp",
        problem_type="classification",
        params={"hidden_layer_sizes": [64, 32], "max_iter": 100, "random_state": 42},
    )
    params = model.get_params()
    assert params["hidden_layer_sizes"] == (64, 32)
    assert params["max_iter"] == 100


def test_mlp_early_stopping():
    """Test MLP with early stopping enabled."""
    model = ModelZoo.get_model(
        "mlp",
        problem_type="classification",
        params={
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
        },
    )
    params = model.get_params()
    assert params["early_stopping"] is True
    assert params["validation_fraction"] == 0.1


class TestRandomForestParameters:
    """Test Random Forest parameter handling."""

    def test_rf_with_custom_params(self):
        """Test Random Forest with custom parameters."""
        model = ModelZoo.get_model(
            "random_forest",
            problem_type="classification",
            params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
        )
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 5


# Regression tests
@pytest.mark.parametrize("model_name", ['mlp', 'random_forest'])
def test_regression_models_can_be_instantiated(model_name):
    """Test that regression models can be created."""
    model = ModelZoo.get_model(model_name, problem_type='regression', params={'random_state': 42})
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_svm_regression_can_be_instantiated():
    """Test that SVM regressor can be created (no random_state param)."""
    model = ModelZoo.get_model('svm', problem_type='regression', params={})
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_xgboost_regression_creation_if_available():
    """Test XGBoost regression instantiation (may not be installed)."""
    try:
        model = ModelZoo.get_model('xgboost', problem_type='regression', params={'random_state': 42})
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    except ImportError:
        pytest.skip("xgboost not installed")


def test_mlp_regression_custom_params():
    """Test that MLP regressor accepts custom parameters."""
    model = ModelZoo.get_model(
        'mlp',
        problem_type='regression',
        params={
            'hidden_layer_sizes': [64, 32],
            'max_iter': 100,
            'random_state': 42
        }
    )
    params = model.get_params()
    assert params['hidden_layer_sizes'] == (64, 32)
    assert params['max_iter'] == 100


def test_random_forest_regression_custom_params():
    """Test Random Forest regressor with custom parameters."""
    model = ModelZoo.get_model(
        'random_forest',
        problem_type='regression',
        params={
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }
    )
    params = model.get_params()
    assert params['n_estimators'] == 50
    assert params['max_depth'] == 5


def test_svm_regression():
    """Test SVM regressor instantiation."""
    model = ModelZoo.get_model(
        'svm',
        problem_type='regression',
        params={
            'C': 1.0,
            'kernel': 'rbf'
        }
    )
    assert model is not None
    params = model.get_params()
    assert params['C'] == 1.0
    assert params['kernel'] == 'rbf'


