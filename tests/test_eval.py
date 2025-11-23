"""Tests for evaluation metrics and reporting."""

import pytest
import numpy as np
import tempfile
import shutil
import json
import os

from ecosci.eval import compute_classification_metrics, evaluate_and_report


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_metrics_are_in_valid_ranges():
    """Test that computed metrics are in valid ranges."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba=None)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "cohen_kappa" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert -1 <= metrics["cohen_kappa"] <= 1


def test_metrics_with_probabilities_include_roc_auc():
    """Test that ROC-AUC is computed when probabilities are provided."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]]
    )

    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)

    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1


def test_evaluate_and_report_creates_json(temp_output_dir):
    """Test that evaluation creates report.json file."""
    y_true = np.array([0, 1, 0, 1, 0, 1] * 5)
    results = [
        {
            "seed": 42,
            "y_pred": np.array([0, 1, 0, 0, 0, 1] * 5),
            "y_proba": None,
            "model_name": "model",
        }
    ]

    summary = evaluate_and_report(results, y_true, temp_output_dir)

    # summary is a list of metric dicts (flattened from all models)
    assert isinstance(summary, list)
    assert len(summary) > 0
    assert "accuracy" in summary[0]
    # Check that a report file was created (model-specific subfolder or combined)
    assert os.path.exists(
        os.path.join(temp_output_dir, "model", "report_model.json")
    ) or os.path.exists(os.path.join(temp_output_dir, "report_all_models.json"))

    # Check JSON is valid
    report_path = (
        os.path.join(temp_output_dir, "model", "report_model.json")
        if os.path.exists(os.path.join(temp_output_dir, "model", "report_model.json"))
        else os.path.join(temp_output_dir, "report_all_models.json")
    )
    with open(report_path, "r") as f:
        report_data = json.load(f)
        assert isinstance(report_data, list)
        assert len(report_data) > 0


def test_evaluate_and_report_multiple_seeds_computes_stats(temp_output_dir):
    """Test that multiple seeds produce mean/std statistics."""
    y_true = np.array([0, 1, 0, 1, 0, 1] * 5)

    results = []
    for seed in [0, 1, 2]:
        np.random.seed(seed)
        y_pred = np.where(np.random.rand(30) > 0.3, y_true, 1 - y_true)
        results.append(
            {"seed": seed, "y_pred": y_pred, "y_proba": None, "model_name": "model"}
        )

    summary = evaluate_and_report(results, y_true, temp_output_dir)

    # summary is a list of metric dicts, one per seed (flattened)
    assert isinstance(summary, list)
    assert len(summary) == 3
    assert all("accuracy" in m for m in summary)
    # Check that a report file was created in model-specific subfolder
    assert os.path.exists(
        os.path.join(temp_output_dir, "model", "report_model.json")
    ) or os.path.exists(os.path.join(temp_output_dir, "report_all_models.json"))


# Regression tests
def test_regression_metrics_computed_correctly():
    """Test that regression metrics are computed correctly."""
    from ecosci.eval import compute_regression_metrics

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "mape" in metrics

    # Basic sanity checks
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
    # R2 can be negative for bad predictions, but should be reasonable here
    assert -1 <= metrics["r2"] <= 1


def test_regression_metrics_perfect_prediction():
    """Test regression metrics for perfect predictions."""
    from ecosci.eval import compute_regression_metrics

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mape"] == pytest.approx(0.0)


def test_regression_metrics_with_zeros():
    """Test that MAPE is None when y_true contains zeros."""
    from ecosci.eval import compute_regression_metrics

    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.1, 1.1, 2.1, 3.1])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mape"] is None
    assert metrics["mse"] >= 0
    assert metrics["r2"] is not None


def test_evaluate_and_report_regression(temp_output_dir):
    """Test evaluation for regression problems."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 6)

    results = [
        {"seed": 42, "y_pred": y_true + np.random.randn(30) * 0.1, "y_proba": None}
    ]

    summary = evaluate_and_report(
        results, y_true, temp_output_dir, problem_type="regression"
    )

    assert isinstance(summary, list)
    assert len(summary) > 0
    assert "mse" in summary[0]
    assert "rmse" in summary[0]
    assert "mae" in summary[0]
    assert "r2" in summary[0]
    # Report is now saved in model-specific subfolder
    assert os.path.exists(os.path.join(temp_output_dir, "model", "report_model.json"))


def test_evaluate_and_report_regression_multiple_seeds(temp_output_dir):
    """Test evaluation for regression with multiple seeds."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 6)

    results = []
    for seed in [0, 1, 2]:
        np.random.seed(seed)
        y_pred = y_true + np.random.randn(30) * 0.2
        results.append({"seed": seed, "y_pred": y_pred, "y_proba": None})

    summary = evaluate_and_report(
        results, y_true, temp_output_dir, problem_type="regression"
    )

    assert isinstance(summary, list)
    assert len(summary) == 3
    assert all("mse" in m for m in summary)
    assert all("r2" in m for m in summary)
    # Report is now saved in model-specific subfolder
    assert os.path.exists(os.path.join(temp_output_dir, "model", "report_model.json"))


# Multi-output tests
def test_multi_output_classification_metrics():
    """Test multi-output classification metrics computation."""
    from ecosci.eval import compute_multi_output_classification_metrics
    
    # Multi-output: predicting 2 outputs
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 0]])
    
    metrics = compute_multi_output_classification_metrics(y_true, y_pred)
    
    assert "accuracy_mean" in metrics
    assert "balanced_accuracy_mean" in metrics
    assert "f1_mean" in metrics
    assert "n_outputs" in metrics
    assert metrics["n_outputs"] == 2
    assert "per_output" in metrics
    assert len(metrics["per_output"]) == 2


def test_multi_output_regression_metrics():
    """Test multi-output regression metrics computation."""
    from ecosci.eval import compute_multi_output_regression_metrics
    
    # Multi-output: predicting 3 outputs
    y_true = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    y_pred = np.array([[1.1, 2.1, 3.1], [2.1, 3.1, 4.1], [2.9, 3.9, 4.9], [4.1, 5.1, 6.1]])
    
    metrics = compute_multi_output_regression_metrics(y_true, y_pred)
    
    assert "mse_mean" in metrics
    assert "rmse_mean" in metrics
    assert "mae_mean" in metrics
    assert "r2_mean" in metrics
    assert "n_outputs" in metrics
    assert metrics["n_outputs"] == 3
    assert "per_output" in metrics
    assert len(metrics["per_output"]) == 3


def test_evaluate_and_report_multi_output_classification(temp_output_dir):
    """Test evaluation for multi-output classification."""
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0]] * 10)
    
    results = [
        {
            "seed": 42,
            "y_pred": np.array([[0, 1], [1, 0], [0, 0], [1, 1]] * 10),
            "y_proba": None,
            "model_name": "model",
        }
    ]
    
    summary = evaluate_and_report(results, y_true, temp_output_dir, problem_type="classification")
    
    assert isinstance(summary, list)
    assert len(summary) > 0
    assert "accuracy_mean" in summary[0]
    assert "n_outputs" in summary[0]


def test_evaluate_and_report_multi_output_regression(temp_output_dir):
    """Test evaluation for multi-output regression."""
    np.random.seed(42)
    y_true = np.random.randn(30, 3)  # 3 outputs
    
    results = [
        {
            "seed": 42,
            "y_pred": y_true + np.random.randn(30, 3) * 0.1,
            "y_proba": None,
            "model_name": "model",
        }
    ]
    
    summary = evaluate_and_report(results, y_true, temp_output_dir, problem_type="regression")
    
    assert isinstance(summary, list)
    assert len(summary) > 0
    assert "mse_mean" in summary[0]
    assert "n_outputs" in summary[0]
    assert summary[0]["n_outputs"] == 3
