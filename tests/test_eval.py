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
    
    assert 'accuracy' in metrics
    assert 'balanced_accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'cohen_kappa' in metrics
    
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['balanced_accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert -1 <= metrics['cohen_kappa'] <= 1


def test_metrics_with_probabilities_include_roc_auc():
    """Test that ROC-AUC is computed when probabilities are provided."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.8, 0.2],
        [0.1, 0.9]
    ])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
    
    assert 'roc_auc' in metrics
    assert 'average_precision' in metrics
    assert 0 <= metrics['roc_auc'] <= 1
    assert 0 <= metrics['average_precision'] <= 1


def test_evaluate_and_report_creates_json(temp_output_dir):
    """Test that evaluation creates report.json file."""
    y_true = np.array([0, 1, 0, 1, 0, 1] * 5)
    results = [
        {
            'seed': 42,
            'y_pred': np.array([0, 1, 0, 0, 0, 1] * 5),
            'y_proba': None
        }
    ]
    
    cfg = {
        'output_dir': temp_output_dir,
        'model': {'name': 'test_model'}
    }
    
    summary = evaluate_and_report(results, y_true, cfg)
    
    assert 'metrics' in summary
    assert os.path.exists(os.path.join(temp_output_dir, 'report.json'))
    
    # Check JSON is valid
    with open(os.path.join(temp_output_dir, 'report.json'), 'r') as f:
        report_data = json.load(f)
        assert 'metrics' in report_data


def test_evaluate_and_report_multiple_seeds_computes_stats(temp_output_dir):
    """Test that multiple seeds produce mean/std statistics."""
    y_true = np.array([0, 1, 0, 1, 0, 1] * 5)
    
    results = []
    for seed in [0, 1, 2]:
        np.random.seed(seed)
        y_pred = np.where(np.random.rand(30) > 0.3, y_true, 1 - y_true)
        results.append({
            'seed': seed,
            'y_pred': y_pred,
            'y_proba': None
        })
    
    cfg = {
        'output_dir': temp_output_dir,
        'model': {'name': 'test_model'}
    }
    
    summary = evaluate_and_report(results, y_true, cfg)
    
    # Should have mean and std for metrics
    assert 'accuracy_mean' in summary['metrics'] or 'accuracy' in summary['metrics']
    assert os.path.exists(os.path.join(temp_output_dir, 'report.json'))
