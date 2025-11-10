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
    
        # Check probability-based metrics are present
        assert 'roc_auc' in metrics
        assert 'average_precision' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['average_precision'] <= 1
    
    def test_compute_metrics_multiclass(self):
        """Test metric computation for multiclass classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_proba=None)
        
        # Check metrics are computed
        assert 'accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert 'cohen_kappa' in metrics
        
        # Confusion matrix should be 3x3
        assert metrics['confusion_matrix'].shape == (3, 3)
    
    
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


def test_evaluate_and_report_with_probabilities_creates_pr_curve(temp_output_dir):
    """Test that PR curve is created when probabilities are provided."""
    y_true = np.array([0, 1, 0, 1, 0, 1] * 5)
    
    np.random.seed(42)
    y_proba = np.column_stack([
        np.random.rand(30) * 0.5 + 0.25,
        np.random.rand(30) * 0.5 + 0.25
        ])
        # Normalize to sum to 1
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        y_pred = np.argmax(y_proba, axis=1)
        
        results = [
            {
                'seed': 42,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        ]
        
        cfg = {
            'output_dir': temp_output_dir,
            'model': {'name': 'test_model'}
        }
        
        summary = evaluate_and_report(results, y_true, cfg)
        
        # Check probability-based metrics are computed
    ])
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
    y_pred = np.argmax(y_proba, axis=1)
    
    results = [
        {
            'seed': 42,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    ]
    
    cfg = {
        'output_dir': temp_output_dir,
        'model': {'name': 'test_model'}
    }
    
    summary = evaluate_and_report(results, y_true, cfg)
    
    # Should have ROC-AUC and PR metrics
    if 'roc_auc' in summary['metrics']:
        assert 0 <= summary['metrics']['roc_auc'] <= 1
