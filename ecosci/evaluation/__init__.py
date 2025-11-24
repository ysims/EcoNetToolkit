"""Evaluation module for EcoNetToolkit.

This module provides comprehensive evaluation and reporting capabilities
for machine learning models, including:
- Metrics computation for classification and regression
- Feature importance analysis
- Visualisation and plotting
- Cross-validation results reporting
- Hyperparameter tuning evaluation

The module is organised into submodules for better maintainability:
- metrics: Core metric computation functions
- plotting: Visualisation utilities
- feature_importance: Feature importance extraction and analysis
- reporting: Main evaluation and reporting orchestration

For backward compatibility, key functions are re-exported from the main module.
"""

# Re-export key functions for backward compatibility
from .metrics import (
    safe_std,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_multi_output_classification_metrics,
    compute_multi_output_regression_metrics,
)

from .reporting import (
    evaluate_and_report,
    evaluate_and_report_cv,
    evaluate_tuning_results,
)

# Make submodules available
from . import metrics
from . import plotting
from . import feature_importance
from . import reporting

__all__ = [
    # Core functions
    'evaluate_and_report',
    'evaluate_and_report_cv',
    'evaluate_tuning_results',
    
    # Metric functions
    'safe_std',
    'compute_classification_metrics',
    'compute_regression_metrics',
    'compute_multi_output_classification_metrics',
    'compute_multi_output_regression_metrics',
    
    # Submodules
    'metrics',
    'plotting',
    'feature_importance',
    'reporting',
]
