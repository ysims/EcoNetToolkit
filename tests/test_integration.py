"""Integration tests for the full pipeline."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
import json

from ecosci.config import load_config
from ecosci.data import CSVDataLoader
from ecosci.models import ModelZoo
from ecosci.trainer import Trainer
from ecosci.eval import evaluate_and_report


@pytest.fixture
def sample_csv():
    """Create a temporary CSV file."""
    np.random.seed(42)
    n = 50
    data = {
        "num1": np.random.randn(n),
        "num2": np.random.randn(n) + 1,
        "label": np.random.choice([0, 1], n),
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        return f.name


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_config(sample_csv, temp_output_dir):
    """Create a temporary config file."""
    config = f"""
problem_type: classification

data:
  path: {sample_csv}
  features: [num1, num2]
  label: label
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  scaling: standard

models:
  - name: logistic
    params:
      random_state: 42

training:
  repetitions: 2
  random_seed: 0

output_dir: {temp_output_dir}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config)
        return f.name


def test_full_pipeline_from_config_to_report(temp_config):
    """Test complete pipeline: load config → load data → train → evaluate."""
    cfg = load_config(temp_config)

    loader = CSVDataLoader(
        path=cfg["data"]["path"],
        features=cfg["data"]["features"],
        label=cfg["data"]["label"],
        test_size=cfg["data"]["test_split"],
        val_size=cfg["data"]["val_split"],
        scaling=cfg["data"]["scaling"],
    )
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()

    trainer = Trainer(
        ModelZoo.get_model,
        problem_type=cfg["problem_type"],
        output_dir=cfg["output_dir"],
    )
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)

    summary = evaluate_and_report(results, y_test, cfg["output_dir"])

    # Check that we got results
    assert len(results['logistic']) == cfg['training']['repetitions']
    assert isinstance(summary, list)
    assert len(summary) == cfg['training']['repetitions']
    # Check that at least one report file exists
    assert (
        os.path.exists(os.path.join(cfg['output_dir'], 'report_logistic.json')) or
        os.path.exists(os.path.join(cfg['output_dir'], 'report_all_models.json'))
    )
    
    # Verify report is valid
    report_path = (
        os.path.join(cfg['output_dir'], 'report_logistic.json')
        if os.path.exists(os.path.join(cfg['output_dir'], 'report_logistic.json'))
        else os.path.join(cfg['output_dir'], 'report_all_models.json')
    )
    with open(report_path, 'r') as f:
        report = json.load(f)
        assert isinstance(report, list)
        assert len(report) > 0


def test_pipeline_with_categorical_features(temp_output_dir):
    """Test pipeline with categorical data."""
    np.random.seed(42)
    n = 50
    data = {
        "num1": np.random.randn(n),
        "cat": np.random.choice(["A", "B", "C"], n),
        "label": np.random.choice([0, 1], n),
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_output_dir, "cat_data.csv")
    df.to_csv(csv_path, index=False)

    # CSVDataLoader auto-detects categorical columns (non-numeric)
    loader = CSVDataLoader(path=csv_path, features=["num1", "cat"], label="label")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare()

    cfg = {
        'problem_type': 'classification',
        'models': [{'name': 'random_forest', 'params': {'n_estimators': 10, 'random_state': 42}}],
        'training': {'repetitions': 1, 'random_seed': 0},
        'output_dir': temp_output_dir
    }

    trainer = Trainer(
        ModelZoo.get_model, problem_type="classification", output_dir=temp_output_dir
    )
    results = trainer.run(cfg, X_train, X_val, X_test, y_train, y_val, y_test)
    
    assert len(results['random_forest']) == 1
    assert len(results['random_forest'][0]['y_pred']) == len(y_test)
