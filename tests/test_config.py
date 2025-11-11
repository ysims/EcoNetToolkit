"""Tests for configuration loading."""

import pytest
import tempfile
from ecosci.config import load_config


@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    config_content = """
problem_type: classification

data:
  path: data/sample.csv
  features: [feature1, feature2]
  label: label
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  scaling: standard

models:
  - name: mlp
    params:
      hidden_layer_sizes: [64]
      max_iter: 100

training:
  repetitions: 3
  random_seed: 0

output_dir: outputs
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        return f.name


def test_load_config_returns_dict(temp_config_file):
    """Test that config loading returns a dictionary."""
    cfg = load_config(temp_config_file)

    assert cfg is not None
    assert isinstance(cfg, dict)
    assert "problem_type" in cfg
    assert "data" in cfg
    assert "models" in cfg
    assert "training" in cfg


def test_config_has_expected_structure(temp_config_file):
    """Test that loaded config has expected nested structure."""
    cfg = load_config(temp_config_file)

    assert cfg["problem_type"] == "classification"
    assert "path" in cfg["data"]
    assert "features" in cfg["data"]
    assert "label" in cfg["data"]
    assert isinstance(cfg["data"]["features"], list)
    assert isinstance(cfg["models"], list)
    assert "name" in cfg["models"][0]
    assert "params" in cfg["models"][0]
    assert isinstance(cfg["models"][0]["params"], dict)

    assert cfg["problem_type"] == "classification"
    assert "path" in cfg["data"]
    assert "features" in cfg["data"]
    assert "label" in cfg["data"]
    assert isinstance(cfg["data"]["features"], list)
    assert isinstance(cfg["models"], list)
    assert "name" in cfg["models"][0]
    assert "params" in cfg["models"][0]
    assert isinstance(cfg["models"][0]["params"], dict)
