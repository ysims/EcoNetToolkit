"""Configuration helpers for EcoNetToolkit.

This module intentionally keeps things simple. Most users will only edit the
YAML file in `configs/` and never touch Python code. We just read that file
and hand a dictionary to the rest of the pipeline.

If a value is missing in the YAML, other modules provide sensible defaults.
"""

import yaml
from dataclasses import dataclass
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Parameters
    ----------
    path : str
        Path to a .yaml or .yml file.

    Returns
    -------
    Dict[str, Any]
        A nested dictionary of configuration values.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Ensure models key exists
    if "models" not in cfg:
        raise ValueError("Config must contain 'models' key with a list of model configurations")
    
    # Ensure models is a list
    if not isinstance(cfg["models"], list):
        cfg["models"] = [cfg["models"]]
    
    return cfg


@dataclass
class SimpleConfig:
    raw: Dict[str, Any]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)
