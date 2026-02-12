import os
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml


def set_seed() -> int:
    """Set seed for torch."""
    seed = int(os.getenv("SET_SEED", time.time_ns()))
    torch.manual_seed(seed)
    return seed


def load_config(config_path: Path, top_key: str = "example_config") -> dict:
    """Load YAML and generate configuration.

    Args:
        config_path: YAML file path.
        top_key: Top key for the configuration. Default to "example_config"
            when not specified.

    Returns:
        Configuration dictionary

    """
    with open(config_path, encoding="utf-8") as f:
        config: Any = yaml.safe_load(f)

    if config is None or config == {}:
        warnings.warn("configuration is empty", stacklevel=1)

    if top_key not in config:
        msg = f"Invalid config. top key `{top_key}` was not found in {config_path}"
        raise KeyError(msg)

    return config
