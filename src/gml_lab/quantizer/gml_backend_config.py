import torch
from torch import nn
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)

default_int8_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

__all__ = ["get_gml_backend_config"]


def _get_default_backend_configs() -> list[BackendPatternConfig]:
    default_ops = [
        nn.ReLU,
    ]
    default_configs: list[BackendPatternConfig] = []
    for op in default_ops:
        default_configs.append(  # noqa: PERF401
            BackendPatternConfig(op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs([default_int8_config])
        )
    return default_configs


def get_gml_backend_config() -> BackendConfig:
    """Get GML Lab custom backend config."""
    return BackendConfig("gml_lab").set_backend_pattern_configs(
        _get_default_backend_configs()
    )
