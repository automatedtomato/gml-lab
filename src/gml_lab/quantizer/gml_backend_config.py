import torch
from torch import nn
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)
from torch.ao.quantization.fx.custom_config import (
    PrepareCustomConfig,
)

import src.gml_lab.nn.fused_modules as fcnn
import src.gml_lab.nn.modules as cnn

default_int8_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

__all__ = ["get_gml_backend_config", "get_prepare_custom_config"]


def _get_default_backend_configs() -> list[BackendPatternConfig]:
    default_ops = [
        nn.ReLU,
        cnn.Add,
        fcnn.AddReLU,
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


def get_prepare_custom_config() -> PrepareCustomConfig:
    """Get PrepareCustomConfig.

    Some modules should not be traced during process in prepare_fx.
    Specify the modules in this function.
    """
    non_traceable_modules = [cnn.Add, fcnn.AddReLU]
    return PrepareCustomConfig().set_non_traceable_module_classes(non_traceable_modules)
