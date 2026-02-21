import torch
import torch.nn.intrinsic as nni
from torch import nn
from torch.ao.quantization import fuser_method_mappings as fuser_mappings
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)

from .backend_mapping import CONV_BN_MAP, CONV_BN_RELU_MAP, CONV_RELU_MAP

default_int8_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

__all__ = ["get_gml_backend_config"]


def _get_default_backend_configs() -> list[BackendPatternConfig]:
    default_ops = [
        # ReLU
        nn.ReLU,
        nn.functional.relu,
        torch.relu,
        "relu",
        nn.Conv2d,
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


def _get_fused_conv_backend_configs() -> list[BackendPatternConfig]:
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    fused_conv_configs: list[BackendPatternConfig] = []
    # Conv + BN + ReLU
    for m in CONV_BN_RELU_MAP:
        fused_conv_configs.append(  # noqa: PERF401
            BackendPatternConfig(m)
            .set_observation_type(observation_type)
            .set_dtype_configs([default_int8_config])
            .set_fuser_method(fuser_mappings.fuse_conv_bn_relu)
            .set_fused_module(nni.ConvBnReLU2d)
        )
    # Conv + BN
    for m in CONV_BN_MAP:
        fused_conv_configs.append(  # noqa: PERF401
            BackendPatternConfig(m)
            .set_observation_type(observation_type)
            .set_dtype_configs([default_int8_config])
            .set_fuser_method(fuser_mappings.fuse_conv_bn)
            # .set_fused_module(nni.ConvBn2d)
        )
    # Conv + ReLU
    for m in CONV_RELU_MAP:
        fused_conv_configs.append(
            BackendPatternConfig(m)
            .set_observation_type(observation_type)
            .set_dtype_configs([default_int8_config])
            # .fuser_method(fuser_mappings.fuse_conv_relu)
            .set_fused_module(nni.ConvReLU2d)
        )
    return fused_conv_configs


def get_gml_backend_config() -> BackendConfig:
    """Get GML Lab custom backend config."""
    return (
        BackendConfig("gml_lab")
        .set_backend_pattern_configs(_get_fused_conv_backend_configs())
        .set_backend_pattern_configs(_get_default_backend_configs())
    )
