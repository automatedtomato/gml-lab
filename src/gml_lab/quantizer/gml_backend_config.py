import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.reference as nnqr
import torch.ao.quantization.fuser_method_mappings as fuser_mapping
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

__all__ = ["get_gml_backend_config", "get_prepare_custom_config"]

different_observer = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT

default_int8_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)


def fuse_conv_relu(is_qat: bool, conv: nn.Module, relu: nn.Module) -> nn.Module:  # noqa: ARG001, FBT001
    """Fuse conv and relu: custom fuser mapping function."""
    return nni.ConvReLU2d(conv, relu)


def _get_default_configs() -> list[BackendPatternConfig]:
    default_ops = [
        nn.ReLU,
        cnn.Add,
        fcnn.AddReLU,
        nni.ConvReLU2d,
    ]
    default_configs: list[BackendPatternConfig] = []
    for op in default_ops:
        default_configs.append(  # noqa: PERF401
            BackendPatternConfig(op)
            .set_observation_type(different_observer)
            .set_dtype_configs([default_int8_config])
        )
    return default_configs


def _get_linear_configs() -> list[BackendPatternConfig]:
    linear_configs: list[BackendPatternConfig] = []

    linear_configs.append(
        BackendPatternConfig(nn.Linear)
        .set_observation_type(different_observer)
        .set_dtype_configs([default_int8_config])
        .set_root_module(nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    linear_configs.append(
        BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
        .set_observation_type(different_observer)
        .set_dtype_configs([default_int8_config])
        .set_root_module(nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
        .set_fuser_method(fuser_mapping.fuse_linear_bn)
    )
    return linear_configs


def _get_conv_configs() -> list[BackendPatternConfig]:
    conv_configs: list[BackendPatternConfig] = []

    conv_configs.append(
        BackendPatternConfig(nn.Conv2d)
        .set_observation_type(different_observer)
        .set_dtype_configs([default_int8_config])
        .set_root_module(nn.Conv2d)
        .set_reference_quantized_module(nnqr.Conv2d)
    )
    conv_configs.append(
        BackendPatternConfig((nn.Conv2d, nn.BatchNorm2d))
        .set_observation_type(different_observer)
        .set_dtype_configs([default_int8_config])
        .set_root_module(nn.Conv2d)
        .set_reference_quantized_module(nnqr.Conv2d)
        .set_fuser_method(fuser_mapping.fuse_conv_bn)
    )
    conv_configs.append(
        BackendPatternConfig((nn.Conv2d, nn.ReLU))
        .set_observation_type(different_observer)
        .set_dtype_configs([default_int8_config])
        .set_root_module(nni.ConvReLU2d)
        .set_reference_quantized_module(nni.ConvReLU2d)
        .set_fuser_method(fuse_conv_relu)
    )
    conv_configs.append(
        BackendPatternConfig((nn.Conv2d, nn.BatchNorm2d, nn.ReLU))
        .set_observation_type(different_observer)
        .set_dtype_configs([default_int8_config])
        .set_root_module(nni.ConvReLU2d)
        .set_reference_quantized_module(nni.ConvReLU2d)
        .set_fuser_method(fuser_mapping.fuse_conv_bn_relu)
    )
    return conv_configs


def get_gml_backend_config() -> BackendConfig:
    """Get GML Lab custom backend config."""
    return (
        BackendConfig("gml_lab")
        .set_backend_pattern_configs(_get_default_configs())
        .set_backend_pattern_configs(_get_linear_configs())
        .set_backend_pattern_configs(_get_conv_configs())
    )


def get_prepare_custom_config() -> PrepareCustomConfig:
    """Get PrepareCustomConfig.

    Some modules should not be traced during process in prepare_fx.
    Specify the modules in this function.
    """
    non_traceable_modules = [cnn.Add, fcnn.AddReLU]
    return PrepareCustomConfig().set_non_traceable_module_classes(non_traceable_modules)
