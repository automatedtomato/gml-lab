from __future__ import annotations

import itertools
import os
import time
from typing import Any

import pytest
import torch

from src.gml_lab.kernel import GMLQuantConv, GMLQuantConvReLU
from tests.models import ConvBN, ConvBNReLUFunc, ConvModule, ConvReLUFunc
from tests.utils.node_info import NodeInfo
from tests.utils.test_utils import (
    NO_GPU,
    SNR_THRESH_NONLINEAR,
    get_test_output_dir,
    run_quantizer_test,
)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    ConvModule,
    ConvBN,
]

INPUT_SHAPE = [224, 224]
kernel_shapes = [
    (3, 1, 1, 1),
    # (2, 3, 1, 3),
    # (1, 2, 3, 1),
]
bias = [True, False]
stride = [
    [1, 1],
    # [1, 2],
    # [2, 1],
    # [2, 2]
]
padding = [
    [0, 0],
    # [1, 1],
    # [0, 1]
]
groups = [1, 2]

combi = itertools.product(bias, stride, padding, groups)
conv_params = [
    {"bias": b, "stride": st, "padding": p, "dilation": 1, "groups": g}
    for b, st, p, g in combi
]


@pytest.mark.skipif(NO_GPU, reason="GPU not available")
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("kernel_shape", kernel_shapes)
@pytest.mark.parametrize("conv_param", conv_params)
def test_conv_bn(
    seed: int,
    model: torch.nn.Module,
    kernel_shape: list[int],
    conv_param: dict[str, Any],
    request: pytest.FixtureRequest,
) -> None:

    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    expected_nodes = [
        NodeInfo.call_function(torch.quantize_per_tensor),
        NodeInfo.call_module(GMLQuantConv),
        NodeInfo.call_method("dequantize"),
    ]

    i, o, h, w = kernel_shape
    in_channels = i * conv_param["groups"]
    out_channels = o * conv_param["groups"]
    test_input_shape = [1, in_channels, *INPUT_SHAPE]
    example_inputs = (torch.randn(test_input_shape),)

    model = model(in_channels, out_channels, (h, w), **conv_param)

    snr = run_quantizer_test(
        model,
        example_inputs,
        example_inputs,
        test_mode="lower_acc",
        out_dir=out_dir,
        device=device,
        expected_nodes=expected_nodes,
    )

    assert snr > SNR_THRESH_NONLINEAR, f"{snr=} < {SNR_THRESH_NONLINEAR}"


models = [
    ConvReLUFunc,
    ConvBNReLUFunc,
]


@pytest.mark.skipif(NO_GPU, reason="GPU not available")
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("kernel_shape", kernel_shapes)
@pytest.mark.parametrize("conv_param", conv_params)
def test_conv_relu(
    seed: int,
    model: torch.nn.Module,
    kernel_shape: list[int],
    conv_param: dict[str, Any],
    request: pytest.FixtureRequest,
) -> None:

    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    expected_nodes = [
        NodeInfo.call_function(torch.quantize_per_tensor),
        NodeInfo.call_module(GMLQuantConvReLU),
        NodeInfo.call_method("dequantize"),
    ]

    i, o, h, w = kernel_shape
    in_channels = i * conv_param["groups"]
    out_channels = o * conv_param["groups"]
    test_input_shape = [1, in_channels, *INPUT_SHAPE]
    example_inputs = (torch.randn(test_input_shape),)

    model = model(in_channels, out_channels, (h, w), **conv_param)

    snr = run_quantizer_test(
        model.to(device),
        example_inputs,
        example_inputs,
        test_mode="lower_acc",
        out_dir=out_dir,
        device=device,
        expected_nodes=expected_nodes,
    )

    assert snr > SNR_THRESH_NONLINEAR, f"{snr=} < {SNR_THRESH_NONLINEAR}"
