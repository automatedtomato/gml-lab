from __future__ import annotations

import itertools
import os
import time
from typing import Any

import pytest
import torch

from tests.models import ConvBN, ConvReLUMod, ConvReLUFunc, ConvBNReLUFunc
from tests.utils.test_utils import (
    NO_GPU,
    SNR_THRESH,
    get_test_output_dir,
    run_quantizer_test,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    # ConvModule,
    # ConvReLUMod,
    # ConvReLUFunc,
    ConvBNReLUFunc
    # ConvBN,
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
def test_relu(
    seed: int,
    model: torch.nn.Module,
    kernel_shape: list[int],
    conv_param: dict[str, Any],
    request: pytest.FixtureRequest,
) -> None:

    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    # expected_nodes = [
    #     NodeInfo.call_function(torch.quantize_per_tensor),
    #     NodeInfo.call_module(GMLQuantConv),
    #     NodeInfo.call_method("dequantize"),
    # ]

    i, o, h, w = kernel_shape
    in_channels = i * conv_param["groups"]
    out_channels = o * conv_param["groups"]
    test_input_shape = [1, in_channels, *INPUT_SHAPE]
    example_inputs = (torch.randn(test_input_shape),)

    model = model(in_channels, out_channels, (h, w), **conv_param)
    from tests.utils.test_utils import quantize_model
    model.eval()
    prepared_model, qdq_model = quantize_model(model, example_inputs, device=device)
    # prepared_model.graph.print_tabular()

    print("prepared model:")
    # prepared_model.graph.print_tabular()
    print(prepared_model.code)
    print("QDQ model:")
    print(qdq_model.code)
    # qdq_model.graph.print_tabular()

    snr = run_quantizer_test(
        model,
        example_inputs,
        out_dir,
        device,
        # expected_nodes,
    )

    assert snr > SNR_THRESH, f"{snr=} < {SNR_THRESH}"
