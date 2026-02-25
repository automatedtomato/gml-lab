from __future__ import annotations

import os
import time

import pytest
import torch

from src.gml_lab.kernel_class import GMLQuantAdd, GMLQuantAddReLU
from tests.models import AddFunc, AddReLU, IncrementalAdd
from tests.utils.test_utils import (
    NO_GPU,
    SNR_THRESH_NONLINEAR,
    NodeInfo,
    get_test_output_dir,
    run_quantizer_test,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    AddFunc,
    AddReLU,
    IncrementalAdd,
]

input_shapes = [
    [16, 64, 256],
    [1, 3, 224, 224],
]


@pytest.mark.skipif(NO_GPU, reason="GPU not available")
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("input_shape", input_shapes)
def test_relu(
    seed: int,
    model: torch.nn.Module,
    input_shape: list[int],
    request: pytest.FixtureRequest,
) -> None:

    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    expected_nodes = [
        NodeInfo.call_function(torch.quantize_per_tensor),
        NodeInfo.call_function(torch.quantize_per_tensor),
        NodeInfo.call_module(GMLQuantAddReLU if model == AddReLU else GMLQuantAdd),  # type: ignore
        NodeInfo.call_method("dequantize"),
    ]

    example_inputs = (torch.randn(input_shape), torch.randn(input_shape) * 2 - 1)
    model = model()
    snr = run_quantizer_test(
        float_model=model,
        example_inputs=example_inputs,
        calib_inputs=example_inputs,
        test_mode="lower_acc",
        out_dir=out_dir,
        expected_nodes=expected_nodes,
        device=device,
    )

    assert snr > SNR_THRESH_NONLINEAR, f"{snr=} < {SNR_THRESH_NONLINEAR}"  # type: ignore
