from __future__ import annotations

import os
import time

import pytest
import torch

from src.gml_lab import kernel_class
from tests.models import LinearBN, LinearModule
from tests.utils.test_utils import (
    NO_GPU,
    SNR_THRESH,
    NodeInfo,
    get_test_output_dir,
    run_quantizer_test,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    LinearModule,
    LinearBN,
]

in_features = [256, 1024]
out_features = [700, 512]
bias = [True, False]


@pytest.mark.skipif(NO_GPU, reason="GPU not available")
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("bias", bias)
@pytest.mark.parametrize("in_features", in_features)
@pytest.mark.parametrize("out_features", out_features)
def test_linear(
    seed: int,
    model: torch.nn.Module,
    in_features: int,
    out_features: int,
    bias: bool,  # noqa: FBT001
    request: pytest.FixtureRequest,
) -> None:

    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    expected_nodes = [
        NodeInfo.call_function(torch.quantize_per_tensor),
        NodeInfo.call_module(kernel_class.GMLQuantFullyConnected),  # type: ignore
        NodeInfo.call_method("dequantize"),
    ]

    test_inputs = (torch.randn((1, in_features)),)
    test_inputs = tuple(i.to(device) for i in test_inputs)
    model = model(in_features=in_features, out_features=out_features, bias=bias).to(
        device
    )
    snr = run_quantizer_test(
        float_model=model,
        example_inputs=test_inputs,
        calib_inputs=test_inputs,
        test_mode="lower_acc",
        out_dir=out_dir,
        expected_nodes=expected_nodes,
        device=device,
    )

    assert snr > SNR_THRESH, f"{snr=} < {SNR_THRESH}"  # type: ignore
