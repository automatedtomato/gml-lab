from __future__ import annotations

import os
import time

import pytest
import torch

from tests.models import LinearBN, LinearModule
from tests.utils.test_utils import (
    SNR_THRESH_NONLINEAR,
    get_test_output_dir,
    run_quantizer_test,
)

device = "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    LinearBN,
    LinearModule,
]

input_shapes = [
    [16, 256],
    [1, 32, 224],
]

bias = [True, False]


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("bias", bias)
def test_relu(
    seed: int,
    model: torch.nn.Module,
    input_shape: list[int],
    bias: bool,  # noqa: FBT001
    request: pytest.FixtureRequest,
) -> None:

    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    test_inputs = (torch.rand(input_shape),)
    model = model(
        in_features=input_shape[-1], out_features=input_shape[1], bias=bias
    ).to(device)
    snr = run_quantizer_test(
        model, test_inputs, test_inputs, "quant_acc", out_dir, device=device
    )

    assert snr > SNR_THRESH_NONLINEAR, f"{snr=} < {SNR_THRESH_NONLINEAR}"  # type: ignore
