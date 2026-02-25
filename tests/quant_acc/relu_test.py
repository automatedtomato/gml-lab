from __future__ import annotations

import os
import time

import pytest
import torch

from tests.models import ReLUFunc1, ReLUMethod, ReLUModule
from tests.utils.test_utils import (
    SNR_THRESH_NONLINEAR,
    get_test_output_dir,
    run_quantizer_test,
)

device = "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    ReLUFunc1,
    ReLUMethod,
    ReLUModule,
]

input_shapes = [
    [16, 64, 256],
    [1, 3, 224, 224],
]


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

    test_inputs = (torch.rand(input_shape),)
    model = model().to(device)
    snr = run_quantizer_test(
        model, test_inputs, test_inputs, "quant_acc", out_dir, device=device
    )

    assert snr > SNR_THRESH_NONLINEAR, f"{snr=} < {SNR_THRESH_NONLINEAR}"  # type: ignore
