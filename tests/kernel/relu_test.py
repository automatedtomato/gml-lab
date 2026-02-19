from __future__ import annotations

import os
import time

import pytest
import torch

from tests.models import ReLUFunc1, ReLUFunc2, ReLUMethod, ReLUModule
from tests.utils.test_utils import (
    NO_GPU,
    SNR_THRESH,
    get_test_output_dir,
    run_quantizer_test,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    ReLUFunc1,
    ReLUFunc2,
    ReLUMethod,
    ReLUModule,
]

input_shapes = [
    [16],
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

    # expected_nodes = [
    #   hogehoge
    # ]

    test_inputs = (torch.randn(input_shape),)
    model = model().to(device)
    snr = run_quantizer_test(
        model,
        test_inputs,
        out_dir,
        # expected_nodes,
    )

    assert snr > SNR_THRESH, f"{snr=} < {SNR_THRESH}"
