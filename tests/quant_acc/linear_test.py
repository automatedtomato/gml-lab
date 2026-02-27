from __future__ import annotations

import os
import time

import pytest
import torch

from tests.models import LinearBN, LinearModule
from tests.utils.test_utils import (
    SNR_THRESH,
    get_test_output_dir,
    run_quantizer_test,
)

device = "cpu"

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

models = [
    LinearBN,
    LinearModule,
]

in_features = [256, 1024]
out_features = [700, 512]
bias = [True, False]


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

    test_inputs = (torch.randn((1, in_features)),)
    model = model(in_features=in_features, out_features=out_features, bias=bias).to(
        device
    )
    snr = run_quantizer_test(
        model, test_inputs, test_inputs, "quant_acc", out_dir, device=device
    )

    assert snr > SNR_THRESH, f"{snr=} < {SNR_THRESH}"  # type: ignore
