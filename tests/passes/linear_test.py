import os
import time

import pytest
import torch
from torch.ao.quantization import observer

from tests.models import LinearBN, LinearFunc, LinearModule
from tests.utils.node_info import NodeInfo
from tests.utils.test_utils import get_test_output_dir, run_quantizer_test

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]
models = [LinearModule, LinearFunc, LinearBN]

input_shapes = [
    (1, 256),
    (1, 224, 432),
]
bias = [True, False]


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("bias", bias)
@pytest.mark.parametrize("input_shape", input_shapes)
def test_unify_linear(
    seed: int,
    model: torch.nn.Module,
    bias: bool,  # noqa: FBT001
    input_shape: tuple[int, ...],
    request: pytest.FixtureRequest,
) -> None:
    device = "cpu"
    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    model = model(in_features=input_shape[-1], out_features=input_shape[1], bias=bias)
    example_inputs = (torch.randn(input_shape),)

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(torch.nn.Linear),  # type: ignore
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
    ]

    run_quantizer_test(
        model,
        example_inputs,
        example_inputs,
        test_mode="unify_pass",
        out_dir=out_dir,
        expected_nodes=expected_nodes,
        device=device,
    )
