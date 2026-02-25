import os
import time

import pytest
import torch
from torch.ao.quantization import observer

from tests.models import ReLUFunc1, ReLUMethod, ReLUModule
from tests.utils.node_info import NodeInfo
from tests.utils.test_utils import get_test_output_dir, run_quantizer_test

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]
models = [ReLUFunc1, ReLUMethod, ReLUModule]


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
def test_unify_relu(
    seed: int, model: torch.nn.Module, request: pytest.FixtureRequest
) -> None:
    device = "cpu"
    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    model = model()
    example_inputs = (torch.randn((1, 3, 28, 28)),)

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(torch.nn.ReLU),  # type: ignore
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
