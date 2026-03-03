import os
import time

import pytest
import torch
from torch.ao.quantization import observer

from tests.models import ConvIdentity, LinearIdentity
from tests.utils.node_info import NodeInfo
from tests.utils.test_utils import run_quantizer_test

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]


@pytest.mark.parametrize("seed", seeds)
def test_conv_identity_relu(seed: int) -> None:

    device = "cpu"
    torch.manual_seed(seed)

    example_inputs = (torch.randn((32, 16, 224, 224)),)
    model = ConvIdentity(16, 32, (3, 3))

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),
        NodeInfo.call_module(torch.nn.Conv2d),
        NodeInfo.call_module(observer.MinMaxObserver),
    ]

    run_quantizer_test(
        model,
        example_inputs,
        example_inputs,
        "unify_pass",
        expected_nodes=expected_nodes,
        device=device,
    )


@pytest.mark.parametrize("seed", seeds)
def test_linear_identity(seed: int) -> None:

    device = "cpu"
    torch.manual_seed(seed)

    example_inputs = (torch.randn((1, 256)),)
    model = LinearIdentity(256, 256, bias=False)

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),
        NodeInfo.call_module(torch.nn.Linear),
        NodeInfo.call_module(observer.MinMaxObserver),
    ]

    run_quantizer_test(
        model,
        example_inputs,
        example_inputs,
        "unify_pass",
        expected_nodes=expected_nodes,
        device=device,
    )
