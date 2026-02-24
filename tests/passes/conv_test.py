from __future__ import annotations

import os
import time

import pytest
import torch
import torch.nn.intrinsic as nni
from torch.ao.quantization import observer

from tests.models import (
    ConvBN,
    ConvBNReLUFunc,
    ConvReLUFunc,
)
from tests.utils.node_info import NodeInfo
from tests.utils.test_utils import get_test_output_dir, run_quantizer_test

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]


@pytest.mark.parametrize("seed", seeds)
def test_unify_conv_bn(seed: int, request: pytest.FixtureRequest) -> None:
    device = "cpu"
    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    model = ConvBN(3, 1, kernel_size=(3, 3))
    example_inputs = (torch.randn((1, 3, 28, 28)),)

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(torch.nn.Conv2d),  # type: ignore
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


models = [
    ConvBNReLUFunc,
    ConvReLUFunc,
]


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
def test_unify_conv_relu(
    seed: int, model: torch.nn.Module, request: pytest.FixtureRequest
) -> None:
    device = "cpu"
    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)

    model = model(3, 1, kernel_size=(3, 3))
    example_inputs = (torch.randn((1, 3, 28, 28)),)

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(nni.ConvReLU2d),  # type: ignore
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
