import os
import time

import pytest
import torch
from torch.ao.quantization import observer

import src.gml_lab.nn.fused_modules as fcnn
import src.gml_lab.nn.modules as cnn
from tests.models import AddFunc, AddMethod, AddOp, AddReLU, IncrementalAdd
from tests.utils.node_info import NodeInfo
from tests.utils.test_utils import get_test_output_dir, run_quantizer_test

seeds = [int(os.getenv("SET_SEED", time.time_ns()))]
models = [AddFunc, AddMethod, AddOp, IncrementalAdd]


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
def test_unify_add(
    seed: int, model: torch.nn.Module, request: pytest.FixtureRequest
) -> None:
    device = "cpu"
    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)
    input_shape = (1, 3, 28, 28)

    model = model()
    example_inputs = (
        torch.randn(input_shape),
        torch.randn(input_shape),
    )

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(cnn.Add),  # type: ignore
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


@pytest.mark.parametrize("seed", seeds)
def test_fused_add_relu(seed: int, request: pytest.FixtureRequest) -> None:
    device = "cpu"
    torch.manual_seed(seed)
    out_dir = get_test_output_dir(request.node.name, __file__)
    input_shape = (1, 3, 28, 28)

    model = AddReLU()
    example_inputs = (
        torch.randn(input_shape),
        torch.randn(input_shape),
    )

    expected_nodes = [
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(observer.MinMaxObserver),  # type: ignore
        NodeInfo.call_module(fcnn.AddReLU),  # type: ignore
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
