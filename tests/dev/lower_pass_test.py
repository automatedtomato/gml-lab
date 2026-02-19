from __future__ import annotations

import pytest
import torch

from src.gml_lab.lowering.passes.lower_relu import lower_relu
from tests.models import ReLUFunc1, ReLUFunc2, ReLUMethod, ReLUModule
from tests.utils.test_utils import quantize_model

models = [ReLUFunc1, ReLUFunc2, ReLUMethod, ReLUModule]


@pytest.mark.parametrize("model", models)
def test_lower_pass(model: torch.nn.Module) -> None:
    model = model()
    example_inputs = (torch.randn((1, 3, 224, 224)),)
    _, qdq_model = quantize_model(model, example_inputs)
    lower_relu(qdq_model)
    graph = qdq_model.graph

    result_nodes = [node.name for node in graph.nodes]

    assert "cuda_relu" in result_nodes
