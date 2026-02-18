import os
import time

import pytest
import torch

from tests.quantizer.utils import quantize_model
from tools.profiler import FxProfiler


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.linear = torch.nn.Linear(8 * 28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, self.linear.in_features)
        return self.linear(x)


model = SimpleModel()
float_model = torch.fx.symbolic_trace(model)
_, qdq_model = quantize_model(model)


seeds = [int(os.getenv("SET_SEED", time.time()))]
models = [float_model, qdq_model]


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("model", models)
def test_profiler_simple_model(seed: int, model: torch.fx.GraphModule) -> None:
    torch.manual_seed(seed)
    device = "cpu"

    profiler = FxProfiler(model)
    example_input = torch.randn(1, 3, 28, 28, dtype=torch.float32).to(device)

    profiler.run(example_input)

    results = profiler.results

    profiler.print_summary()
    profiler.dump_to_json(f"results/{type(model).__name__}.json")

    assert len(results) > 0
