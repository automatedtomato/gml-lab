import os
import time

import pytest
import torch

from src.gml_lab.modeling import FxWrapper, load_model
from src.gml_lab.quantizer import build_qconfig_mapping

from .utils import get_model_size, quantize

models = [
    "resnet18_8xb32_in1k",
]
seed = [int(os.getenv("SET_SEED", time.time_ns()))]

input_shape = [
    [1, 3, 224, 224],
    [1, 3, 130, 305],
]

BATCH = 8
TOTAL_CALIB_BATCHES = 10


@pytest.mark.parametrize("seed", seed)
@pytest.mark.parametrize("model_arch", models)
@pytest.mark.parametrize("input_shape", input_shape)
def test_quantize_model(seed: int, model_arch: str, input_shape: list[int]) -> None:
    torch.manual_seed(seed)

    model = load_model(arch=model_arch)
    float_model = FxWrapper(model)

    fp32_size = get_model_size(float_model)

    qconfig_mapping = build_qconfig_mapping("fbgemm")
    example_inputs = (torch.randn(input_shape),)

    dummy_loader = [torch.randn(BATCH, 3, 224, 224) for _ in range(100)]

    _, q_model = quantize(
        float_model,
        example_inputs,
        qconfig_mapping,
        dummy_loader,
        TOTAL_CALIB_BATCHES,
    )

    int8_size = get_model_size(q_model)
    print(f"FP32 Size: {fp32_size:.2f} MB")
    print(f"INT8 Size: {int8_size:.2f} MB")

    y = q_model(example_inputs[0])
    assert y.shape == (1, 1000)

    structure = q_model.code
    assert "quantized" in structure or "quantize_per_tensor" in structure
