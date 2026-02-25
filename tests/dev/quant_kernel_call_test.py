from __future__ import annotations

import pytest
import torch

from tests.models import ReLUFunc1, ReLUFunc2, ReLUMethod, ReLUModule
from tests.utils.test_utils import INT8_MAX, INT8_MIN

try:
    from src.gml_lab.kernel_class import GMLQuantReLU
except ImportError as e:
    print(f"Error importing GMLQuantReLU: {e}")
    GMLQuantReLU = None


@pytest.mark.skip
def test_gml_kernel_init() -> None:
    if GMLQuantReLU is None:
        pytest.skip("GMLQuantReLU kernel is not implemented.")
    test_scale = 5.27
    test_zp = 24

    module = GMLQuantReLU(scale=test_scale, zero_point=test_zp)
    assert hasattr(module, "scale"), "Module must have `scale` attr."
    assert hasattr(module, "zero_point"), "Module must have `zero_point` attr"
    assert isinstance(module.scale, torch.Tensor)
    assert isinstance(module.zero_point, torch.Tensor)
    assert torch.isclose(module.scale, torch.tensor(test_scale))
    assert module.zero_point.item() == test_zp


@pytest.mark.skip
def test_gml_kernel_forward() -> None:
    if GMLQuantReLU is None:
        pytest.skip("GMLQuantReLU kernel is not implemented.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    module = GMLQuantReLU(scale=0.1, zero_point=0).to(device)

    input_int = torch.randint(
        INT8_MIN, INT8_MAX, (1, 3, 28, 28), dtype=torch.int8, device=device
    )

    try:
        out = module(input_int)
    except Exception as e:
        print(f"Forward pass failed: {e}")

    assert out.dtype == torch.int8, f"Output dtype must be int8, got {out.dtype}"
    assert out.shape == input_int.shape


models = [
    ReLUFunc1,
    ReLUFunc2,
    ReLUMethod,
    ReLUModule,
]
