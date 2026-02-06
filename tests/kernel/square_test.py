from __future__ import annotations

import os
import time

import pytest
import torch

from tests.utils.test_utils import NO_GPU

import gml_lab_custom_ops as custom_ops

seeds = [int(os.getenv("SEED", time.time_ns()))]

input_shapes = [
    [3, 128],
    [1, 32, 256],
    [1, 45, 130],
    [1, 3, 256, 512],
]

dtypes = [
    torch.float32,
]


@pytest.mark.gpu
@pytest.mark.skipif(NO_GPU, reason="GPU not available")
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("dtype", dtypes)
def test_square(
    seed: int,
    input_shape: list[int],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(seed)
    test_input = torch.randn(input_shape, dtype=dtype, device="cuda")

    out_ref = torch.square(test_input)
    out_target = custom_ops.square(test_input)

    assert torch.allclose(out_ref, out_target)
