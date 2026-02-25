import os
import time

import pytest
import torch

from tests.utils.test_utils import SNR_THRESH, calc_blob_snr

try:
    import gml_lab_custom_ops as custom_ops
except ImportError:
    custom_ops = None


seeds = [int(os.getenv("SET_SEED", time.time_ns()))]

input_shapes = [
    (1024, 1024),
    (8, 64, 256),
    (1, 3, 224, 224),
]


@pytest.mark.skip
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_custom_relu_kernel(
    seed: int, input_shape: tuple[int], dtype: torch.dtype
) -> None:

    if custom_ops is None:
        pytest.fail(
            "GML Custom Ops module not found. "
            "Run 'pip install -e .' or build extensions."
        )

    device = "cuda"
    torch.manual_seed(seed)

    input_tensor = (torch.rand(input_shape, dtype=dtype, device=device) * 2) - 1.0

    ref_output = torch.relu(input_tensor)

    custom_output = custom_ops.relu(input_tensor)

    snr = calc_blob_snr(ref_output, custom_output)

    assert snr > SNR_THRESH, f"{snr=:.2f} dB < {SNR_THRESH=} dB"
