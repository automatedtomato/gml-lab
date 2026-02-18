from __future__ import annotations

import io

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.gml_lab.quantizer import (
    build_qconfig_mapping,
    gml_convert_fx,
    gml_prepare_fx,
)

NO_GPU = not torch.cuda.is_available()

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min

SNR_THRESH_LINEAR = 50


def get_model_size(model: torch.fx.GraphModule | torch.nn.Module) -> float:
    """Calculate size (MB) by serializing."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def get_test_dataloader(
    input_shape: tuple[int, ...], num_samples: int = 320, batch_size: int = 64
) -> DataLoader:
    dummy_input = torch.randn(num_samples, *input_shape)
    dummy_target = torch.randint(INT8_MIN, INT8_MAX, (num_samples,))
    dataset = TensorDataset(dummy_input, dummy_target)
    return DataLoader(dataset, batch_size=batch_size)


def quantize_model(
    float_model: torch.nn.Module,
    input_shape: tuple[int, ...] = (3, 28, 28),
    device: str = "cpu",
) -> tuple[torch.fx.GraphModule, ...]:
    """Quantize model."""
    float_model = float_model.to(device).eval()
    dummy_inputs = (torch.randn(1, *input_shape),)
    dummy_inputs_on_device = tuple(i.to(device) for i in dummy_inputs)
    qconfig_mapping = build_qconfig_mapping()
    calib_loader = get_test_dataloader(input_shape)

    prepared_model = gml_prepare_fx(
        float_model, dummy_inputs_on_device, qconfig_mapping
    )

    with torch.no_grad():
        for batch in calib_loader:
            inputs = batch[0].to(device)
            _ = prepared_model(inputs)

    qdq_model = gml_convert_fx(prepared_model)

    return prepared_model, qdq_model


def _calc_blob_snr(blob_org: torch.Tensor, blob_target: torch.Tensor) -> float:
    """Compute P-SNR between given two tensors."""
    blob_org = blob_org.cpu().detach().flatten().numpy()
    blob_target = blob_target.cpu().detach().flatten().numpy()

    blob_dif = blob_org - blob_target
    blob_max = np.max(blob_org)
    blob_min = np.min(blob_org)

    blob_amp = blob_max - blob_min
    if blob_amp == 0.0:
        return float("nan")
    blob_mse = np.mean(blob_dif**2)
    if blob_mse == 0.0:
        return float("inf")

    return -20 * np.log10(np.sqrt(blob_mse) / blob_amp)
