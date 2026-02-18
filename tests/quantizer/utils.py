from __future__ import annotations

import io

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.gml_lab.quantizer import (
    build_qconfig_mapping,
    gml_convert_fx,
    gml_prepare_fx,
)

int8_max = torch.iinfo(torch.int8).max
int8_min = torch.iinfo(torch.int8).min


def get_model_size(model: torch.fx.GraphModule | torch.nn.Module) -> float:
    """Calculate size (MB) by serializing."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def get_test_dataloader(
    input_shape: tuple[int, ...], num_samples: int = 320, batch_size: int = 64
) -> DataLoader:
    dummy_input = torch.randn(num_samples, *input_shape)
    dummy_target = torch.randint(int8_min, int8_max, (num_samples,))
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
