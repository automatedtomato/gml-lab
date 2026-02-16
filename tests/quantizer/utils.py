from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import torch

from src.gml_lab.modeling import FxWrapper
from src.gml_lab.quantizer import (
    calibrate_model,
    gml_convert_fx,
    gml_prepare_fx,
)

if TYPE_CHECKING:
    from torch.ao.quantization.qconfig_mapping import QConfigMapping


def get_model_size(model: torch.fx.GraphModule | torch.nn.Module) -> float:
    """Calculate size (MB) by serializing."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def quantize(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    qconfig_mapping: QConfigMapping | dict[str, Any],
    calib_loader: Any,  # noqa: ANN401
    total_calib_batches: int,
    data_preprocessor: torch.nn.Module,
) -> tuple[torch.fx.GraphModule, ...]:
    """Quantize model."""
    float_model = FxWrapper(model)
    prepared_model = gml_prepare_fx(float_model, example_inputs, qconfig_mapping)
    prepared_model = calibrate_model(
        prepared_model,
        data_preprocessor=data_preprocessor,
        calib_loader=calib_loader,
        total_calib_batches=total_calib_batches,
    )
    qdq_model = gml_convert_fx(prepared_model, qconfig_mapping)

    return prepared_model, qdq_model
