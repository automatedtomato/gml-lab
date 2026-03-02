from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from torch.ao.quantization.quantize_fx import (
    convert_fx,
    convert_to_reference_fx,
)

from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.ao.quantization.backend_config import BackendConfig
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.fx import GraphModule


logger = get_logger("gml_convert_fx")


def gml_convert_fx(
    prepared_model: GraphModule,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
    fake_quantize: bool = True,  # noqa: FBT001, FBT002
) -> GraphModule:
    """Convert a calibrated model to fake-quantized model.

    See `torch.ao.quantization.convert_to_referencefx` for reference.

    Args:
        prepared_model (torch.nn.Module): prepared model to be quantized
        qconfig_mapping (QConfigMapping): quantization configuration
        backend_config (QConfigMapping): backend configuration
        fake_quantize (bool): If true, convert to fake quantize model,
                                if false to int8 model

    Returns:
        torch.fx.GraphModule: the prepared model with fake quant.

    """
    prepared_model = copy.deepcopy(prepared_model)
    if fake_quantize:
        logger.info("Process fake quantization.")
        return convert_to_reference_fx(
            graph_module=prepared_model,
            qconfig_mapping=qconfig_mapping,
            backend_config=backend_config,
        )
    logger.info("Model quantize to int8.")
    return convert_fx(
        graph_module=prepared_model,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
    )
