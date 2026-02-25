from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
from torch.ao.quantization.quantize_fx import prepare_fx

from src.gml_lab.quantizer.gml_backend_config import get_prepare_custom_config
from src.gml_lab.quantizer.passes import fuse_add_relu, unify_add, unify_relu

if TYPE_CHECKING:
    from torch.ao.quantization.backend_config import BackendConfig
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.fx import GraphModule


def _gml_fuse_fx(gm: GraphModule) -> GraphModule:
    """Apply custom fused pass."""
    fuse_add_relu(gm)
    return gm


def gml_prepare_fx(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    qconfig_mapping: QConfigMapping | dict[str, Any],
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    """Prepare a model for PTQ.

    See `torch.ao.quantization.prepare_fx` for reference.

    Args:
        model (torch.nn.Module): float model to be quantized
        example_inputs (tule[Any, ...]): tuple of input tensor
        qconfig_mapping (QConfigMapping): quantization configuration
        backend_config (QConfigMapping): backend configuration

    Returns:
        torch.fx.GraphModule: the prepared model with fake quant.

    """
    model = copy.deepcopy(model)
    gm = torch.fx.symbolic_trace(model)

    unify_add(gm)
    unify_relu(gm)

    fused_model = _gml_fuse_fx(gm)

    fused_model.eval()
    prepare_custom_config = get_prepare_custom_config()

    return prepare_fx(
        model=fused_model,
        prepare_custom_config=prepare_custom_config,
        qconfig_mapping=qconfig_mapping,
        backend_config=backend_config,
        example_inputs=example_inputs,
    )
