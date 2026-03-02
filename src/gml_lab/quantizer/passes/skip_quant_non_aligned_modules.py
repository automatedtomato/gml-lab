from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.ao.quantization.qconfig_mapping import QConfigMapping


def skip_quant_non_aligned_modules(
    gm: torch.fx.GraphModule, qconfig_mapping: QConfigMapping | dict[Any]
) -> QConfigMapping:
    """Skip quantization process of modules that do not meet hardware constraints."""
    logger = get_logger("passes")
    for name, module in gm.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels % 16 != 0:
            logger.info(
                f'Skip quantization for "{name}" due to hardware alignment '
                f"(in_channels={module.in_channels})."
            )
            qconfig_mapping.set_module_name(name, None)
    return qconfig_mapping
