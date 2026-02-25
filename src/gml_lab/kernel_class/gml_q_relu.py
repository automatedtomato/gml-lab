from __future__ import annotations

import os

import torch

from src.gml_lab.kernel_class.gml_quant_base import GMLQuantUnaryOpsBase
from src.gml_lab.logger import get_logger

logger = get_logger("gml_q_relu")

try:
    import gml_lab_custom_ops as custom_ops
except ImportError as e:
    logger.error(f"Error importing custom_ops: {e}")
    custom_ops = None

enable_custom_ops = os.getenv("ENABLE_CUSTOMOPS", "1") not in ["0", False]

if not enable_custom_ops:
    custom_ops = None


class GMLQuantReLU(GMLQuantUnaryOpsBase):
    """GML custom quantized ReLU kernel class."""

    def __init__(
        self,
        output_scale: float | tuple[float],
        output_zp: int | tuple[int],
    ) -> None:
        super().__init__(output_scale, output_zp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zp = self.output_zp.item()
        scale = self.output_scale.item()
        if custom_ops is None:
            x = x.dequantize()
            out = torch.nn.functional.relu(x)
            return torch.quantize_per_tensor(
                out,
                scale=scale,
                zero_point=zp,
                dtype=torch.qint8,
            )

        x = x.int_repr()
        out = custom_ops.quant_relu(x, zp)

        return torch._make_per_tensor_quantized_tensor(out, scale=scale, zero_point=zp)
