from __future__ import annotations

import os

import torch

from src.gml_lab.logger import get_logger

logger = get_logger("gml_q_relu")
enable_custom_ops = os.getenv("ENABLE_CUSTOMOPS", "1") not in ["0", False]

if not enable_custom_ops:
    custom_ops = None

try:
    import gml_lab_custom_ops as custom_ops
except ImportError as e:
    logger.error(f"Error importing custom_ops: {e}")
    custom_ops = None


class GMLQuantReLU(torch.nn.Module):
    """GML custom quantized ReLU kernel class."""

    def __init__(
        self,
        scale: float | tuple[float],
        zero_point: int | tuple[int],
    ) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("zero_point", torch.tensor(zero_point, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run kernel simuated forward pass."""
        zp = self.zero_point.item()
        if custom_ops is None:
            x = x.dequantize()
            out = torch.clamp(x, min=zp)
            return torch.quantize_per_tensor(
                x,
                scale=self.scale.item(),
                zero_point=zp,
                dtype=torch.qint8,
            )

        x = x.int_repr()
        out = custom_ops.quant_relu(x, zp)

        return torch._make_per_tensor_quantized_tensor(
            out, scale=self.scale.item(), zero_point=zp
        )
