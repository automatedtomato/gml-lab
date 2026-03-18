from __future__ import annotations

import os

import torch

from src.gml_lab.kernel_class.gml_quant_base import GMLQuantUnaryOpsBase
from src.gml_lab.logger import get_logger

from .utils import LUTAct

logger = get_logger("gml_q_lut")

try:
    import gml_lab_custom_ops as custom_ops
except ImportError as e:
    logger.error(f"Error importing custom_ops: {e}")
    custom_ops = None

enable_custom_ops = os.getenv("ENABLE_CUSTOMOPS", "1") not in ["0", False]

if not enable_custom_ops:
    custom_ops = None


class GMLQuantLUT(GMLQuantUnaryOpsBase):
    """GML custom quantized LUT activation kernel class."""

    def __init__(
        self,
        input_scale: float,
        input_zp: int,
        output_scale: float | tuple[float],
        output_zp: int | tuple[int],
        lut: str,
    ) -> None:
        super().__init__(output_scale, output_zp)
        self.input_scale = input_scale
        self.input_zp = input_zp

        lut_tensor = LUTAct.generate(
            lut=lut,
            in_scale=input_scale,
            in_zp=input_zp,
            out_scale=output_scale,
            out_zp=output_zp,
        )
        self.register_buffer("lut", lut_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zp = self.output_zp.item()
        scale = self.output_scale.item()
        x = x.int_repr()
        if custom_ops is None:
            x = x.to(torch.long).to("cpu")
            out = self.lut[x + 128]
        else:
            lut = self.lut.to("cuda")
            out = custom_ops.quant_lut(x, lut)
        return torch._make_per_tensor_quantized_tensor(
            out,
            scale=scale,
            zero_point=zp,
        )
