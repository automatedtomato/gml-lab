from __future__ import annotations

import os

import torch

from src.gml_lab.kernel_class.gml_quant_base import GMLQuantBinaryOpsBase
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


class GMLQuantAddBase(GMLQuantBinaryOpsBase):
    """GML custom quantized Add kernel base class."""

    def __init__(
        self,
        input_scale_a: float,
        input_za: int,
        input_scale_b: float,
        input_zb: int,
        output_scale: float,
        output_zp: int,
    ) -> None:
        super().__init__(output_scale, output_zp)
        self.register_buffer(
            "input_scale_a", torch.tensor(input_scale_a, dtype=torch.float32)
        )
        self.register_buffer("input_za", torch.tensor(input_za, dtype=torch.int32))
        self.register_buffer(
            "input_scale_b", torch.tensor(input_scale_b, dtype=torch.float32)
        )
        self.register_buffer("input_zb", torch.tensor(input_zb, dtype=torch.int32))
        requant_scale_a = input_scale_a / output_scale
        requant_scale_b = input_scale_b / output_scale
        self.register_buffer(
            "requant_scale_a", torch.tensor(requant_scale_a, dtype=torch.float32)
        )
        self.register_buffer(
            "requant_scale_b", torch.tensor(requant_scale_b, dtype=torch.float32)
        )

    def _inner_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        has_relu: bool,  # noqa: FBT001
    ) -> torch.Tensor:
        scale = self.output_scale.item()
        zp = self.output_zp.item()
        if custom_ops is None:
            a_float = a.dequantize()
            b_float = b.dequantize()
            out_float = torch.add(a_float, b_float)
            if has_relu:
                out_float = torch.nn.functional.relu(out_float)
            return torch.quantize_per_tensor(
                out_float,
                scale=scale,
                zero_point=zp,
                dtype=torch.qint8,
            )
        a_q = a.int_repr()
        b_q = b.int_repr()

        out_int8 = custom_ops.quant_add(
            a_q,
            b_q,
            self.input_za.item(),
            self.input_zb.item(),
            zp,
            self.requant_scale_a.item(),
            self.requant_scale_b.item(),
            has_relu,
        )

        return torch._make_per_tensor_quantized_tensor(
            out_int8,
            scale=scale,
            zero_point=zp,
        )


class GMLQuantAdd(GMLQuantAddBase):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._inner_forward(a, b, has_relu=False)


class GMLQuantAddReLU(GMLQuantAddBase):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._inner_forward(a, b, has_relu=True)
