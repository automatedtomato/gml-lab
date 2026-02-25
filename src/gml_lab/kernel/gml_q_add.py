from __future__ import annotations

import os

import torch

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


class GMLQuantAddBase(torch.nn.Module):
    """GML custom quantized Add kernel base class."""

    def __init__(
        self,
        in_scale_a: float,
        in_za: int,
        in_scale_b: float,
        in_zb: int,
        out_scale: float,
        out_zp: int,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "input_scale_a", torch.tensor(in_scale_a, dtype=torch.float32)
        )
        self.register_buffer("in_za", torch.tensor(in_za, dtype=torch.int32))
        self.register_buffer(
            "input_scale_b", torch.tensor(in_scale_b, dtype=torch.float32)
        )
        self.register_buffer("in_zb", torch.tensor(in_zb, dtype=torch.int32))
        self.register_buffer(
            "output_scale", torch.tensor(out_scale, dtype=torch.float32)
        )
        self.register_buffer("output_zp", torch.tensor(out_zp, dtype=torch.int32))
        requant_scale_a = in_scale_a / out_scale
        requant_scale_b = in_scale_b / out_scale
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
        """Simulate kernel forward pass."""
        if custom_ops is None:
            a_float = a.dequantize()
            b_float = b.dequantize()
            out_float = torch.add(a_float, b_float)
            if has_relu:
                out_float = torch.nn.functional.relu(out_float)
            return torch.quantize_per_tensor(
                out_float,
                scale=self.output_scale.item(),
                zero_point=self.output_zp.item(),
                dtype=torch.qint8,
            )
        a_q = a.int_repr()
        b_q = b.int_repr()

        out_int8 = custom_ops.quant_add(
            a_q,
            b_q,
            self.in_za.item(),
            self.in_zb.item(),
            self.output_zp.item(),
            self.requant_scale_a.item(),
            self.requant_scale_b.item(),
            has_relu,
        )

        return torch._make_per_tensor_quantized_tensor(
            out_int8,
            scale=self.output_scale.item(),
            zero_point=self.output_zp.item(),
        )


class GMLQuantAdd(GMLQuantAddBase):
    """Quantized Add."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run kernel simulated forward pass."""
        return self._inner_forward(a, b, has_relu=False)


class GMLQuantAddReLU(GMLQuantAddBase):
    """Quantized AddReLU."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run kernel simulated forward pass."""
        return self._inner_forward(a, b, has_relu=True)
