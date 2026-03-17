from __future__ import annotations

import torch
import torch.nn.functional as F


class LUTAct:
    """Factory class for LUT generation."""

    @classmethod
    def generate(
        cls,
        lut: str,
        in_scale: float,
        in_zp: int,
        out_scale: float,
        out_zp: int,
    ) -> torch.Tensor:
        """Generate LUT for specified activation function."""
        quant_input = torch.arange(-128 ,128, dtype=torch.float32)
        float_input = (quant_input - in_zp) * in_scale
        if lut == "gelu":
            float_out = cls._calc_gelu(float_input)
        else:
            msg = f"LUT type {lut} is not supported."
            raise ValueError(msg)

        quant_out = torch.round(float_out / out_scale) + out_zp
        return torch.clamp(quant_out, -128, 127).to(torch.int8)

    @staticmethod
    def _calc_gelu(x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)
