from __future__ import annotations

import os

import torch

from src.gml_lab.logger import get_logger

from .gml_quant_base import GMLQuantUnaryOpsBase

logger = get_logger("gml_fusedq_conv")
enable_custom_ops = os.getenv("ENABLE_CUSTOMOPS", "1") not in ["0", False]

if not enable_custom_ops:
    custom_ops = None

# try:
#     import gml_lab_custom_ops as custom_ops
# except ImportError as e:
#     logger.error(f"Error importing custom_ops: {e}")
#     custom_ops = None
custom_ops = None


class GMLQuantConvBase(GMLQuantUnaryOpsBase):
    """GML custom quantized Conv kernel class."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        weight_scale: float,
        weight_zp: int,
        input_scale: float,
        input_zp: int,
        output_scale: float,
        output_zp: int,
        weight_quant_axis: int | None = 0,
    ) -> None:
        super().__init__(output_scale, output_zp)
        self.register_buffer(
            "input_scale", torch.tensor(input_scale, dtype=torch.float32)
        )
        self.register_buffer("input_zp", torch.tensor(input_zp, dtype=torch.int32))

        w_scale_t = torch.tensor(
            weight_scale, dtype=torch.float32, device=weight.device
        )
        w_zp_t = torch.tensor(weight_zp, dtype=torch.int32, device=weight.device)
        self.register_buffer("weight_scale", w_scale_t.clone().detach())
        self.register_buffer("weight_zp", w_zp_t.clone().detach())

        requant_scale = (input_scale * w_scale_t) / output_scale
        self.register_buffer("requant_scale", requant_scale.clone().detach())

        if w_scale_t.numel() == 1:
            weight = torch.quantize_per_tensor(
                weight, w_scale_t.item(), w_zp_t.item(), dtype=torch.qint8
            )
        else:
            weight = torch.quantize_per_channel(
                weight,
                w_scale_t,
                w_zp_t,
                axis=weight_quant_axis,
                dtype=torch.qint8,
            )
        self.register_buffer("weight", weight.int_repr())

        if bias is not None:
            self.register_buffer("bias", bias)
            bias_scale = input_scale * w_scale_t
            bias_int32 = torch.round(bias / bias_scale).to(torch.int32)
            self.register_buffer("bias_int32", bias_int32)
        else:
            self.bias = None
            self.bias_int32 = None

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight_quant_axis = weight_quant_axis

    def _inner_forward(self, x: torch.Tensor, fuse_relu: bool) -> torch.Tensor:  # noqa: FBT001
        """Simulate kernel forward pass."""
        scale = self.output_scale.item()
        zp = self.output_zp.item()
        w_scale = self.weight_scale

        if custom_ops is None:
            w_zp = self.weight_zp
            if w_scale.dim() == 1:
                w_scale = w_scale.view(-1, 1, 1, 1)
                w_zp = w_zp.view(-1, 1, 1, 1)
            weight = (self.weight.to(torch.int32) - w_zp) * w_scale
            x = x.dequantize()
            out = torch.nn.functional.conv2d(
                x,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            if fuse_relu:
                out = torch.nn.functional.relu(out)

            return torch.quantize_per_tensor(
                out,
                scale=scale,
                zero_point=zp,
                dtype=torch.qint8,
            )

        x = x.int_repr()
        if w_scale.dim() == 1:
            w_scale = w_scale.view(-1, 1, 1, 1)

        is_per_channel = self.weight_scale.numel() > 1
        out = custom_ops.quant_fused_conv(
            x,
            self.weight,
            self.bias_int32,
            self.requant_scale,
            zp,
            fuse_relu,
            is_per_channel,
        )

        raise NotImplementedError


class GMLQuantConv(GMLQuantConvBase):
    """Quantized Conv2d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run kernel simulated forward pass."""
        return self._inner_forward(x, fuse_relu=False)


class GMLQuantConvReLU(GMLQuantConvBase):
    """Quantized ConvReLU2d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run kernel simulated forward pass."""
        return self._inner_forward(x, fuse_relu=True)
