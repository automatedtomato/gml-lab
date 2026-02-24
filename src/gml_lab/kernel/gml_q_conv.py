from __future__ import annotations

import os

import torch

from src.gml_lab.logger import get_logger

logger = get_logger("gml_q_relu")
enable_custom_ops = os.getenv("ENABLE_CUSTOMOPS", "1") not in ["0", False]

if not enable_custom_ops:
    custom_ops = None

# try:
#     import gml_lab_custom_ops as custom_ops
# except ImportError as e:
#     logger.error(f"Error importing custom_ops: {e}")
#     custom_ops = None


class GMLQuantConvBase(torch.nn.Module):
    """GML custom quantized Conv kernel class."""

    def __init__(
        self,
        conv_module: torch.nn.Module,
        in_scale: float | tuple[float],
        in_zp: int | tuple[int],
        out_scale: float | tuple[float],
        out_zp: int | tuple[int],
    ) -> None:
        super().__init__()
        self.register_buffer("input_scale", torch.tensor(in_scale, dtype=torch.float32))
        self.register_buffer("input_zp", torch.tensor(in_zp, dtype=torch.int32))
        self.register_buffer(
            "output_scale", torch.tensor(out_scale, dtype=torch.float32)
        )
        self.register_buffer("output_zp", torch.tensor(out_zp, dtype=torch.int32))
        self.weight = torch.nn.Parameter(conv_module.weight.detach().clone())
        if conv_module.bias is not None:
            self.bias = torch.nn.Parameter(conv_module.bias.detach().clone())
        else:
            self.register_parameter("bias", None)
        self.stride = conv_module.stride
        self.padding = conv_module.padding
        self.groups = conv_module.groups
        self.dilation = conv_module.dilation

        if conv_module.weight.is_quantized:
            weight_scale = conv_module.weight.q_scale()
            weight_zp = conv_module.weight.q_zero_point()
            self.weight = torch.nn.Parameter(
                conv_module.weight.int_repr().detach().clone()
            )
        else:
            weight_scale = 1.0
            weight_zp = 0
            self.weight = torch.nn.Parameter(conv_module.weight.detach().clone())

        self.register_buffer(
            "weight_scale", torch.tensor(weight_scale, dtype=torch.float32)
        )
        self.register_buffer("weight_zp", torch.tensor(weight_zp, dtype=torch.int32))

    def _inner_forward(self, x: torch.Tensor, has_relu: bool) -> torch.Tensor:  # noqa: FBT001
        """Simulate kernel forward pass."""
        x = x.dequantize()
        out = torch.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if has_relu:
            out = torch.nn.functional.relu(out)

        return torch.quantize_per_tensor(
            out,
            scale=self.output_scale.item(),
            zero_point=self.output_zp.item(),
            dtype=torch.qint8,
        )


class GMLQuantConv(GMLQuantConvBase):
    """Quantized Conv2d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run kernel simulated forward pass."""
        return self._inner_forward(x, has_relu=False)


class GMLQuantConvReLU(GMLQuantConvBase):
    """Quantized ConvReLU2d."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run kernel simulated forward pass."""
        return self._inner_forward(x, has_relu=True)
