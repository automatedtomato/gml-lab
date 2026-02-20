from __future__ import annotations

import torch


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
        _in = x.int_repr()
        out = torch.clamp(_in, min=self.zero_point.item())
        return torch._make_per_tensor_quantized_tensor(
            out, scale=self.scale.item(), zero_point=self.zero_point.item()
        )
