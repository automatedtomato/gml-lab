from __future__ import annotations

import torch


class Add(torch.nn.Module):
    """Modulized Add operator."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return torch.add(a, b)
