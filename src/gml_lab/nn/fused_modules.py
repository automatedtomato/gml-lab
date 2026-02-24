from __future__ import annotations

import torch
from torch import nn


class AddReLU(nn.Module):
    """Custom fused Add ReLU module."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return nn.functional.relu(torch.add(a, b))
