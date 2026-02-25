from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class GMLQuantModuleBase(torch.nn.Module, ABC):
    """Base module for all the kernel classes."""

    def __init__(
        self, output_scale: float | tuple[float, ...], output_zp: int | tuple[int, ...]
    ) -> None:
        super().__init__()
        self.register_buffer(
            "output_scale", torch.tensor(output_scale, dtype=torch.float32)
        )
        self.register_buffer("output_zp", torch.tensor(output_zp, dtype=torch.int32))

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
        """Abstract forward pass."""
        pass  # noqa: PIE790


class GMLQuantUnaryOpsBase(GMLQuantModuleBase):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class GMLQuantBinaryOpsBase(GMLQuantModuleBase):
    @abstractmethod
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pass
