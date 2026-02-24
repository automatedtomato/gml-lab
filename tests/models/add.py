import torch
from torch import nn


class AddFunc(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.add(a, b)


class AddOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class AddMethod(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.add(b)


class IncrementalAdd(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a += b
        return a
