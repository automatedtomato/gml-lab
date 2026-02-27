import torch
from torch import nn


class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LinearFunc(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class LinearBN(LinearModule):
    def __init__(self, in_features: int, out_features: int, *, bias: bool) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(super().forward(x))
