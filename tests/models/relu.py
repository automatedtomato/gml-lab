import torch
import torch.nn.functional as F
from torch import nn


class ReLUModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


class ReLUFunc1(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class ReLUFunc2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class ReLUMethod(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu()
