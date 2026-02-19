import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module


class ReLUModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


class ReLUFunc1(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class ReLUFunc2(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class ReLUMethod(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu()
