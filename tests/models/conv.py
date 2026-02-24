import torch
from torch import nn


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBN(ConvModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(super().forward(x))


class ConvReLUFunc(ConvModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(super().forward(x))


class ConvReLUMod(ConvModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(super().forward(x))


class ConvBNReLUMod(ConvBN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs,
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(super().forward(x))


class ConvBNReLUFunc(ConvBN):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(super().forward(x))
