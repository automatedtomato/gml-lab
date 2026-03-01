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


class ConvFunc(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int],
        bias: bool,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


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
