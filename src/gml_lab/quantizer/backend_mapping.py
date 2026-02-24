from torch import nn

CONV_BN_MAP = [(nn.Conv2d, nn.BatchNorm2d)]
CONV_BN_RELU_MAP = [
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
    (nn.Conv2d, nn.BatchNorm2d, nn.functional.relu),
]
CONV_RELU_MAP = [
    (nn.Conv2d, nn.ReLU),
    (nn.Conv2d, nn.functional.relu)
]
