import torch
from torch.ao.quantization import get_default_qconfig_mapping, observer
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.qconfig_mapping import QConfigMapping

from src.gml_lab.utils import INT8_MAX, INT8_MIN


def get_gml_qconfig(method: str = "per_tensor") -> QConfig:
    """Get QConfig for PTQ."""
    if method == "per_tensor":
        qconfig = QConfig(
            activation=observer.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                quant_min=INT8_MIN,
                quant_max=INT8_MAX,
            ),
            weight=observer.default_per_channel_weight_observer,
        )
    else:
        msg = f"method `{method}` is not implemented yet."
        raise NotImplementedError(msg)
    return qconfig


def get_gml_qconfig_mapping(method: str = "per_tensor") -> QConfigMapping:
    """Get QConfig mapping for PTQ.

    Args:
        method (str): observer type.

    Returns:
        QConfigMapping

    """
    qconfig = get_gml_qconfig(method)
    return QConfigMapping().set_global(qconfig)


def build_qconfig_mapping(backend: str = "fbgemm") -> QConfigMapping:
    """Build quantization config mapping.

    Args:
        backend (str): Quantization backend.
            - "fbgemm": for x86 servers (default)
            - "qnnpack": for ARM/Mobile

    Returns:
        QConfigMapping: pytorch quantization configuration

    """
    return get_default_qconfig_mapping(backend)
