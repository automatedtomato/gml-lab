from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.qconfig_mapping import QConfigMapping


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
