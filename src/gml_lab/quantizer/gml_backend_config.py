import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)

int8_config = DTypeConfig(
    input_dtype=torch.qint8,
    output_dtype=torch.qint8,
)


def get_gml_backend_config() -> BackendConfig:
    """Get GML Lab custom backend config."""
    backend_config = BackendConfig("gml_lab")

    default_ops = [
        torch.nn.ReLU,
        torch.nn.functional.relu,
        torch.relu,
        "relu",
    ]

    for op in default_ops:
        default_config = (
            BackendPatternConfig(op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs([int8_config])
        )
        backend_config.set_backend_pattern_config(default_config)
    return backend_config
