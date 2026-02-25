import copy

import torch

from src.gml_lab.lowering.passes import lower_add, lower_relu


def lower_to_gml(qdq_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Travers the graph module and lower a quantized reference model.

    Args:
        qdq_model (GraphModule): GraphModule after gml_convert_fx.

    Returns:
        GraphModule: lowered GraphModule with custom CUDA kernels.

    """
    gml_model = copy.deepcopy(qdq_model)
    lower_add(gml_model)
    lower_relu(gml_model)
    return gml_model
