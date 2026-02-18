from __future__ import annotations

import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.observer import ObserverBase
from torch.fx import GraphModule, Node


def is_quantize_node(node: Node) -> bool:
    """Return true if give node is quantize function."""
    return (
        isinstance(node, Node)
        and node.op in ["call_function", "call_module"]
        and node.target in [torch.quantize_per_tensor, torch.quantize_per_channel]
    )


def is_observer_node(node: Node, modules: dict[str, GraphModule]) -> bool:
    """Return true if given node is observer."""
    return (
        isinstance(node, Node)
        and node.op == "call_module"
        and node.target in modules
        and issubclass(type(modules[node.target]), ObserverBase)
    )


def is_fake_quant_node(node: Node, modules: dict[str, GraphModule]) -> bool:
    """Return True if the given node corresponds to a FakeQuantize module."""
    return (
        isinstance(node, Node)
        and node.op == "call_module"
        and node.target in modules
        and isinstance(type(modules[node.target]), FakeQuantizeBase)
    )
