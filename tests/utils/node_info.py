from collections.abc import Callable

import torch
from torch.fx import Node
from torch.nn import Module


class NodeInfo:
    """Node information composed of node.op and node.target.

    Args:
        op (str): "call_method", "call_function" or "call_module"
        target (str | Callable | torch.nn.Module):
            str for "call_method"
            Callable for "call_function"
            torch.nn.Module for "call_module

    """

    def __init__(self, op: str, target: str | Callable | torch.nn.Module) -> None:
        self.op = op
        self.target = target

    @classmethod
    def call_method(cls, target: str) -> "NodeInfo":
        return NodeInfo("call_method", target)

    @classmethod
    def call_function(cls, target: Callable) -> "NodeInfo":
        return NodeInfo("call_function", target)

    @classmethod
    def call_module(cls, target: torch.nn.Module) -> "NodeInfo":
        return NodeInfo("call_module", target)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeInfo):
            return False
        return self.op == other.op and self.target == other.target

    def __hash__(self) -> int:
        return hash((self.op, self.target))


def get_node_info(node: Node, modules: dict[str, Module]) -> NodeInfo | None:
    if node.op in ["placeholder", "output", "get_attr"]:
        node_info = None
    elif node.op == "call_module":
        target_cls = type(modules[node.target])
        node_info = NodeInfo.call_module(target_cls)
    elif node.op == "call_function":
        node_info = NodeInfo.call_function(node.target)
    elif node.op == "call_method":
        node_info = NodeInfo.call_method(node.target)
    else:
        msg = f"Node type {node.op} is not supported."
        raise TypeError(msg)
    return node_info
