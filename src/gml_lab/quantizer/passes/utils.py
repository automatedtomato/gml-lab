from typing import Any

import torch


def get_arg(idx: int, node: torch.fx.Node, name: str, default: Any) -> Any:  # noqa: ANN401
    """Get arg or kwarg from node."""
    if idx < len(node.args):
        return node.args[idx]
    return node.kwargs.get(name, default)
