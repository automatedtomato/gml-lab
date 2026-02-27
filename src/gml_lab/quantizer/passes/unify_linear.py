from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.fx import GraphModule


def unify_linear(gm: GraphModule) -> None:
    """Find linear function and unify to nn.Linear."""
    logger = get_logger("unify_pass")
    graph = gm.graph
    cnt = 0

    for node in graph.nodes:
        if node.op != "call_function" or node.target != F.linear:
            continue
        weight_node = node.args[1]
        if weight_node.op != "get_attr":
            continue
        weight_tensor = gm
        for w in str(weight_node.target).split("."):
            weight_tensor = getattr(weight_tensor, w)

        bias_tensor = None
        bias_node = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")  # noqa: PLR2004
        if bias_node is not None:
            bias_tensor = gm
            for b in str(bias_node.target).split("."):
                bias_tensor = getattr(bias_tensor, b)

        kwargs = {
            "in_features": weight_tensor.shape[1],
            "out_features": weight_tensor.shape[0],
            "bias": bias_tensor is not None,
        }

        new_module = torch.nn.Linear(**kwargs)
        with torch.no_grad():
            new_module.weight.copy_(weight_tensor)
            if bias_tensor is not None:
                new_module.bias.copy_(bias_tensor)

        new_name = graph._graph_namespace.create_name(f"unified_linear_{cnt}", None)
        gm.add_submodule(new_name, new_module)
        with graph.inserting_before(node):
            new_node = graph.call_module(
                new_name,
                args=(node.args[0],),
                kwargs={},
            )
            node.replace_all_uses_with(new_node)
            new_node.name = new_name
        graph.erase_node(node)
        logger.info(
            f' The function "{node.name}" is replaced with '
            f'the new module "{new_node.name}"'
        )
        cnt += 1

    gm.graph.lint()
    gm.recompile()
