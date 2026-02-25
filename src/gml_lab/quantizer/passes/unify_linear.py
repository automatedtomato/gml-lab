from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.fx import GraphModule


def unify_linear(gm: GraphModule) -> None:
    """Find linear finction and unify to nn.Linear."""
    logger = get_logger("unify_pass")
    graph = gm.graph
    cnt = 0
    new_module = None

    for node in graph.nodes:
        if node.op != "call_function" or node.target != F.linear:
            continue
        weight_node = node.args[1]
        if weight_node.op != "get_attr":
            continue
        for w in str(weight_node).split("."):
            weight_tensor = getattr(gm, str(w))
        bias = node.args[2] is not None

        kwargs = {
            "in_features": weight_tensor.shape[1],
            "out_features": weight_tensor.shape[0],
            "bias": bias,
        }
        new_module = torch.nn.Linear(**kwargs)

        if new_module is None:
            continue

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
        new_module = None

    gm.graph.lint()
    gm.recompile()
