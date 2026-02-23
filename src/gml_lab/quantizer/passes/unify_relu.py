from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.fx import GraphModule


def unify_relu(gm: GraphModule) -> None:
    """Find ReLU finction/method and unify to nn.ReLU."""
    logger = get_logger("unify_pass")
    graph = gm.graph
    cnt = 0
    new_module = None

    for node in graph.nodes:
        if node.op == "call_function" and node.target == F.relu:
            kwargs = node.kwargs
            new_module = torch.nn.ReLU(**kwargs)
        elif node.op == "call_method" and node.target == "relu":
            new_module = torch.nn.ReLU()
        if new_module is not None:
            new_name = graph._graph_namespace.create_name(f"unified_relu_{cnt}", None)
            gm.add_submodule(new_name, new_module)

            with graph.inserting_before(node):
                new_node = graph.call_module(
                    new_name,
                    args=node.args,
                    kwargs={},
                )
                node.replace_all_uses_with(new_node)
                new_node.name = new_name
            graph.erase_node(node)
            logger.info(
                f' The function/method "{node.name}" is replaced with '
                f'the new module "{new_node.name}"'
            )
            cnt += 1
            new_module = None

    gm.graph.lint()
    gm.recompile()
