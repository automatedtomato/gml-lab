from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch

import src.gml_lab.nn.modules as cnn
from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.fx import GraphModule


def unify_add(gm: GraphModule) -> None:
    """Find Add finction/method and unify to custom module cnn.Add."""
    logger = get_logger("unify_pass")
    graph = gm.graph
    cnt = 0
    new_module = None

    add_target = [
        torch.add,
        operator.add,
        operator.iadd,
        "add",
        "add_",
    ]

    for node in list(graph.nodes):
        if node.op not in {"call_method", "call_function"}:
            continue
        if node.target not in add_target:
            continue
        new_module = cnn.Add()
        new_name = graph._graph_namespace.create_name(f"unified_add_{cnt}", None)
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
