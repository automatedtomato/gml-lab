from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.gml_lab.kernel_class import GMLQuantReLU
from src.gml_lab.logger import get_logger
from src.gml_lab.utils import is_dequant_node, is_per_tensor_quant_node

from .utils import extract_qparams, remove_unused_nodes

if TYPE_CHECKING:
    from torch.fx import GraphModule


def lower_relu(gm: GraphModule) -> None:
    """Find `DQ -> ReLu -> Q` pattern and convert to custom kernel module."""
    logger = get_logger("lower_pass")

    cnt = 0
    graph = gm.graph

    for node in graph.nodes:
        if not is_per_tensor_quant_node(node):
            continue

        target_node = node.args[0]
        if target_node.op != "call_module":
            continue

        submodule = gm.get_submodule(target_node.target)
        if not isinstance(submodule, torch.nn.ReLU):
            continue

        dq_node = target_node.args[0]
        if not is_dequant_node(dq_node):
            continue

        kwargs = extract_qparams(gm, node)
        new_module = GMLQuantReLU(**kwargs)
        new_name = graph._graph_namespace.create_name(f"gml_q_relu_{cnt}", None)
        gm.add_submodule(new_name, new_module)

        with gm.graph.inserting_after(node):
            new_node = graph.call_module(new_name, args=(dq_node.args[0],), kwargs={})
            node.replace_all_uses_with(new_node)
            new_node.name = new_name
        remove_unused_nodes(graph, [node, target_node, dq_node])
        logger.info(
            f' The ReLU module "{target_node.name}" is replaced with '
            f'the new quant module "{new_node.name}"'
        )
        cnt += 1
    gm.graph.lint()
    gm.recompile()
