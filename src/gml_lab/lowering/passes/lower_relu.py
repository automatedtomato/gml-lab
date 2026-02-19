from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.fx import GraphModule

try:
    import gml_lab_custom_ops as custom_ops
except ImportError as e:
    print(f"Error importing custom_ops: {e}")
    custom_ops = None


def lower_relu(gm: GraphModule) -> None:
    """Lower ReLU in graph module."""
    logger = get_logger("lower_pass")

    if custom_ops is None:
        logger.warning("custom_ops is None. Skipping pass.")
        return

    relus = [torch.nn.functional.relu, torch.relu]
    graph = gm.graph

    for node in graph.nodes:
        if (node.op == "call_function" and node.target in relus) or (
            node.op == "call_method" and node.target == "relu"
        ):
            new_name = graph._graph_namespace.create_name("cuda_" + node.name, None)
            with gm.graph.inserting_after(node):
                new_node = graph.call_function(
                    custom_ops.relu, args=node.args, kwargs={}
                )
                node.replace_all_uses_with(new_node)
                new_node.name = new_name
            graph.erase_node(node)
            logger.info(
                f' The function/method "{node.name}" is replaced with '
                f'custom_ops function "{new_node.name}"'
            )

        elif node.op == "call_module":
            submodule = gm.get_submodule(node.target)
            if isinstance(submodule, torch.nn.ReLU):
                new_name = graph._graph_namespace.create_name("cuda_" + node.name, None)
                with gm.graph.inserting_after(node):
                    new_node = graph.call_function(
                        custom_ops.relu, args=node.args, kwargs={}
                    )
                    node.replace_all_uses_with(new_node)
                    new_node.name = new_name
                graph.erase_node(node)
                logger.info(
                    f' The module "{node.name}" is replaced with '
                    f'custom_ops function "{new_node.name}"'
                )
    gm.graph.lint()
    gm.recompile()
