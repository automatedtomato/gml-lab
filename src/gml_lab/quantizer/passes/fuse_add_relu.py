from __future__ import annotations

from typing import TYPE_CHECKING

import src.gml_lab.nn.fused_modules as fcnn
from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from torch.fx import GraphModule


def fuse_add_relu(gm: GraphModule) -> None:
    """Find Add ReLU sequence and fuse into custom fused module fcnn.AddReLU."""
    logger = get_logger("unify_pass")
    graph = gm.graph
    cnt = 0
    new_module = None

    for node in graph.nodes:
        if node.op != "call_module" or not node.target.startswith("unified_add_"):
            continue
        users = list(node.users.keys())
        next_node = users[0]
        if next_node.op != "call_module" or not next_node.target.startswith(
            "unified_relu_"
        ):
            continue
        new_module = fcnn.AddReLU()
        new_name = graph._graph_namespace.create_name(f"fused_add_relu_{cnt}", None)
        gm.add_submodule(new_name, new_module)

        with graph.inserting_before(node):
            new_node = graph.call_module(
                new_name,
                args=node.args,
                kwargs={},
            )
            next_node.replace_all_uses_with(new_node)
            new_node.name = new_name
        graph.erase_node(next_node)
        graph.erase_node(node)
        logger.info(
            f' "{node.name}" and {next_node.name} is fused into '
            f'a new module "{new_node.name}"'
        )
        cnt += 1
        new_module = None

    gm.graph.lint()
    gm.recompile()
