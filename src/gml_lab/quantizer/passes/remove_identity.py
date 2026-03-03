from __future__ import annotations

import torch

from src.gml_lab.logger import get_logger


def remove_identity(gm: torch.fx.GraphModule) -> None:
    """Find and remove nn.Identity from graph, bypass its input."""
    logger = get_logger("remove_pass")
    graph = gm.graph
    remove = False

    for node in graph.nodes:
        if node.op != "call_module":
            continue
        submod = gm.get_submodule(node.target)
        if isinstance(submod, torch.nn.Identity):
            node.replace_all_uses_with(node.args[0])
            logger.info(f'The nn.Identity "{node.name}" is removed from the graph.')
            remove = True

    if remove:
        graph.eliminate_dead_code()
        gm.delete_all_unused_submodules()
        graph.lint()
        gm.recompile()
