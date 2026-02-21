from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.gml_lab.logger import get_logger
from src.gml_lab.utils import is_dequant_node, is_per_tensor_quant_node

from .utils import extract_qparams

if TYPE_CHECKING:
    from torch.fx import GraphModule


def lower_conv(gm: GraphModule) -> None:
    """Find the pattern and convert to custom kernel module.

    Patterns:
        - DQ -> Conv -> Q : GMLQuantConv
        - DQ -> Conv -> ReLU -> Q : GMLQuantConvReLU

    """
    logger = get_logger("lower_pass")

    relus = [torch.nn.functional.relu, torch.relu]
    modules = dict(gm.named_modules(remove_duplicate=False))
    cnt = 0
    graph = gm.graph

    for node in graph.nodes:
        if not is_per_tensor_quant_node(node):
            continue
        target_node = node.args[0]
        is_target_valid = False
        if (target_node.op == "call_function" and target_node.target in relus) or (
            target_node.op == "call_method" and target_node.target == "relu"
        ):
            is_target_valid = True
        elif target_node.op == "call_module":
            submodule = gm.get_submodule(target_node.target)
            if isinstance(submodule, torch.nn.ReLU):
                is_target_valid = True

        if not is_target_valid:
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
        graph.erase_node(node)
        graph.erase_node(target_node)
        graph.erase_node(dq_node)
        logger.info(
            f' The function/method "{target_node.name}" is replaced with '
            f'the new module "{new_node.name}"'
        )
        cnt += 1
    gm.graph.lint()
    gm.recompile()
