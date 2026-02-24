from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.intrinsic as nni

from src.gml_lab.kernel import GMLQuantConv, GMLQuantConvReLU
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

    cnt = 0
    graph = gm.graph

    for node in graph.nodes:
        if not is_per_tensor_quant_node(node):
            continue

        target_node = node.args[0]
        if target_node.op != "call_module":
            continue

        submodule = gm.get_submodule(target_node.target)
        if not isinstance(submodule, torch.nn.Conv2d | nni.ConvReLU2d):
            continue

        dq_node = target_node.args[0]

        if not is_dequant_node(dq_node):
            continue

        kwargs = extract_qparams(gm, node)
        out_qparams = extract_qparams(gm, node)
        in_qparams = extract_qparams(gm, dq_node.args[0])

        kwargs = {
            "in_scale": in_qparams["scale"],
            "in_zp": in_qparams["zero_point"],
            "out_scale": out_qparams["scale"],
            "out_zp": out_qparams["zero_point"],
        }
        if isinstance(submodule, torch.nn.Conv2d):
            conv_layer = submodule
            kwargs.update({"conv_module": conv_layer})
            new_module = GMLQuantConv(**kwargs)
            new_name = graph._graph_namespace.create_name(f"gml_q_conv_{cnt}", None)

        elif isinstance(submodule, nni.ConvReLU2d):
            conv_layer = submodule[0]
            kwargs.update({"conv_module": conv_layer})
            new_module = GMLQuantConvReLU(**kwargs)
            new_name = graph._graph_namespace.create_name(
                f"gml_q_conv_relu_{cnt}", None
            )

        gm.add_submodule(new_name, new_module)

        with gm.graph.inserting_after(node):
            new_node = graph.call_module(new_name, args=(dq_node.args[0],), kwargs={})
            node.replace_all_uses_with(new_node)
            new_node.name = new_name
        graph.erase_node(node)
        graph.erase_node(target_node)
        graph.erase_node(dq_node)
        logger.info(
            f' The Conv/ConvReLU module "{target_node.name}" is replaced with '
            f'the new quant module "{new_node.name}"'
        )
        cnt += 1
    gm.graph.lint()
    gm.recompile()
