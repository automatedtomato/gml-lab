from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.gml_lab.kernel_class import GMLQuantLUT
from src.gml_lab.logger import get_logger
from src.gml_lab.utils import is_dequant_node, is_per_tensor_quant_node

from .utils import extract_qparams, remove_unused_nodes

LUT = {
    torch.nn.GELU: "gelu",
}

if TYPE_CHECKING:
    from torch.fx import GraphModule


def lower_lut_act(gm: GraphModule) -> None:
    """Find `DQ -> LUT activation -> Q` pattern and convert to custom kernel module."""
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
        lut = LUT.get(type(submodule))
        if lut is None:
            continue

        dq_node = target_node.args[0]
        if not is_dequant_node(dq_node):
            continue

        in_qparams = extract_qparams(gm, node)
        out_qparams = extract_qparams(gm, dq_node.args[0])

        kwargs = {
            "input_scale": in_qparams["output_scale"],
            "input_zp": in_qparams["output_zp"],
            "output_scale": out_qparams["output_scale"],
            "output_zp": out_qparams["output_zp"],
        }
        new_module = GMLQuantLUT(**kwargs, lut=lut)
        new_name = graph._graph_namespace.create_name(f"gml_q_lut_{cnt}", None)
        gm.add_submodule(new_name, new_module)

        with gm.graph.inserting_after(node):
            new_node = graph.call_module(new_name, args=(dq_node.args[0],), kwargs={})
            new_node.meta["source_module"] = target_node.target
            node.replace_all_uses_with(new_node)
            new_node.name = new_name
        remove_unused_nodes(graph, [node, target_node, dq_node])
        logger.info(
            f' The LUT activation module "{target_node.name}" is replaced with '
            f'the new quant module "{new_node.name}"'
        )
        cnt += 1
    gm.graph.lint()
    gm.recompile()
