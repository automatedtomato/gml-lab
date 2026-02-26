from __future__ import annotations

from typing import TYPE_CHECKING

import torch.ao.nn.quantized.reference as nnqr

from src.gml_lab.kernel_class import GMLQuantFullyConnected
from src.gml_lab.logger import get_logger
from src.gml_lab.utils import is_dequant_node, is_per_tensor_quant_node

from .utils import extract_qparams, remove_unused_nodes

if TYPE_CHECKING:
    from torch.fx import GraphModule


def lower_linear(gm: GraphModule) -> None:
    """Find `DQ -> Linear -> Q` pattern and convert to custom kernel module."""
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

        if not isinstance(submodule, nnqr.Linear):
            continue

        dq_node = target_node.args[0]
        if not is_dequant_node(dq_node):
            continue

        out_qparams = extract_qparams(gm, node)
        in_qparams = extract_qparams(gm, dq_node.args[0])
        w_scale = submodule.weight_scale
        w_zp = submodule.weight_zero_point
        weight_scale = w_scale.item() if w_scale.numel() == 1 else w_scale.tolist()
        weight_zp = w_zp.item() if w_zp.numel() == 1 else w_zp.tolist()

        kwargs = {
            "weight": submodule.weight,
            "bias": getattr(submodule, "bias", None),
            "weight_scale": weight_scale,
            "weight_zp": weight_zp,
            "input_scale": in_qparams["output_scale"],
            "input_zp": in_qparams["output_zp"],
            "output_scale": out_qparams["output_scale"],
            "output_zp": out_qparams["output_zp"],
        }
        new_module = GMLQuantFullyConnected(**kwargs)
        new_name = graph._graph_namespace.create_name(f"gml_q_fc_{cnt}", None)
        gm.add_submodule(new_name, new_module)

        with gm.graph.inserting_after(node):
            new_node = graph.call_module(new_name, args=(dq_node.args[0],), kwargs={})
            node.replace_all_uses_with(new_node)
            new_node.name = new_name
        remove_unused_nodes(graph, [node, target_node, dq_node])
        logger.info(
            f' The Linear module "{target_node.name}" is replaced with '
            f'the new quant module "{new_node.name}"'
        )
        cnt += 1
    gm.graph.lint()
    gm.recompile()
