from __future__ import annotations

from typing import TYPE_CHECKING

import src.gml_lab.nn.fused_modules as fcnn
import src.gml_lab.nn.modules as cnn
from src.gml_lab.kernel_class import GMLQuantAdd, GMLQuantAddReLU
from src.gml_lab.logger import get_logger
from src.gml_lab.utils import is_dequant_node, is_per_tensor_quant_node

from .utils import extract_qparams, remove_unused_nodes

if TYPE_CHECKING:
    from torch.fx import GraphModule


def lower_add(gm: GraphModule) -> None:
    """Find `DQ -> Add (-> ReLU) -> Q` pattern and convert to custom kernel module."""
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
        if not isinstance(submodule, cnn.Add | fcnn.AddReLU):
            continue
        dq_a = target_node.args[0]
        dq_b = target_node.args[1]
        if not is_dequant_node(dq_a) or not is_dequant_node(dq_b):
            continue
        out_qparams = extract_qparams(gm, node)
        a_qparams = extract_qparams(gm, dq_a.args[0])
        b_qparams = extract_qparams(gm, dq_b.args[0])

        kwargs = {
            "input_scale_a": a_qparams["output_scale"],
            "input_za": a_qparams["output_zp"],
            "input_scale_b": b_qparams["output_scale"],
            "input_zb": b_qparams["output_zp"],
            "output_scale": out_qparams["output_scale"],
            "output_zp": out_qparams["output_zp"],
        }

        if isinstance(submodule, cnn.Add):
            new_module = GMLQuantAdd(**kwargs)
            new_name = graph._graph_namespace.create_name(f"gml_q_add_{cnt}", None)
        elif isinstance(submodule, fcnn.AddReLU):
            new_module = GMLQuantAddReLU(**kwargs)  # type: ignore
            new_name = graph._graph_namespace.create_name(f"gml_q_add_relu_{cnt}", None)

        gm.add_submodule(new_name, new_module)

        with gm.graph.inserting_after(node):
            new_node = graph.call_module(
                new_name, args=(dq_a.args[0], dq_b.args[0]), kwargs={}
            )
            node.replace_all_uses_with(new_node)
            new_node.name = new_name
        remove_unused_nodes(graph, [node, target_node, dq_a, dq_b])
        logger.info(
            f' The Add/AddReLU module "{target_node.name}" is replaced with '
            f'the new quant module "{new_node.name}"'
        )
        cnt += 1
    gm.graph.lint()
    gm.recompile()
