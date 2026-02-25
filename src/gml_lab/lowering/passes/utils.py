from __future__ import annotations

from typing import Any

import torch

from src.gml_lab.kernel_class.gml_quant_base import GMLQuantModuleBase


def extract_qparams(gm: torch.fx.GraphModule, node: torch.fx.Node) -> dict[str, Any]:
    """Extract qparams from node and return dict of qparams."""
    qparams = {}
    if node.op == "call_module":
        submod = gm.get_submodule(str(node.target))
        if isinstance(submod, GMLQuantModuleBase):
            qparams.update(
                {
                    "output_scale": submod.output_scale.item(),
                    "output_zp": submod.output_zp.item(),
                }
            )
            return qparams

        msg = f"Expected GMLQuantModuleBase. Got {type(submod)} for node `{node.name}`"
        raise ValueError(msg)

    if node.op == "call_function":
        qparams.update(
            {
                "output_scale": getattr(gm, node.args[1].target).item(),
                "output_zp": getattr(gm, node.args[2].target).item(),
            }
        )
        if node.target == torch.quantize_per_channel:
            qparams.update({"axis": node.args[3]})
        return qparams

    msg = f"Unsupported node for extraction: op={node.op}, target={node.target}"
    raise ValueError(msg)


def remove_unused_nodes(graph: torch.fx.Graph, node_list: list[torch.fx.Node]) -> None:
    """Safely erase nodes from the graph if they have no users.

    NOTE: The "nodes" list should be ordered from downstream (outputs)
        to upstream (inputs) to allow cascading deletion.
    """
    for node in node_list:
        if len(node.users) == 0:
            graph.erase_node(node)
