from __future__ import annotations

from typing import Any

import torch


def extract_qparams(gm: torch.fx.GraphModule, node: torch.fx.Node) -> dict[str, Any]:
    """Extract qparams from node and return dict of qparams."""
    qparams = {
        "scale": getattr(gm, node.args[1].target).item(),
        "zero_point": getattr(gm, node.args[2].target).item(),
    }
    if node.target == torch.quantize_per_channel:
        qparams.update({"axis": node.args[3]})
    return qparams
