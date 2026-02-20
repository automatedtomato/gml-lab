from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.gml_lab.lowering.lower_to_gml import lower_to_gml
from src.gml_lab.quantizer import (
    build_qconfig_mapping,
    gml_convert_fx,
    gml_prepare_fx,
)
from tools.visualize_graph import dump_graph

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx import GraphModule, Node
    from torch.nn import Module

NO_GPU = not torch.cuda.is_available()

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min

SNR_THRESH = 45.0

save_test_results = os.getenv("SAVE_TEST_RESULTS", None) is not None


class NodeInfo:
    """Node information composed of node.op and node.target.

    Args:
        op (str): "call_method", "call_function" or "call_module"
        target (str | Callable | torch.nn.Module):
            str for "call_method"
            Callable for "call_function"
            torch.nn.Module for "call_module

    """

    def __init__(self, op: str, target: str | Callable | torch.nn.Module) -> None:
        self.op = op
        self.target = target

    @classmethod
    def call_method(cls, target: str) -> NodeInfo:
        return NodeInfo("call_method", target)

    @classmethod
    def call_function(cls, target: Callable) -> NodeInfo:
        return NodeInfo("call_function", target)

    @classmethod
    def call_module(cls, target: torch.nn.Module) -> NodeInfo:
        return NodeInfo("call_module", target)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeInfo):
            return False
        return self.op == other.op and self.target == other.target

    def __hash__(self) -> int:
        return hash(self.op, self.target)


def get_node_info(node: Node, modules: dict[str, Module]) -> NodeInfo | None:
    if node.op in ["placeholder", "output", "get_attr"]:
        node_info = None
    elif node.op == "call_module":
        target_cls = type(modules[node.target])
        node_info = NodeInfo.call_module(target_cls)
    elif node.op == "call_function":
        node_info = NodeInfo.call_function(node.target)
    elif node.op == "call_method":
        node_info = NodeInfo.call_method(node.target)
    else:
        msg = f"Node type {node.op} is not supported."
        raise TypeError(msg)
    return node_info


def get_model_size(model: GraphModule | Module) -> float:
    """Calculate size (MB) by serializing."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def get_test_output_dir(test_name: str, file_loc: str) -> Path:
    """Return directory name based on the file location and test name."""
    name = test_name.replace("[", "_").replace("]", "")
    return Path(file_loc).parent / "results" / Path(name)


def check_graph_structure(gm: GraphModule, expected_nodes: list[NodeInfo]) -> None:
    """Check if type and target of nodes in gm match expected nodes."""
    expected_nodes_cp = copy.deepcopy(expected_nodes)
    modules = dict(gm.named_modules(remove_duplicate=False))

    def _get_node_info_list_from_gm(
        gm: GraphModule,
    ) -> list[NodeInfo | None]:
        nodes = list(gm.graph.nodes)
        return [
            get_node_info(node, modules)
            for node in nodes
            if get_node_info(node, modules)
        ]

    target_nodes_info = _get_node_info_list_from_gm(gm)
    for node in gm.graph.nodes:
        node_info = get_node_info(node, modules)
        if not node_info:
            continue
        assert len(expected_nodes_cp) > 0, (
            "Unexpected nodes in test GraphModule.\n"
            f"  target nodes: {target_nodes_info}\n"
            f"  expected nodes: {expected_nodes}"
        )
        expected_node_info = expected_nodes_cp.pop(0)
        assert node_info == expected_node_info, (
            "NodeInfos are defferent:"
            f"  target node: (op={node.op}, target= "
            f"{type(modules[node.target]) if node.op == 'call_module' else node.target})\n"  # noqa: E501
            f"  expected node: (op={expected_node_info.op}, target= )"
            f"{expected_node_info.target}"
        )
    assert len(expected_nodes_cp) == 0, (
        f"Some nodes are missing.\n  target nodes: {target_nodes_info}, "
        f"  expected nodes: {expected_nodes}"
    )


def quantize_model(
    float_model: Module,
    example_inputs: tuple[torch.Tensor],
    device: str = "cpu",
) -> tuple[GraphModule, ...]:
    """Quantize model."""
    float_model = float_model.to(device).eval()
    example_inputs_on_device = tuple(i.to(device) for i in example_inputs)
    qconfig_mapping = build_qconfig_mapping()

    prepared_model = gml_prepare_fx(
        float_model, example_inputs_on_device, qconfig_mapping
    )

    prepared_model.eval()
    with torch.no_grad():
        _ = prepared_model(*example_inputs)

    qdq_model = gml_convert_fx(prepared_model)

    return prepared_model.eval(), qdq_model.eval()


def _calc_blob_snr(blob_org: torch.Tensor, blob_target: torch.Tensor) -> float:
    """Compute P-SNR between given two tensors."""
    blob_org = blob_org.cpu().detach().flatten().numpy()
    blob_target = blob_target.cpu().detach().flatten().numpy()

    blob_dif = blob_org - blob_target
    blob_max = np.max(blob_org)
    blob_min = np.min(blob_org)

    blob_amp = blob_max - blob_min
    if blob_amp == 0.0:
        return float("nan")
    blob_mse = np.mean(blob_dif**2)
    if blob_mse == 0.0:
        return float("inf")

    return -20 * np.log10(np.sqrt(blob_mse) / blob_amp)


def run_quantizer_test(
    float_model: Module,
    example_inputs: tuple[torch.Tensor],
    out_dir: Path,
    expected_nodes: list[NodeInfo] | None = None,
    original_model_traceable: bool = True,  # noqa: FBT001, FBT002
) -> float:
    """Run quantizer test suite."""
    # TODO: Add scatter plot logic
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(iter(example_inputs)).device

    prepared_model, qdq_model = quantize_model(
        float_model, example_inputs, device=device
    )

    gml_model = lower_to_gml(qdq_model)

    if save_test_results:
        torch.save(float_model.state_dict(), out_dir / "float_model.pth")
        torch.save(prepared_model.state_dict(), out_dir / "prepared_model.pth")
        torch.save(gml_model.state_dict(), out_dir / "qdq_model.pth")
        if original_model_traceable:
            dump_graph(torch.fx.symbolic_trace(float_model), "float_model", out_dir)
        dump_graph(prepared_model, "prepared_model", out_dir)
        dump_graph(gml_model, "qdq_model", out_dir)

    if expected_nodes is not None:
        check_graph_structure(gml_model, expected_nodes)

    out_org = float_model(*example_inputs)
    out_target = qdq_model(*example_inputs)

    snr = _calc_blob_snr(out_org, out_target)
    with open(out_dir / "snr.txt", "w") as f:
        f.write(str(snr))
    return snr
