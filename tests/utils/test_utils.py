from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.gml_lab.lowering.lower_to_gml import lower_to_gml
from src.gml_lab.quantizer import (
    get_gml_backend_config,
    get_gml_qconfig_mapping,
    gml_convert_fx,
    gml_prepare_fx,
)
from tools.visualize_graph import dump_graph

from .node_info import NodeInfo, get_node_info
from .test_logger import get_logger

if TYPE_CHECKING:
    from torch.fx import GraphModule
    from torch.nn import Module

NO_GPU = not torch.cuda.is_available()

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min

SNR_THRESH = 50.0

save_test_results = os.getenv("SAVE_TEST_RESULTS", None) is not None


def generate_scatter(
    out_ref: torch.Tensor,
    out_target: GraphModule,
    out_dir: Path,
) -> None:
    """Compare results of target and reference models and draw scatter plot."""
    out_ref_flat = out_ref.detach().cpu().numpy().flatten()
    out_target_flat = out_target.detach().cpu().numpy().flatten()
    assert len(out_ref_flat) == len(out_target_flat)
    diff = out_target_flat != out_ref_flat
    plt.plot(out_ref_flat[diff], out_target_flat[diff], ".", alpha=0.5)
    plt.axline((0, 0), slope=1.0, color="red")
    plt.xlabel("ref")
    plt.ylabel("target")
    plt.savefig(out_dir / "plt.png")
    plt.close()


def calc_blob_snr(blob_org: torch.Tensor, blob_target: torch.Tensor) -> float:
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
    example_inputs: tuple[torch.Tensor, ...],
    method: str = "per_tensor",
    device: str = "cpu",
) -> tuple[GraphModule, ...]:
    """Quantize model."""
    float_model = float_model.eval()
    backend_config = get_gml_backend_config()
    qconfig_mapping = get_gml_qconfig_mapping(method)

    example_inputs = tuple(i.to(device) for i in example_inputs)

    prepared_model = gml_prepare_fx(
        float_model, example_inputs, qconfig_mapping, backend_config
    )
    example_inputs = tuple(i.to(device) for i in example_inputs)

    prepared_model.eval().to(device)
    with torch.no_grad():
        _ = prepared_model(*example_inputs)

    qdq_model = gml_convert_fx(prepared_model, qconfig_mapping, backend_config)

    return prepared_model.eval(), qdq_model.eval()


def run_quantizer_test(
    float_model: Module,
    example_inputs: tuple[torch.Tensor, ...],
    test_mode: Literal["unify_pass", "quant_acc", "lower_acc"],
    out_dir: Path | None = None,
    expected_nodes: list[NodeInfo] | None = None,
    original_model_traceable: bool = True,  # noqa: FBT001, FBT002
    device: str = "cpu",
) -> float | None:
    """Run quantizer test suite."""
    logger = get_logger("run_quantizer_test")
    float_model.eval()
    example_inputs = tuple(i.to(device) for i in example_inputs)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    float_model = float_model.to(device)

    prepared_model, qdq_model = quantize_model(
        float_model, example_inputs, device=device
    )

    if test_mode == "unify_pass" and expected_nodes is not None:
        check_graph_structure(prepared_model, expected_nodes)
        return None

    if test_mode == "quant_acc":
        out_target = qdq_model(*example_inputs)

    if test_mode == "lower_acc":
        gml_model = lower_to_gml(qdq_model)
        if expected_nodes is not None:
            check_graph_structure(gml_model, expected_nodes)
        out_target = gml_model(*example_inputs)

    out_org = float_model(*example_inputs)

    if save_test_results and out_dir is not None:
        try:
            torch.save(float_model.state_dict(), out_dir / "float_model.pth")
            torch.save(prepared_model.state_dict(), out_dir / "prepared_model.pth")
            torch.save(qdq_model.state_dict(), out_dir / "qdq_model.pth")
            torch.save(gml_model.state_dict(), out_dir / "gml_model.pth")
            if original_model_traceable:
                dump_graph(torch.fx.symbolic_trace(float_model), "float_model", out_dir)
            dump_graph(prepared_model, "prepared_model", out_dir)
            dump_graph(qdq_model, "qdq_model", out_dir)
            dump_graph(gml_model, "gml_model", out_dir)
            generate_scatter(out_org, out_target, out_dir)
        except Exception as e:
            logger.error(f"Error while save test results: {e}")
    snr = calc_blob_snr(out_org, out_target)
    if out_dir is not None:
        with open(out_dir / "snr.txt", "w") as f:
            f.write(str(snr))
    return snr
