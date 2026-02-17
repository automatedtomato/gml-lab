from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

# import pandas as pd
import torch
from torch.fx import GraphModule, Node
from tqdm import tqdm

from src.gml_lab.utils import is_fake_quant_node, is_observer_node, is_quantize_node

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CompareInfo:
    """Comparison information between original and target models."""

    org_node: str
    target_node: str
    snr: float
    cos_sim: float
    input_scale: list[list[float]] | None = None
    input_zp: list[list[float]] | None = None
    output_scale: list[list[float]] | None = None
    output_zp: list[list[float]] | None = None


def compute_blob_snr(blob_org: np.ndarray, blob_target: np.ndarray) -> float:
    """Compute P-SNR between given two tensors."""
    blob_org = blob_org.astype(float)
    blob_target = blob_target.astype(float)

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


def compute_blob_cos_sim(blob_org: np.ndarray, blob_target: np.ndarray) -> float:
    """Compute cosine similarity between given two tensors."""
    blob_org = blob_org.astype(float).flatten()
    blob_target = blob_target.astype(float).flatten()

    blob_org_norm = np.linalg.norm(blob_org)
    blob_target_norm = np.linalg.norm(blob_target)

    if blob_org_norm == 0 or blob_target_norm == 0:
        return float("inf")
    return np.dot(blob_org, blob_target) / (
        np.linalg.norm(blob_org) * np.linalg.norm(blob_target)
    )


def get_quant_params_from_node_names(
    qdq_model: GraphModule,
    module_name: str,
) -> dict[str, list[list]]:
    """Get input and output scale/zp of the given node."""
    qparams: dict[str, list[list]] = {
        "input_scale": [],
        "input_zp": [],
        "output_scale": [],
        "output_zp": [],
    }
    nodes = {node.name: node for node in qdq_model.graph.nodes}
    def _get_input_scale_zp(node: Node) -> None:
        if is_quantize_node(node):
            input_scale_node = node.args[1]
            input_zp_node = node.args[2]
            assert isinstance(input_scale_node, Node)
            assert isinstance(input_zp_node, Node)
            qparams["input_scale"].append(
                torch.atleast_1d(getattr(qdq_model, input_scale_node.target)).tolist()
            )
            qparams["input_zp"].append(
                torch.atleast_1d(getattr(qdq_model, input_zp_node.target)).tolist()
            )
            return
        for arg in node.args:
            if not isinstance(arg, Node):
                continue
            _get_input_scale_zp(arg)

    target_node = nodes.get(node_name)
    if target_node is None:
        return qparams
    next_node = next(iter(target_node.users))
    if is_quantize_node(next_node):
        output_scale_node = next_node.args[1]
        output_zp_node = next_node.args[2]
        qparams["output_scale"].append(
            torch.atleast_1d(getattr(qdq_model, output_scale_node.target)).tolist()
        )
        qparams["output_zp"].append(
            torch.atleast_1d(getattr(qdq_model, output_zp_node.target)).tolist()
        )
    _get_input_scale_zp(target_node)
    return qparams


class ActivationCollector:
    """Collect output from specified layer."""

    def __init__(self, model: GraphModule, target_names: list[str]) -> None:
        self.model = model
        self.target_names = target_names
        self.outputs: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []

    def _get_hook(self, name: str) -> Callable:
        def hook(model, input, output) -> None:  # noqa: A002, ANN001, ARG001
            if isinstance(output, torch.Tensor):
                self.outputs[name] = output.detach().cpu().numpy()

        return hook

    def __enter__(self) -> None:  # noqa: D105
        self.outputs.clear()
        self.hooks = []
        for name in self.target_names:
            module = self.model.get_submodule(name)
            h = module.register_forward_hook(self._get_hook(name))
            self.hooks.append(h)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        for h in self.hooks:
            h.remove()


def get_target_layer(gm: GraphModule) -> tuple[list[str], ...]:
    """Generate list of compared layers."""
    modules = dict(gm.named_modules(remove_duplicate=False))
    target_nodes: list[str] = []
    target_names: list[str] = []
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        if node.target not in modules:
            continue
        if is_quantize_node(node):
            continue
        if is_observer_node(node, modules):
            continue
        if is_fake_quant_node(node, modules):
            continue
        target_nodes.append(node.target)
        target_names.append(node.name)
    return target_nodes, target_names


def run_sensitivity_analysis(
    prepared_model: GraphModule,
    qdq_model: GraphModule,
    analysis_input: torch.Tensor,
) -> list[CompareInfo]:
    """Run sensitivity analysis."""
    target_nodes, target_names = get_target_layer(prepared_model)

    with (
        # torch.no_grad(),
        ActivationCollector(prepared_model, target_nodes) as collector_fp32,
        ActivationCollector(qdq_model, target_nodes) as collector_qdq,
    ):
        _ = prepared_model(analysis_input)
        _ = qdq_model(analysis_input)

        outputs_fp32 = collector_fp32.outputs
        outputs_qdq = collector_qdq.outputs

    results = []

    for name in tqdm(target_names, desc="extract feat map"):

        feat_org = outputs_fp32[name]
        feat_target = outputs_qdq[name]

        snr = compute_blob_snr(feat_org, feat_target)
        cos_sim = compute_blob_cos_sim(feat_org, feat_target)

        qparams = get_quant_params_from_node_names(qdq_model, name)

        info = CompareInfo(
            org_node=name,
            target_node=name,
            snr=snr,
            cos_sim=cos_sim,
            input_scale=qparams.get("input_scale"),
            input_zp=qparams.get("input_zp"),
            output_scale=qparams.get("output_scale"),
            output_zp=qparams.get("output_zp"),
        )
        results.append(info)

    return results
