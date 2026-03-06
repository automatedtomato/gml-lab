from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.fx import GraphModule, Interpreter, Node
from tqdm import tqdm

from src.gml_lab.utils import is_fake_quant_node, is_observer_node, is_quantize_node


@dataclass
class CompareInfo:
    """Comparison information between original and target models."""

    org_node: str
    target_node: str
    snr: float
    cos_sim: float
    feat_org: np.ndarray | None = field(default=None, repr=False)
    feat_target: np.ndarray | None = field(default=None, repr=False)
    input_scale: list[list[float]] | None = None
    input_zp: list[list[float]] | None = None
    output_scale: list[list[float]] | None = None
    output_zp: list[list[float]] | None = None
    plot_path: Path | None = None


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


def _extract_fq_params(qdq_model: GraphModule, node: Node) -> tuple[Any, Any] | None:
    if node.op == "call_module":
        try:
            mod = qdq_model.get_submodule(node.target)
            if isinstance(mod, FakeQuantize):
                return mod.scale, mod.zero_point
        except AttributeError:
            pass
    return None


def _get_input_scale_zp(
    qdq_model: GraphModule, qparams: dict[str, Any], node: Node
) -> None:
    fq_params = _extract_fq_params(qdq_model, node)
    if fq_params is not None:
        scale, zp = fq_params
        qparams["input_scale"].append(torch.atleast_1d(scale).tolist())
        qparams["input_zp"].append(torch.atleast_1d(zp).tolist())
        return

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
        if isinstance(arg, Node):
            _get_input_scale_zp(qdq_model, qparams, arg)


def get_quant_params_from_node_names(
    qdq_model: GraphModule,
    node_name: str,
) -> dict[str, list[list]]:
    """Get input and output scale/zp of the given node."""
    qparams: dict[str, list[list]] = {
        "input_scale": [],
        "input_zp": [],
        "output_scale": [],
        "output_zp": [],
    }

    target_node = None
    for node in qdq_model.graph.nodes:
        if node.op == "call_module" and node.target == node_name:
            target_node = node
            break

    if target_node is None:
        return qparams

    next_node = next(iter(target_node.users)) if target_node.users else None
    if next_node:
        fq_params = _extract_fq_params(qdq_model, next_node)
        if fq_params is not None:
            scale, zp = fq_params
            qparams["output_scale"].append(torch.atleast_1d(scale).tolist())
            qparams["output_zp"].append(torch.atleast_1d(zp).tolist())
        elif is_quantize_node(next_node):
            output_scale_node = next_node.args[1]
            output_zp_node = next_node.args[2]
            qparams["output_scale"].append(
                torch.atleast_1d(getattr(qdq_model, output_scale_node.target)).tolist()
            )
            qparams["output_zp"].append(
                torch.atleast_1d(getattr(qdq_model, output_zp_node.target)).tolist()
            )

    _get_input_scale_zp(qdq_model, qparams, target_node)
    return qparams


class FxActivationCollector(Interpreter):
    """Collect output using Fx Interpreter based on meta tags or node targets."""

    def __init__(self, model: GraphModule, target_modules: list[str]) -> None:
        super().__init__(model)
        self.target_modules = target_modules
        self.outputs: dict[str, np.ndarray] = {}

    def run_node(self, node: Node) -> Any:  # noqa: ANN401
        """Override FxInterpreter.run_node()."""
        result = super().run_node(node)

        module_key = node.meta.get("source_module", None)
        if module_key is None and node.op == "call_module":
            module_key = str(node.target)
        if module_key in self.target_modules:
            val = result
            if isinstance(val, tuple | list) and len(val) > 0:
                val = val[0]

            if isinstance(val, torch.Tensor):
                if val.is_quantized:
                    val = val.dequantize()
                self.outputs[module_key] = val.detach().cpu().numpy()
        return result


def get_target_layer(gm: GraphModule) -> tuple[list[str], list[str]]:
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
    ref_model: GraphModule,
    target_model: GraphModule,
    analysis_input: torch.Tensor,
) -> list[CompareInfo]:
    """Run sensitivity analysis."""
    target_nodes, _ = get_target_layer(ref_model)

    collector_ref = FxActivationCollector(ref_model, target_nodes)
    collector_target = FxActivationCollector(target_model, target_nodes)
    with torch.no_grad():
        collector_ref.run(analysis_input)
        collector_target.run(analysis_input)

    outputs_ref = collector_ref.outputs
    outputs_target = collector_target.outputs

    results = []

    for name in tqdm(target_nodes, desc="Extracting feat map"):
        if name not in outputs_ref or name not in outputs_target:
            continue

        feat_org = outputs_ref[name]
        feat_target = outputs_target[name]

        snr = compute_blob_snr(feat_org, feat_target)
        cos_sim = compute_blob_cos_sim(feat_org, feat_target)

        qparams = get_quant_params_from_node_names(target_model, name)

        info = CompareInfo(
            org_node=name,
            target_node=name,
            feat_org=feat_org,
            feat_target=feat_target,
            snr=snr,
            cos_sim=cos_sim,
            input_scale=qparams.get("input_scale"),
            input_zp=qparams.get("input_zp"),
            output_scale=qparams.get("output_scale"),
            output_zp=qparams.get("output_zp"),
        )
        results.append(info)

    return results


def _plot_layer_scatter(
    feat_org: np.ndarray,
    feat_target: np.ndarray,
    layer_name: str,
    save_path: Path,
) -> None:
    """Plot scatter by layer."""
    if feat_org is None or feat_target is None:
        return
    x = feat_org
    y = feat_target

    plt.figure(figsize=(6, 6))

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="ideal")

    plt.scatter(x, y, alpha=0.3, s=5, c="tab:blue")

    plt.title(f"Layer: {layer_name}")
    plt.xlabel("Float Output")
    plt.ylabel("Quant Output")
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    plt.grid(True, alpha=0.3)  # noqa: FBT003
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


def _plot_overview(results: list[CompareInfo], save_path: Path) -> None:
    """Generate overview graph of SNR/CosSim."""
    data = [
        {"layer": r.target_node, "snr": r.snr, "cos_sim": r.cos_sim} for r in results
    ]
    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan)

    fig, ax1 = plt.subplots(figsize=(12, 6))  # noqa: RUF059

    # SNR
    color = "tab:blue"
    ax1.set_xlabel("layers")
    ax1.set_ylabel("SNR (dB)", color=color)
    ax1.plot(
        df.index,
        df["snr"],
        color=color,
        marker="o",
        markersize=4,
        linestyle="-",
        label="SNR",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.3)  # noqa: FBT003

    # CosSim
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("cos_sim", color=color)
    ax2.plot(
        df.index,
        df["cos_sim"],
        color=color,
        marker="x",
        markersize=4,
        linestyle="--",
        label="Cos Sim",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    layers = df["layer"].tolist()
    step = max(1, len(layers) // 20)
    ax1.set_xticks(df.index[::step])
    ax1.set_xticklabels(layers[::step], rotation=45, ha="right", fontsize=8)

    plt.title("Quantization Sensitivity Overview")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_analysis_report(
    ref_model: GraphModule,
    target_model: GraphModule,
    analysis_input: torch.Tensor,
    dump_dir: str | Path,
) -> None:
    """Recieve results and generate analysis report."""
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dump_dir / "images"
    images_dir.mkdir(exist_ok=True)

    results = run_sensitivity_analysis(ref_model, target_model, analysis_input)

    md_lines = []
    md_lines.append("# Quantization Sensitivity Report")
    md_lines.append("")

    overview_path = images_dir / "overview.png"
    _plot_overview(results, overview_path)

    rel_overview_path = f"images/{overview_path.name}"
    md_lines.append("## Overview")
    md_lines.append(f"![Overview]({rel_overview_path})")
    md_lines.append("")

    md_lines.append("## Layer-wise Details")

    for i, info in enumerate(tqdm(results, desc="Generating plots")):
        layer_name = info.target_node
        safe_name = layer_name.replace(".", "_")
        scatter_filename = f"{i}_{safe_name}.png"
        scatter_path = images_dir / scatter_filename

        if info.feat_org is not None and info.feat_target is not None:
            _plot_layer_scatter(
                info.feat_org, info.feat_target, layer_name, scatter_path
            )

        md_lines.append(f"### {i}. {layer_name}")

        snr_str = f"{info.snr:.2f}" if info.snr != float("inf") else "Inf"
        md_lines.append("| metric | value |")
        md_lines.append("| :--- | :--- |")
        md_lines.append(f"| **SNR** | {snr_str} dB |")
        md_lines.append(f"| **cos_sim** | {info.cos_sim:.6f} |")
        md_lines.append("")

        if scatter_path.exists():
            md_lines.append(f"![Scatter](images/{scatter_filename})")

        md_lines.append("")

        md_lines.append("<details>")
        md_lines.append("<summary>scales & zero points</summary>")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(f"Input Scale:  {info.input_scale}")
        md_lines.append(f"Input ZP:     {info.input_zp}")
        md_lines.append(f"Output Scale: {info.output_scale}")
        md_lines.append(f"Output ZP:    {info.output_zp}")
        md_lines.append("```")
        md_lines.append("</details>")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    report_path = dump_dir / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
