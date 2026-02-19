from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.gml_lab.quantizer import (
    build_qconfig_mapping,
    gml_convert_fx,
    gml_prepare_fx,
)
from tools.visualize_graph import dump_graph

if TYPE_CHECKING:
    from torch.fx import GraphModule
    from torch.nn import Module

NO_GPU = not torch.cuda.is_available()

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min

SNR_THRESH = 45

save_test_results = os.getenv("SAVE_TEST_RESULTS", None) is not None


def get_model_size(model: GraphModule | Module) -> float:
    """Calculate size (MB) by serializing."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def get_test_output_dir(test_name: str, file_loc: str) -> Path:
    """Return directory name based on the file location and test name."""
    name = test_name.replace("[", "_").replace("]", "")
    return Path(file_loc).parent / "results" / Path(name)


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
    # expected_nodes: list[?],
    original_model_traceable: bool = True,  # noqa: FBT001, FBT002
) -> float:
    """Run quantizer test suite."""
    # TODO: Add node matching check ans scatter plot logic
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(iter(example_inputs)).device

    prepared_model, qdq_model = quantize_model(
        float_model, example_inputs, device=device
    )

    if save_test_results:
        torch.save(float_model.state_dict(), out_dir / "float_model.pth")
        torch.save(prepared_model.state_dict(), out_dir / "prepared_model.pth")
        torch.save(qdq_model.state_dict(), out_dir / "qdq_model.pth")
        if original_model_traceable:
            dump_graph(torch.fx.symbolic_trace(float_model), "float_model", out_dir)
        dump_graph(prepared_model, "prepared_model", out_dir)
        dump_graph(qdq_model, "qdq_model", out_dir)

    out_org = float_model(*example_inputs)
    out_target = qdq_model(*example_inputs)

    snr = _calc_blob_snr(out_org, out_target)
    with open(out_dir / "snr.txt", "w") as f:
        f.write(str(snr))
    return snr
