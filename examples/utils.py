from __future__ import annotations

import os
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mmengine.runner import Runner
from mmpretrain.utils import register_all_modules

from src.gml_lab.modeling import FxWrapper
from src.gml_lab.quantizer import (
    calibrate_model,
    gml_convert_fx,
    gml_prepare_fx,
)
from tools.profiler import FxProfiler

if TYPE_CHECKING:
    from pathlib import Path

    import mmengine
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.utils.data import DataLoader


def set_seed() -> int:
    """Set seed for reproducibility.

    Reference:
        https://github.com/open-mmlab/mmengine/blob/v0.10.1/mmengine/runner/utils.py#L47
    """
    seed = int(os.getenv("SET_SEED", time.time_ns()))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    return seed


def prepare_dataloader(cfg: mmengine.config.Config) -> tuple[DataLoader, ...]:
    """Prepare test and calib data loaders."""
    register_all_modules(init_default_scope=True)
    test_loader = Runner.build_dataloader(cfg.test_dataloader)
    calib_loader = Runner.build_dataloader(cfg.test_dataloader)
    return test_loader, calib_loader


def quantize(
    float_model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    qconfig_mapping: QConfigMapping | dict[str, Any],
    calib_loader: Any,  # noqa: ANN401
    total_calib_batches: int,
    data_preprocessor: torch.nn.Module,
    fake_quantize: bool = True,  # noqa: FBT001, FBT002
) -> tuple[torch.fx.GraphModule, ...]:
    """Quantize model."""
    wrapped_model = FxWrapper(float_model)
    prepared_model = gml_prepare_fx(wrapped_model, example_inputs, qconfig_mapping)
    prepared_model = calibrate_model(
        prepared_model,
        calib_loader=calib_loader,
        data_preprocessor=data_preprocessor,
        total_calib_batches=total_calib_batches,
    )
    qdq_model = gml_convert_fx(prepared_model, qconfig_mapping, None, fake_quantize)

    return prepared_model, qdq_model


def perf_profile(
    model: torch.nn.Module | torch.fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...],
    save_path: str | Path | None = None,
) -> None:
    """Perform profiling."""
    if isinstance(model, torch.nn.Module):  # Assume float_model
        float_model = FxWrapper(model)
        gm = torch.fx.symbolic_trace(float_model)
    if isinstance(model, torch.fx.GraphModule):
        gm = model
    profiler = FxProfiler(gm)
    profiler.run(*example_inputs)
    if save_path is not None:
        profiler.dump_to_json(save_path)
    profiler.print_summary()
