import time
from collections import OrderedDict
from datetime import datetime

import mmengine
import torch
from codecarbon import OfflineEmissionsTracker
from torch.utils.data import DataLoader

from .modeling import build_mm_model


def evaluate(  # noqa: PLR0913
    cfg: mmengine.config.Config,
    model_arch: str,
    warmup_rounds: int,
    target_model: torch.nn.Module | torch.fx.GraphModule,
    target_type: str,
    test_loader: DataLoader,
    data_preprocessor: torch.nn.Module,
    device: str,
    seed: int,
) -> dict[str, float]:
    """Get metrics from the target_model."""
    mm_model = build_mm_model(target_model, cfg)

    runner = mmengine.runner.Runner(
        model=mm_model,
        work_dir=cfg.work_dir,
        test_dataloader=test_loader,
        test_evaluator=cfg.test_evaluator,
        test_cfg=cfg.test_cfg,
        default_scope=cfg.default_scope,
        default_hooks=cfg.get("default_hooks", None),
    )

    tracker = OfflineEmissionsTracker(
        project_name=f"GML-Lab-{target_type}",
        experiment_id=seed,
        country_iso_code="JPN",
        output_dir=f"work_dirs/{model_arch}",
    )

    data_preprocessor = data_preprocessor.to(device)
    org_input = next(iter(test_loader))
    warmup_input = data_preprocessor(org_input, training=False)["inputs"]
    warmup_input = warmup_input.to(device)
    with torch.no_grad():
        for _ in range(warmup_rounds):
            _ = target_model(warmup_input)

    torch.cuda.synchronize()
    tracker.start_task(f"{target_type}")
    start_time = time.time()

    try:
        metrics = runner.test()
    finally:
        torch.cuda.synchronize()
        end_time = time.time()
        em = tracker.stop_task(f"{target_type}")
        tracker.stop()

    dur = end_time - start_time
    carbon_em = em.emissions * 1000
    total_power = em.energy_consumed * 1000
    avg_power = total_power / (dur / 3600)

    print(
        f"[Result] {target_type}_model: {metrics}, {dur:.2f} sec, "
        f"emissions={carbon_em:4f} g, total_power={total_power:4f} Wh, "
        f"avg_power={avg_power:2f} W"
    )
    metrics = {k: v for k, v in metrics.items() if k not in ["date_time", "time"]}
    ret = {"time": datetime.now(), "type": target_type, "warmup rounds": warmup_rounds}
    ret.update(metrics)
    ret.update(
        {
            "duration": dur,
            "carbon_emission_g": carbon_em,
            "total_energy_wh": total_power,
            "avg_power_w": avg_power,
        }
    )
    return OrderedDict(ret)
