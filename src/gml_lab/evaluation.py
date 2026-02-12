import time

import mmengine
from codecarbon import OfflineEmissionsTracker

from .model_builder import FxWrapper, MMLabWrapper, load_model


def evaluate(
    cfg: mmengine.config.Config,
    *,
    model_arch: str,
    target_type: str,
    seed: int,
) -> dict[str, float]:
    """Get metrics from the target_model."""
    model = load_model(model_arch, pretrained=True)
    data_preprocessor_cfg = cfg.get("data_preprocessor", {}).copy()
    if "type" not in data_preprocessor_cfg:
        data_preprocessor_cfg["type"] = "ImgDataPreprocessor"
    data_preprocessor_cfg.pop("num_classes", None)
    if "to_rgb" in data_preprocessor_cfg:
    # 値を取り出しつつ削除し、新しいキーで入れ直す
        is_to_rgb = data_preprocessor_cfg.pop("to_rgb")
        data_preprocessor_cfg["bgr_to_rgb"] = is_to_rgb

    wrapped_model = FxWrapper(model)
    mm_model = MMLabWrapper(wrapped_model, data_preprocessor=data_preprocessor_cfg)

    runner = mmengine.runner.Runner(
        model=mm_model,
        work_dir=cfg.work_dir,
        test_dataloader=cfg.test_dataloader,
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

    tracker.start_task(f"{target_type}")
    start_time = time.time()

    try:
        metrics = runner.test()
    finally:
        end_time = time.time()
        em = tracker.stop_task(f"{target_type}")
        tracker.stop()

    dur = end_time - start_time
    carbon_em = em.emissions * 1000
    total_power = em.energy_consumed * 1000
    avg_power = total_power / (dur / 3600)

    print(
        f"[Result] {metrics}, {dur:.2f} sec, emissions={carbon_em:4f} g, "
        f"total_power={total_power:4f} Wh, avg_power={avg_power:2f} W"
    )
    metrics.update(
        {
            "duration": dur,
            "carbon_emission_g": carbon_em,
            "total_energy_wh": total_power,
            "avg_power_w": avg_power,
        }
    )
    return metrics
