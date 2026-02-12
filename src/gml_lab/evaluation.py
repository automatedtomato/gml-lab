import time

import mmengine
from codecarbon import OfflineEmissionsTracker


def evaluate(
    cfg: mmengine.config.Config,
    *,
    model_arch: str,
    target_type: str,
    seed: int,
) -> dict[str, float]:
    """Get metrics from the target_model."""
    runner = mmengine.runner.Runner.from_cfg(cfg)

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
