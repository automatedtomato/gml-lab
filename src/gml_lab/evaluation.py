import time

import mmengine


def evaluate(cfg: mmengine.config.Config) -> None:
    """Get metrics from the target_model."""
    runner = mmengine.runner.Runner.from_cfg(cfg)
    start = time.time()
    metrics = runner.test()
    end = time.time()
    print(f"[Result] {metrics}, {end - start:.2f} sec")
