from __future__ import annotations

from pathlib import Path

import mmengine
import mmpretrain

from .logger import get_logger

logger = get_logger("config_builder")

DATA_CONFIGS = {
    "imagenet_lmdb": "configs/imagenet_lmdb.py",
}


def build_config(
    model_arch: str, data_setting: str, batch_size: int = 64
) -> mmengine.config.Config:
    """Merge given model and data config, return mmengine.config.Config."""
    kwds = model_arch.replace("_", "-").split("-")[:3]
    checkpoints = [m for m in mmpretrain.list_models() if all(k in m for k in kwds)]
    for ch in checkpoints:
        info = mmpretrain.ModelHub.get(ch)
        if not info.weights:
            checkpoints.remove(ch)
    model_info = mmpretrain.ModelHub.get(checkpoints[0])
    config_path = (
        Path(mmpretrain.__file__).parent / ".mim" / model_info.full_model.config
    )

    if not config_path.exists():
        msg = f"Config file not found at: {config_path}"
        raise FileNotFoundError(msg)

    cfg = mmengine.config.Config.fromfile(str(config_path))

    if not model_info.weights:
        logger.warning(f"No checkpoint url found in ModelHub for {model_arch}.")
    cfg.load_from = model_info.weights

    logger.info(f"[model] name={model_arch}, weights={model_info.weights}")

    if data_setting not in DATA_CONFIGS:
        msg = f"Model `{data_setting}` not found in data configs "
        raise ValueError(msg)

    data_cfg = mmengine.config.Config.fromfile(DATA_CONFIGS[data_setting])
    logger.info(f"[data] name={data_setting}, config_file={DATA_CONFIGS[data_setting]}")

    cfg.merge_from_dict(data_cfg.to_dict())

    cfg.work_dir = f"work_dirs/{model_arch}"
    cfg.val_dataloader.batch_size = batch_size
    cfg.test_dataloader.batch_size = batch_size

    return cfg


if __name__ == "__main__":
    # Fucntional test
    try:
        cfg = build_config("resnet18_8xb32_in1k", "imagenet_lmdb")
        print(f"Model Type: {cfg.model.type}")  # exp: ImageClassifier
        print(f"Dataset Type: {cfg.val_dataloader.dataset.type}")  # exp: ImageNetLMDB
    except Exception as e:
        print(f"Error: {e}")
