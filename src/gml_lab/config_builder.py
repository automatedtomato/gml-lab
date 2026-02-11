from pathlib import Path

import mmengine
import mmpretrain

from .logger import get_logger

logger = get_logger("config_builder")


BASE = Path(mmpretrain.__file__).parent / ".mim" / "configs"

MODEL_ZOO = {
    "resnet18_8xb32_in1k": f"{BASE}/resnet/resnet18_8xb32_in1k.py",
}

DATA_CONFIGS = {
    "imagenet_lmdb": "configs/imagenet_lmdb.py",
}


def build_config(model_arch: str, data_setting: str) -> mmengine.config.Config:
    """Merge given model and data config, return mmengine.config.Config."""
    if model_arch not in MODEL_ZOO:
        msg = (
            f"Model `{model_arch}` not found in model zoo "
            "specified in `src/gml_lab/config_builder`."
        )
        raise ValueError(msg)
    if data_setting not in DATA_CONFIGS:
        msg = (
            f"Model `{data_setting}` not found in data configs "
            "specified in `src/gml_lab/config_builder`."
        )
        raise ValueError(msg)
    model_cfg = mmengine.config.Config.fromfile(MODEL_ZOO[model_arch])
    logger.info(f"[model] name={model_arch}, config_file={MODEL_ZOO[model_arch]}.")

    data_cfg = mmengine.config.Config.fromfile(DATA_CONFIGS[data_setting])
    logger.info(f"[data] name={data_setting}, config_file={DATA_CONFIGS[data_setting]}")

    model_cfg.merge_from_dict(data_cfg.to_dict())

    return model_cfg


if __name__ == "__main__":
    # Fucntional test
    try:
        cfg = build_config("resnet18_8xb32_in1k", "imagenet_lmdb")
        print(f"Model Type: {cfg.model.type}")  # exp: ImageClassifier
        print(f"Dataset Type: {cfg.val_dataloader.dataset.type}")  # exp: ImageNetLMDB
    except Exception as e:
        print(f"Error: {e}")
