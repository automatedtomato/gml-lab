from pathlib import Path

import mmpretrain

__all__ = [
    "DATA_CONFIGS",
    "MODEL_ZOO",
]

BASE = Path(mmpretrain.__file__).parent / ".mim" / "configs"

MODEL_ZOO = {
    "resnet18_8xb32_in1k": f"{BASE}/resnet/resnet18_8xb32_in1k.py",
    "vit-base-p16_64xb64_in1k": (
        f"{BASE}/vision_transformer/vit-base-p16_64xb64_in1k.py"
    ),
}

DATA_CONFIGS = {
    "imagenet_lmdb": "configs/imagenet_lmdb.py",
}
