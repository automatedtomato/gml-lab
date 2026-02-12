from __future__ import annotations

from typing import Any

import mmpretrain


def load_model(arch: str, pretrained: bool = True) -> Any:  # noqa: ANN401, FBT001, FBT002
    """Load model with mmpretrain apis."""
    return mmpretrain.get_model(arch, pretrained=pretrained)
