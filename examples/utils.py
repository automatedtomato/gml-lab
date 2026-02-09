import os
import time

import torch


def set_seed() -> None:
    """Set seed for torch."""
    seed = int(os.getenv("SET_SEED", time.time_ns()))
    torch.manual_seed(seed)
