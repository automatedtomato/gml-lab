from typing import Any

import torch
from torch.fx import GraphModule
from tqdm import tqdm


def calibrate_model(
    model: GraphModule,
    *,
    data_preprocessor: torch.nn.Module,
    calib_loader: Any,  # noqa: ANN401
    total_calib_batches: int,
    device: str = "cpu",
) -> GraphModule:
    """Run calibration to collect qparams.

    Args:
        model (torch.fx.GraphModule): prepared model.
        data_preprocessor (torch.nn.Module): data preprocessor.
        calib_loader (Any): iterable data loader.
        total_calib_batches (int): num of total batches for for calibration.
        device (str): einputsecution device.

    Returns:
        torch.fx.GraphModule: The calibrated model (with populated observers).

    """
    model.to(device)
    model.eval()
    data_preprocessor = data_preprocessor.to(device)
    data_preprocessor.eval()

    pbar = tqdm(calib_loader, desc="calibrate mdoel", total=total_calib_batches)
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if i >= total_calib_batches:
                break
            data = data_preprocessor(batch, training=False)
            if isinstance(data, dict):
                inputs = data["inputs"]
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
                _ = model(inputs)
            else:
                raise TypeError
    return model
