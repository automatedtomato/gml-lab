from __future__ import annotations

from typing import Any

import mmengine
import torch
from mmpretrain.structures import DataSample


class FxWrapper(torch.nn.Module):
    """Wrapper class for calling torch.fx/torch.ao.

    Purposes:
        - Removes non-essential logic such as loss calculation, training-specific
            branching, and data pre-processing steps that are not part of
            the inference graph.
        - Ensures the model can be successfully traced by `torch.fx.symbolic_trace`,
            which often fails on dynamic control flow present in full framework models.
        - Provides a consistent `forward` signature (input -> logits)
            expected by quantization backends.

    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Create wrapper module to remove pre-process and post-process.

        Args:
            model (torch.nn.Module): Target model to be wrapped.
                The current implementation wraps networks only with elements such as
                `backbone`, `neck`, and `head`. For further configuration, necessary
                to define them separetely.

        """
        super().__init__()
        self.backbone = model.backbone
        self.neck = getattr(model, "neck", None)
        self.head = getattr(model, "head", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass without unnecessary processing."""
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        if self.head is not None:
            x = self.head(x)
        return x


class MMLabWrapper(mmengine.model.BaseModel):
    """A wrapper class to wrap GraphModule suitable for OpenMMLab runner.

    This class bridges the gap between raw computational graphs (e.g., traced,
    quantized, or optimized models via torch.fx) and the high-level `mmengine.Runner`.
    It handles data preprocessing and formats the raw tensor outputs into `DataSample`
    objects, which are required by OpenMMLab evaluators.

    This implementation adapts the logic from
    `mmpretrain.models.classifiers.ImageClassifier`.

    Reference:
        - Base Model Interface: https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model.py

    Args:
        graph_module (torch.fx.GraphModule): The traced computational graph module.
        data_preprocessor (dict[str, Any] | None): The configuration for the data
            preprocessor (e.g., mean/std normalization). This is essential for
            ensuring the input tensors are normalized correctly before passing
            them to the graph.

    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule | FxWrapper,
        data_preprocessor: dict[str, Any] | None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor)
        self.graph_module = graph_module

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: list[DataSample] | None = None,
        mode: str = "tensor",
    ) -> torch.Tensor | None | list[DataSample]:
        """Run forward path.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...).
            data_samples (list[DataSample] | None): A list of data samples containing
                meta information (e.g., ground truth labels). Defaults to None.
            mode (str): The execution mode.
                - "tensor": Returns the raw logits as a tensor.
                - "predict": Returns a list of `DataSample` objects with
                  the `pred_score` field populated.
                Defaults to "tensor".

        Returns:
            torch.Tensor | list[DataSample] | None:
                - If mode is "tensor": Returns the output logits (torch.Tensor).
                - If mode is "predict": Returns a list of `DataSample` objects.
                - Otherwise: Returns None.

        """
        logits = self.graph_module(inputs)
        if mode == "predict":
            if data_samples is None:
                data_samples = [DataSample() for _ in range(len(logits))]
            for sample, score in zip(data_samples, logits, strict=False):
                sample.set_pred_score(score)
            return data_samples
        if mode == "tensor":
            return logits
        return None


def build_mm_model(
    gm: torch.fx.GraphModule | FxWrapper, cfg: mmengine.config.Config
) -> MMLabWrapper:
    """Build MMLabWrapper with clean config.

    This function handles cleaning up the data_preprocessor
    configuration (e.g., removing 'num_classes', fixing 'to_rgb').
    """
    data_preprocessor_cfg = cfg.get("data_preprocessor", {}).copy()

    if "type" not in data_preprocessor_cfg:
        data_preprocessor_cfg["type"] = "ImgDataPreprocessor"

    data_preprocessor_cfg.pop("num_classes", None)

    if "to_rgb" in data_preprocessor_cfg:
        data_preprocessor_cfg["bgr_to_rgb"] = data_preprocessor_cfg.pop("to_rgb")

    return MMLabWrapper(gm, data_preprocessor=data_preprocessor_cfg)
