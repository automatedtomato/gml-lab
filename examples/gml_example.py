from __future__ import annotations

import argparse

import torch

from examples.utils import set_seed
from src.gml_lab.model_loader import load_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenMMLab examples.")
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        help=(
            "Specify the model name compatible with `mim download`. "
            "To see available models, run `mim searh mmdet`."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run main example path."""
    set_seed()
    args = parse_args()

    model = load_model(args.arch)
    print(model)

    test_input = torch.randn((1, 3, 224, 224))
    out = model(test_input)
    print(out.shape)


if __name__ == "__main__":
    main()
