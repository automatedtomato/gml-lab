from __future__ import annotations

import argparse

from examples.utils import set_seed
from src.gml_lab.config_builder import build_config
from src.gml_lab.evaluation import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenMMLab examples.")
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="resnet18_8xb32_in1k",
        help=(
            "Specify the model name compatible with `mim download`. "
            "To see available models, run `mim searh mmdet`."
        ),
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="imagenet_lmdb",
        help="Specify the dataset which is used in evalutaion.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Specify batch size for evaluation. Default to 64.",
    )
    return parser.parse_args()


def main() -> None:
    """Run main example path."""
    seed = set_seed()
    args = parse_args()

    cfg = build_config(
        model_arch=args.arch,
        data_setting=args.data,
        batch_size=args.batch_size,
    )

    _ = evaluate(cfg, model_arch=args.arch, target_type="float", seed=seed)


if __name__ == "__main__":
    main()
