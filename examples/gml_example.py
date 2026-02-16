from __future__ import annotations

import argparse
import math

import torch

from examples.utils import prepare_dataloader, quantize, set_seed
from src.gml_lab.config_builder import build_config
from src.gml_lab.evaluation import evaluate
from src.gml_lab.modeling import load_model
from src.gml_lab.quantizer import build_qconfig_mapping


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
    parser.add_argument(
        "--calib-images",
        type=int,
        help=(
            "Sepcify the number of images to be used in calibaretion processs. "
            "If not specifyied, `calib_images` = `batch_size`."
        ),
    )
    parser.add_argument(
        "--float-eval", action="store_true", help="If specified, evaluate float model."
    )
    parser.add_argument(
        "--qdq-eval", action="store_true", help="If specified, evaluate QDQ model."
    )
    return parser.parse_args()


def main() -> None:
    """Run main example path."""
    seed = set_seed()
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = build_config(
        model_arch=args.arch,
        data_setting=args.data,
        batch_size=args.batch_size,
    )

    float_model = load_model(args.arch, pretrained=True)
    data_preprocessor = float_model.data_preprocessor

    test_loader, calib_loader = prepare_dataloader(cfg)

    print(f"[DATASET] data={args.data}, size={len(test_loader.dataset)}")

    org_input = next(iter(test_loader))
    example_input = float_model.data_preprocessor(org_input, training=False)["inputs"]
    example_inputs = (example_input.to(device),)

    qconfig_mapping = build_qconfig_mapping()
    calib_images = args.batch_size if args.calib_images is None else args.calib_images
    total_calib_batches = math.ceil(calib_images / args.batch_size)

    _, qdq_model = quantize(
        float_model,
        example_inputs,
        qconfig_mapping,
        calib_loader,
        total_calib_batches,
        data_preprocessor,
    )

    if args.float_eval:
        _ = evaluate(cfg, args.arch, float_model, "float", test_loader, seed)

    if args.qdq_eval or args.quant_eval:
        _ = evaluate(cfg, args.arch, qdq_model, "qdq", test_loader, seed)


if __name__ == "__main__":
    main()
