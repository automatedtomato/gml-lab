from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from examples.utils import perf_profile, prepare_dataloader, quantize, set_seed
from src.gml_lab.config_builder import build_mm_config
from src.gml_lab.evaluation import evaluate
from src.gml_lab.lowering.lower_to_gml import lower_to_gml
from src.gml_lab.modeling import load_model
from tools.layer_by_layer_analysis import generate_analysis_report
from tools.visualize_graph import dump_graph


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenMMLab examples.")
    parser.add_argument(
        "eval_options",
        nargs="*",
        type=str,
        help=("Select evaluation options from ['float', 'qdq', 'custom_ops']."),
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="resnet18_8xb32_in1k",
        help=(
            "Specify the model name compatible with `mim download`. "
            "To see available models, run `mim search mmdet`."
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
        "--graph-dump-dir",
        type=Path,
        help=(
            "Directory where visualized graph are saved in dot format. "
            "If specified, FxGraphDrawer visualize graphs, "
            "which requires graphviz and pydot as dependencies."
        ),
    )
    parser.add_argument(
        "--lbl-dump-dir",
        type=Path,
        help=(
            "Directory where layer-by-layer sensitivity analysis reports are saved. "
        ),
    )
    parser.add_argument(
        "--enable-profile",
        action="store_true",
        help=(
            "If specified, FxProfiler runs and gather profiling info. "
            "Save results to `examples/results/arch-name/ ."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run main example path."""
    seed = set_seed()
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = build_mm_config(
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
    example_inputs = tuple(i.to(device) for i in example_input)

    calib_images = args.batch_size if args.calib_images is None else args.calib_images
    total_calib_batches = math.ceil(calib_images / args.batch_size)

    prepared_model, qdq_model = quantize(
        float_model,
        example_inputs,
        calib_loader,
        total_calib_batches,
        data_preprocessor,
    )

    gml_model = lower_to_gml(qdq_model)

    if args.graph_dump_dir is not None:
        dump_graph(prepared_model, args.arch + "_prepared", args.graph_dump_dir)
        dump_graph(qdq_model, args.arch + "_qdq", args.graph_dump_dir)
        dump_graph(gml_model, args.arch + "_cuda", args.graph_dump_dir)

    if args.lbl_dump_dir is not None:
        prepared_model.eval()
        qdq_model.eval()
        gml_model.eval()
        analysis_input = example_input[:1].to(device)
        # prepared_model.eval().to("cpu")
        # qdq_model.eval().to("cpu")
        # gml_model.eval().to("cpu")
        # analysis_input = example_input[:1].to("cpu")
        generate_analysis_report(
            prepared_model, qdq_model, analysis_input, args.lbl_dump_dir
        )
        generate_analysis_report(
            qdq_model, gml_model, analysis_input, args.lbl_dump_dir
        )
        prepared_model.to(device)
        qdq_model.to(device)
        gml_model.to(device)

    if args.enable_profile:
        float_model.to(device)
        qdq_model.to(device)
        gml_model.to(device)
        save_dir = Path(f"examples/results/{args.arch}")
        save_dir.mkdir(parents=True, exist_ok=True)
        perf_profile(float_model, example_inputs, save_dir / "float_prof.json")
        perf_profile(qdq_model, example_inputs, save_dir / "qdq_prof.json")
        perf_profile(gml_model, example_inputs, save_dir / "cuda_prof.json")

    if "float" in args.eval_options:
        _ = evaluate(cfg, args.arch, float_model, "float", test_loader, seed)
    if "qdq" in args.eval_options:
        _ = evaluate(cfg, args.arch, qdq_model, "qdq", test_loader, seed)
    if "cuda" in args.eval_options:
        _ = evaluate(cfg, args.arch, qdq_model, "cuda", test_loader, seed)


if __name__ == "__main__":
    main()
