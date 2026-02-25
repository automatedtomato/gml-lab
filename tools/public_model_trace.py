import argparse
import contextlib
from pathlib import Path

import mmpretrain
import torch

from src.gml_lab.modeling import FxWrapper

from .visualize_graph import dump_graph


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trace Attempt.")
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
        "--graph-dump-dir",
        type=Path,
        default=Path("results/graph"),
        help=(
            "Directory where visualized graph are saved in dot format. "
            "If specified, FxGraphDrawer visualize graphs, "
            "which requires graphviz and pydot as dependencies."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Check public model graph structure."""
    args = parse_args()
    device = "cpu"

    float_model = mmpretrain.get_model(model=args.arch, pretrained=True, device=device)
    wrapped_model = FxWrapper(float_model)
    traced_model = torch.fx.symbolic_trace(wrapped_model)

    dump_graph(traced_model, args.arch, args.graph_dump_dir)

    with open(args.graph_dump_dir / "qdq.txt", "w") as f, contextlib.redirect_stdout(f):
        traced_model.graph.print_tabular()


if __name__ == "__main__":
    main()
