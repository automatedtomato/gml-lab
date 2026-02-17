from pathlib import Path

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer


def dump_graph(
    model: torch.fx.GraphModule, model_name: str, save_dir: Path, mode: str = "dot"
) -> None:
    """Visualize given model."""
    graph_drawer = FxGraphDrawer(
        graph_module=model,
        name=model_name,
        ignore_parameters_and_buffers=True,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / Path(model_name + "." + mode)
    with save_path.open("wb") as f:
        f.write(graph_drawer.get_dot_graph().create(format="dot"))
