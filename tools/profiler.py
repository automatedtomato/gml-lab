from __future__ import annotations

import inspect
import json
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
from torch.fx import GraphModule, Interpreter, Node

from src.gml_lab.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger("profiler")


def _serialize_value(value: Any) -> Any:  # noqa: ANN401
    """Convert values to a json format."""
    if isinstance(value, tuple):
        return tuple(_serialize_value(v) for v in value)
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return str(value)


class FxProfiler(Interpreter):
    """Custom FxProfiler class."""

    def __init__(self, gm: GraphModule) -> None:
        super().__init__(gm)
        self.results: list[dict[str, Any]] = []
        self.total_runtime_us: float = 0.0

    def run(self, *args: Any) -> Any:  # noqa: ANN401
        """Run fx interpreter on GraphModule."""
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter_ns()
            result = super().run(*args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter_ns()
        self.total_runtime_us = (end_time - start_time) / 1000.0
        return result

    def run_node(self, node: Node) -> Any:  # noqa: ANN401
        """Run fx interpreter on a node."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter_ns()
        result = super().run_node(node)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter_ns()
        exec_time_us = (end_time - start_time) / 1000.0

        op_type = "Unknown"
        params = {}

        if node.op == "call_module":
            submodule = self.module.get_submodule(str(node.target))
            op_type = type(submodule).__name__
            params = self._extract_params(submodule)

        elif node.op == "call_function":
            op_type = (
                node.target.__name__
                if hasattr(node.target, "__name__")
                else str(node.target)
            )

        elif node.op == "call_method":
            op_type = str(node.target)
        else:
            op_type = node.op

        output_shape = None
        if isinstance(result, torch.Tensor):
            output_shape = list(result.shape)
        elif (
            isinstance(result, (tuple, list))
            and len(result) > 0
            and isinstance(result[0], torch.Tensor)
        ):
            output_shape = [
                list(r.shape) for r in result if isinstance(r, torch.Tensor)
            ]

        op_info = {
            "node_name": node.name,
            "op_type": op_type,
            "op_kind": node.op,  # call_module, call_function etc.
            "exec_time_us": exec_time_us,
            "params": params,
            "output_shape": _serialize_value(output_shape),
        }
        self.results.append(op_info)
        return result

    def _extract_params(self, gm: GraphModule) -> dict[str, Any]:
        params: dict[str, Any] = {}

        if hasattr(gm, "weight") and isinstance(gm.weight, torch.Tensor):
            params["weight"] = gm.weight.shape
            try:
                sig = inspect.signature(gm.__class__.__init__)

                for param_name in sig.parameters:
                    if param_name in ["self", "args", "kwrgs"]:
                        continue
                    if hasattr(gm, param_name):
                        value = getattr(gm, param_name)
                        if not isinstance(value, torch.Tensor | torch.nn.Module):
                            params[param_name] = _serialize_value(value)
            except (ValueError, TypeError):
                pass

        return params

    def dump_to_json(self, filepath: str | Path) -> None:
        """Dump profile result to json."""
        output_data = {
            "total_runtime_us": self.total_runtime_us,
            "node_profiles": self.results,
        }
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

    def print_summary(self) -> None:
        """Print profiling summary in std."""
        ignored_ops = {"placeholder", "get_attr", "output"}
        grouped_results = defaultdict(list)
        grouped_duration = defaultdict(int)
        for res in self.results:
            if res["op_kind"] in ignored_ops:
                continue
            grouped_results[res["op_type"]].append(res)
            grouped_duration[res["op_type"]] = (
                grouped_duration.get(res["op_type"], 0) + res["exec_time_us"]
            )
        print("\nProfiling Summary")
        print("=" * 60)
        print(f"Total Graph Exec Time: {self.total_runtime_us} us\n")
        print("Op Duraions:")
        for op_type, result_list in grouped_results.items():
            print(f"  {op_type} {grouped_duration[op_type]:>7.1f} us")
            params_map = defaultdict(list)
            params_duration = defaultdict(int)
            for res in result_list:
                params_str = ", ".join(f"{k}: {v}" for k, v in res["params"].items())
                params_map[params_str].append(res["exec_time_us"])
                params_duration[params_str] = (
                    params_duration.get(params_str, 0) + res["exec_time_us"]
                )

            for params_str, times in params_map.items():
                if params_str:
                    print(f"    {params_str} {params_duration[params_str]:>7.1f} us")
                for time_us in times:
                    print(f"      {time_us:>7.1f} us\n")
        print("-" * 60)
        print(f"Graph Dur: {self.total_runtime_us:>9.1f} us\n")
