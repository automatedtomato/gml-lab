# 002. Use Abstract Base Class and Polymorphism for Quantized Modules

Date: 2026-02-25

## Context

In our GML-Lab optimization pipeline, the graph lowering passes (e.g., `lower_add`, `lower_relu`) replace standard PyTorch nodes with our custom INT8 CUDA kernels. During the development of the ResNet18 pipeline, we encountered two major architectural bottlenecks:

1. **Qparam Extraction on Mutated Nodes**: Due to residual connections, a lowering pass frequently encounters nodes that have *already* been mutated into custom modules. The extraction utility (`extract_qparams`) crashed because it relied on hardcoded `getattr` logic specific to PyTorch's `quantize_per_tensor` function. Scaling this by adding `if isinstance(...)` for every new kernel violates the Open/Closed Principle.
2. **Dangling Graph References**: Aggressively calling `graph.erase_node()` on Dequantize (DQ) nodes caused `RuntimeError`s when those nodes were still consumed by other branches (e.g., Skip Connections).

## Options

1. **Option A (Duck Typing & Try/Catch)**: Rely on `hasattr` checks for qparams and wrap node deletions in `try/except` blocks. This is easy to write but masks potential bugs and provides zero type-safety or structural guarantees for future kernel developers.
2. **Option B (Abstract Base Class Hierarchy & Safe GC)**: Introduce a formal class hierarchy (`GMLQuantModuleBase` -> `UnaryOpsBase` / `BinaryOpsBase`) that enforces standard buffer registration (`output_scale`, `output_zp`). Additionally, implement a centralized garbage collection helper (`remove_unused_nodes`) that checks `len(node.users) == 0` before deletion.

## Decision

We decided to adopt **Option B**

By standardizing the interface through `GMLQuantModuleBase`, the lowering utility `extract_qparams` is now completely decoupled from specific kernel implementations. Future kernels (Linear, Conv2d) simply need to inherit from the appropriate base class. Furthermore, the `remove_unused_nodes` helper ensures our graph mutations remain topologically sound, preventing crashes on complex networks like ResNet.

## Consequences

* **Positive**: Adding new kernels requires zero changes to the lowering utility. The graph mutation code is significantly cleaner and less error-prone.
* **Negative**: Kernel developers must strictly adhere to the ABC constraints (e.g., calling `super().__init__` with the correct arguments) to ensure buffers are registered properly.