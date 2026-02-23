# 001. Normalize FX Graph Operations to Modules Before Quantization

Date: 2026-02-21

## Context

In our GML-Lab quantization pipeline, we rely on PyTorch's `BackendConfig` to fuse operator patterns like `Conv2d + BatchNorm2d + ReLU` into a single intrinsic module (`nni.ConvReLU2d`).

We discovered that PyTorch's default fusers silently fail when the activation function is a functional call (`F.relu`) or a method call (`x.relu()`), as they strictly expect `nn.Module` instances. While we *could* bypass this by writing custom fuser methods that inject dummy `nn.Module`s, this approach has two major flaws:

1. It causes our `BackendConfig` to explode with redundant mappings for every functional and method variant.
2. Unfused standalone operations (e.g., a residual `F.relu` or `x.add()`) remain in the graph as `call_function` or `call_method`. This forces our downstream `lower_to_gml` compiler pass to handle multiple node types for a single mathematical operation, leading to severely bloated and unmaintainable lowering logic.

## Options

1. **Option A (Custom Fusers)**: Expand `BackendConfig` with custom fusers that handle functional/method variants by injecting dummy modules, and add branching logic to our lowering passes.
2. **Option B (IR Normalization Pass)**: Introduce a pre-processing graph transformation pass (Unification Pass) before `prepare_fx`. This pass will intercept functional and method calls, instantiate equivalent `nn.Module` classes, and replace the nodes with standard `call_module` operations.

## Decision

We decided to adopt **Option B (IR Normalization Pass)**.

By unifying the Intermediate Representation (IR) to strictly use `call_module` for targeted operations, we keep our `BackendConfig` minimal. More importantly, this strictly normalized IR dramatically simplifies the pattern matching required in our downstream lowering passes to GML custom CUDA kernels. This aligns perfectly with our core objective of building a clean, robust vertical optimization pipeline.

## Consequences

(To be filled after the implementation and evaluation of the pass)