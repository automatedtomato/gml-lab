# 001. Normalize FX Graph Operations to Modules Before Quantization

Date: 2026-02-21

## Context

In our GML-Lab quantization pipeline, we rely on PyTorch's `BackendConfig` and `torch.ao.quantization.fuser_method_mappings` to fuse operator patterns like `Conv2d + BatchNorm2d + ReLU` into a single intrinsic module (`nni.ConvReLU2d`). 

We discovered that PyTorch's default fusers silently fail to fuse these patterns when the activation function is implemented as a functional call (`F.relu`, `torch.relu`) or a method call (`x.relu()`) instead of an `nn.Module` (`nn.ReLU`). Since `mmpretrain` models utilize diverse implementation styles, accommodating all variations directly in the `BackendConfig` would lead to an explosion of custom fuser mappings and significantly complicate our subsequent lowering pass to custom CUDA kernels.

## Options

1. **Option A (PyTorch Native Approach)**: Expand `BackendConfig` by registering every possible combination of modules, functions, and methods, and implement custom wrapper fuser functions for each functional/method variant.
2. **Option B (IR Normalization Approach)**: Introduce a pre-processing graph transformation pass before `prepare_fx`. This pass will intercept functional (`call_function`) and method (`call_method`) calls, instantiate their equivalent `nn.Module` classes, and replace the nodes with standard `call_module` operations.

## Decision

We decided to adopt **Option B (IR Normalization Approach)**. 

By unifying the Intermediate Representation (IR) to strictly use `call_module` for targeted operations, we keep our `BackendConfig` minimal and clean. More importantly, this strictly normalized IR dramatically simplifies the downstream compiler passes—specifically, the pattern matching required to lower operations to GML custom CUDA kernels. This aligns perfectly with our core objective of bridging the gap between high-level abstractions and low-level hardware implementations.

## Consequences

(To be filled after the implementation and evaluation of the pass)
