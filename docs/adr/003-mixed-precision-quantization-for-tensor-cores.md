# 003. Mixed Precision Quantization for Tensor Cores

Date: 2026-03-02

## Context

During the INT8 quantization of the ResNet model using CUTLASS `sm_80` `OpClassTensorOp`, a `misaligned address` memory access violation occurred at the first convolutional layer (`backbone.conv1`). Investigation revealed two primary issues:

1. **Hardware Constraints**: NVIDIA Ampere INT8 Tensor Cores require 128-bit (16-byte) memory alignment. Consequently, the number of input channels (`in_channels`) must be a multiple of 16. The initial RGB image input (`c_in=3`) inherently violates this requirement.
2. **Accuracy Degradation**: Academic literature and industry best practices indicate that quantizing the first feature extraction layer to INT8 may significantly degrade the overall Signal-to-Noise Ratio (SNR) and model accuracy.

## Options

1. **Option A (Channel Padding)**: Pad the 3-channel input with 13 zero-channels to force a 16-channel alignment, allowing execution via the INT8 Tensor Core kernel.
   * *Pros*: Enables uniform INT8 kernel execution across all layers.
   * *Cons*: Increases computational and memory overhead by ~5.3x for the padded layer, and fails to mitigate the inherent quantization noise/accuracy drop.
2. **Option B (Mixed Precision / Selective Quantization)**: Skip INT8 quantization for the problematic layers (those failing the 16-channel alignment), falling back to PyTorch's native FP32/FP16 execution.

## Decision

We decided to adopt **Option B (Mixed Precision)**.

We introduced a dynamic QConfig application utility (`skip_quant_non_aligned_modules`) that iterates through the model's `named_modules()`. It automatically detects `Conv2d` layers where `in_channels % 16 != 0`, dynamically excluding them from quantization by setting their QConfig to `None`.

## Consequences

* **Positive**: Completely resolves the hardware alignment crashes while maintaining exceptionally high inference accuracy (only a 0.12pt drop compared to the FP32 baseline). The logic is scalable and model-agnostic.
* **Negative**: The skipped layers are executed in standard FP32, incurring a minor overhead in memory bandwidth and computation time compared to a hypothetical full INT8 model, though this is negligible in the context of the entire network's profile.
