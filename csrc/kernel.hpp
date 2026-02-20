#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int kDefaultThreads = 256;
inline int get_blocks(int size, int threads = kDefaultThreads) {
    return (size + threads - 1) / threads;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "must be a CUDA tensor")
#define CHECK_INT8(x) TORCH_CHECK(x.dtype() == torch::kInt8, "must be int8 tensor")

extern torch::Tensor square(const torch::Tensor& input);
extern torch::Tensor relu(const torch::Tensor& input);
extern torch::Tensor quant_relu(const torch::Tensor& input, int32_t zero_point);