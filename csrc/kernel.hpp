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
#define CHECK_TYPE(a, b) TORCH_CHECK(a.scalar_type() == b.scalar_type(), "must be the same type")
#define CHECK_SIZE(a, b) TORCH_CHECK(a.sizes() == b.sizes(), "must be the same size")

extern torch::Tensor square(const torch::Tensor& input);
extern torch::Tensor relu(const torch::Tensor& input);
extern torch::Tensor quant_relu(const torch::Tensor& input, int32_t zero_point);
extern torch::Tensor quant_add(
    const torch::Tensor& a, const torch::Tensor& b, int32_t za, int32_t zb, int32_t out_zp,
    float requant_scale_a, float requant_scale_b, bool has_relu
);