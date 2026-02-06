#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "must be a CUDA tensor.")

extern torch::Tensor square(const torch::Tensor& input);