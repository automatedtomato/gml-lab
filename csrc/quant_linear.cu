#include "kernel.hpp"
#include <algorithm>
#include <cmath>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm.h>
#include <optional>

using ElementInput = int8_t;  // input, self.weight
using ElementOutput = int8_t; // final output
using ElementAccum = int32_t; // compute bucket
using ElementCompute = float; // for requant calc

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

using EpilogueBase = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementAccum, // tmp output -> requant process
    4,            // int32_t = 4byte : 16 / 4 = 4 elts
    ElementAccum, // bucket
    ElementCompute>;

using GemmBase =
    cutlass::gemm::device::Gemm<ElementInput, RowMajor,    // input
                                ElementInput, ColumnMajor, // weight
                                ElementAccum, RowMajor,    // output
                                ElementAccum,              // bias
                                cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                cutlass::gemm::GemmShape<128, 128, 64>,
                                cutlass::gemm::GemmShape<64, 64, 64>,
                                cutlass::gemm::GemmShape<16, 8, 16>, EpilogueBase>;

// ===== Requant kernels =====
__global__ void kernel_requantize_per_tensor(const int32_t *__restrict__ input_int32,
                                             int8_t *__restrict__ output_int8,
                                             float scale, float zp, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = static_cast<float>(input_int32[idx]);
        float transformed = val * scale + zp;
        float rounded = floorf(transformed + 0.5f);
        output_int8[idx] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, rounded)));
    }
}

__global__ void kernel_requantize_per_channel(const int32_t *__restrict__ input_int32,
                                              int8_t *__restrict__ output_int8,
                                              const float *__restrict__ scales,
                                              float zp, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int col = idx % n;
        float scale = scales[col];
        float val = static_cast<float>(input_int32[idx]);
        float transformed = val * scale + zp;
        float rounded = floorf(transformed + 0.5f);
        output_int8[idx] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, rounded)));
    }
}

// ===== Pytorch Wrapper =====
torch::Tensor quant_linear(const torch::Tensor &input, const torch::Tensor &weight,
                           std::optional<torch::Tensor> bias,
                           const torch::Tensor &scales, int32_t output_zp,
                           bool is_per_channel) {
    CHECK_CUDA(input);
    CHECK_INT8(input);
    CHECK_CUDA(weight);
    CHECK_INT8(weight);

    int m = input.size(0);
    int k = input.size(1);
    int n = weight.size(0);

    int32_t *bias_ptr = nullptr;
    if (bias.has_value() && bias->numel() > 0) {
        CHECK_CUDA(bias.value());
        bias_ptr = bias->data_ptr<int32_t>();
    }

    // Calc matmul with CUTLASS (M x N)
    auto temp_output = torch::empty({m, n}, input.options().dtype(torch::kInt32));
    auto final_output = torch::empty({m, n}, input.options().dtype(torch::kInt8));

    typename GemmBase::Arguments args({m, n, k},
                                      {input.data_ptr<int8_t>(), k}, // stride = k
                                      {weight.data_ptr<int8_t>(), k}, {bias_ptr, 0},
                                      {temp_output.data_ptr<int32_t>(), n}, {1.0f, 1.0f}
                                      // alpha=1, beta=1 (D = 1*(A*B) + 1*C)
    );

    GemmBase gemm_op;
    TORCH_CHECK(gemm_op(args) == cutlass::Status::kSuccess, "CUTLASS GEMM failed");

    // Requantize int32 buffer with custom kernel
    int total_elements = m * n;
    int blocks = get_blocks(total_elements);

    if (is_per_channel) {
        kernel_requantize_per_channel<<<blocks, kDefaultThreads>>>(
            temp_output.data_ptr<int32_t>(), final_output.data_ptr<int8_t>(),
            scales.data_ptr<float>(), output_zp, m, n);
    } else {
        float scale_val = scales.data_ptr<float>()[0];
        kernel_requantize_per_tensor<<<blocks, kDefaultThreads>>>(
            temp_output.data_ptr<int32_t>(), final_output.data_ptr<int8_t>(), scale_val,
            output_zp, total_elements);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Requantize kernel failed: ", cudaGetErrorString(err));

    return final_output;
}
