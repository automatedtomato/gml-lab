#include "kernel.hpp"
#include <algorithm>
#include <cmath>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/tensor.h>
#include <optional>

using ElementInput = int8_t;
using ElementOutput = int8_t;
using ElementAccum = int32_t;
using ElementCompute = float;

using LayoutInput = cutlass::layout::TensorNHWC;
using LayoutWeight = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using EpilogueBase =
    cutlass::epilogue::thread::LinearCombinationClamp<ElementAccum, 4, ElementAccum,
                                                      ElementCompute>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInput, LayoutInput, ElementInput, LayoutWeight, ElementAccum, LayoutOutput,
    ElementAccum, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>, // Threadblock tile shape
    cutlass::gemm::GemmShape<64, 64, 64>,   // Warp tile shape
    cutlass::gemm::GemmShape<16, 8, 16>,    // TensorCore instruction shape
    EpilogueBase, SwizzleThreadBlock,
    3, // Num stages
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

using ConvBase = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

// ===== Requant kernels =====
__global__ void kernel_requantize_per_tensor(const int32_t *__restrict__ input_int32,
                                             int8_t *__restrict__ output_int8,
                                             float scale, float zp, bool fuse_relu,
                                             int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = static_cast<float>(input_int32[idx]);
        float transformed = val * scale + zp;
        float rounded = floorf(transformed + 0.5f);

        if (fuse_relu) {
            rounded = fmaxf(0.0f, rounded);
        }

        output_int8[idx] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, rounded)));
    }
}

__global__ void kernel_requantize_per_channel(const int32_t *__restrict__ input_int32,
                                              int8_t *__restrict__ output_int8,
                                              const float *__restrict__ scales,
                                              float zp, bool fuse_relu, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int col = idx % n;
        float scale = scales[col];
        float val = static_cast<float>(input_int32[idx]);
        float transformed = val * scale + zp;
        float rounded = floorf(transformed + 0.5f);

        if (fuse_relu) {
            rounded = fmaxf(0.0f, rounded);
        }
        output_int8[idx] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, rounded)));
    }
}

// ===== Pytorch Wrapper =====
torch::Tensor fused_quant_conv(const torch::Tensor &input, const torch::Tensor &weight,
                               std::optional<torch::Tensor> bias,
                               torch::IntArrayRef stride, torch::IntArrayRef padding,
                               torch::IntArrayRef dilation, int64_t groups,
                               const torch::Tensor &scales, int32_t output_zp,
                               bool fuse_relu, bool is_per_channel) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_INT8(input);
    CHECK_INT8(weight);

    int64_t n_batch = input.size(0); // batch * seq_len
    int64_t c_in = input.size(3);    // in_features
    int64_t c_out = weight.size(0);  // out_features

    cutlass::conv::Conv2dProblemSize prob_size(
        cutlass::Tensor4DCoord(n_batch, input.size(1), input.size(2),
                               c_in), // N, H, W, C
        cutlass::Tensor4DCoord(
            c_out, weight.size(1), weight.size(2),
            c_in / groups), // out_feautres, kH, kW, in_geatures / groupd
        cutlass::Tensor4DCoord(padding[0], padding[0], padding[1], padding[1]),
        cutlass::MatrixCoord(stride[0], stride[1]),
        cutlass::MatrixCoord(dilation[0], dilation[1]),
        cutlass::conv::Mode::kCrossCorrelation,
        1, // split_k_slices
        groups);

    int32_t *bias_ptr = nullptr;
    if (bias.has_value() && bias->numel() > 0) {
        CHECK_CUDA(bias.value());
        CHECK_INT32(bias.value());
        bias_ptr = bias->data_ptr<int32_t>();
    }

    auto out_extent = prob_size.output_extent(); // N, H_out, W_out, K
    auto temp_output =
        torch::empty({out_extent[0], out_extent[1], out_extent[2], out_extent[3]},
                     input.options().dtype(torch::kInt32));
    auto final_output =
        torch::empty({out_extent[0], out_extent[1], out_extent[2], out_extent[3]},
                     input.options().dtype(torch::kInt8));

    typename ConvBase::Arguments args(
        prob_size,
        {input.data_ptr<int8_t>(), LayoutInput::packed(prob_size.activation_extent())},
        {weight.data_ptr<int8_t>(), LayoutWeight::packed(prob_size.filter_extent())},
        {bias_ptr, LayoutOutput::Stride(0)},
        {temp_output.data_ptr<int32_t>(), LayoutOutput::packed(out_extent)},
        {1.0f, 1.0f} // alpha=1, beta1 (D = 1*(A*B) + 1*C)
    );

    ConvBase conv_op;
    TORCH_CHECK(conv_op(args) == cutlass::Status::kSuccess, "CUTLASS Conv failed");

    int total_elements = out_extent[0] * out_extent[1] * out_extent[2] * out_extent[3];
    int blocks = get_blocks(total_elements);

    if (is_per_channel) {
        int spacial_size = out_extent[0] * out_extent[1] *
                           out_extent[2]; // Separate spacial and channe
        kernel_requantize_per_channel<<<blocks, kDefaultThreads>>>(
            temp_output.data_ptr<int32_t>(), final_output.data_ptr<int8_t>(),
            scales.data_ptr<float>(), output_zp, fuse_relu, spacial_size,
            out_extent[3]);
    } else {
        float scale_val = scales[0].item<float>();
        kernel_requantize_per_tensor<<<blocks, kDefaultThreads>>>(
            temp_output.data_ptr<int32_t>(), final_output.data_ptr<int8_t>(), scale_val,
            output_zp, fuse_relu, total_elements);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "Requantize kernel failed: ", cudaGetErrorString(err));

    return final_output;
}
