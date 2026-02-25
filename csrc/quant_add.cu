#include "kernel.hpp"
#include <cmath>
#include <algorithm>

/**
 * @brief Kernel function for GMLQuantAdd/GMLQuantAddReLU
 *
 * @param a input tensor (int8_t)
 * @param b input tensor (int8_t)
 * @param output (int8_t)
 * @param za input(a) zero point (int32_t)
 * @param zb input(b) zero point (int32_t)
 * @param out_zp (int32_t)
 * @param requant_scale_a input(a) scale for requantization (float)
 * @param requant_scale_a input(b) scale for requantization (float)
 * @param lower_bound lower clip bound (int8_t)
 * @param upper_bound upper clip bound (int8_t)
 * @param size (int)
 */

__global__ void kernel_quant_add(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b, int8_t* __restrict__ output,
    int32_t za, int32_t zb, int32_t out_zp, float requant_scale_a,
    float requant_scale_b, int8_t lower_bound, int8_t upper_bound, int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {

        int32_t val_a = static_cast<int32_t>(a[idx]) - za;
        int32_t val_b = static_cast<int32_t>(b[idx]) - zb;

        float sum = (val_a * requant_scale_a) + (val_b * requant_scale_b);

        int32_t out_int32 = static_cast<int32_t>((roundf(sum)) + out_zp);

        if (out_int32 < lower_bound) out_int32 = lower_bound;
        if (out_int32 > upper_bound) out_int32 = upper_bound;

        output[idx] = static_cast<int8_t>(out_int32);
    }
}

/**
 * @brief Add / AddReLU for int8 input
 */
torch::Tensor quant_add(
    const torch::Tensor& a, const torch::Tensor& b, int32_t za, int32_t zb,
    int32_t out_zp, float requant_scale_a, float requant_scale_b,
    bool has_relu
) {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
    CHECK_INT8(a);
    CHECK_INT8(b);
    CHECK_SIZE(a, b);

    auto output = torch::empty_like(a);
    int size = a.numel();
    int blocks = get_blocks(size);

    int8_t lower_bound = has_relu ? std::max(-128, out_zp) : -128;
    int8_t upper_bound = 127;

    kernel_quant_add<<<blocks, kDefaultThreads>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), output.data_ptr<int8_t>(),
        za, zb, out_zp, requant_scale_a, requant_scale_b, lower_bound, upper_bound, size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel_q_add: %s\n", cudaGetErrorString(err));
    }

    return output;
}
