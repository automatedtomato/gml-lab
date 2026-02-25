#include "kernel.hpp"

/**
 * @brief Kernel function for GMLQuantReLU
 *
 * @param input Input tensor (int8_t)
 * @param output (int8_t)
 * @param zero_point (int32_t)
 * @param size (int)
 */
__global__ void kernel_quant_relu(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int32_t zero_point,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int8_t val = input[idx];
        int32_t zp = static_cast<int32_t>(zero_point);
        output[idx] = val > zp ? val : zp;
    }
}

/**
 * @brief ReLU for int8 input
 */
torch::Tensor quant_relu(const torch::Tensor& input, int32_t zero_point) {
    CHECK_CUDA(input);
    CHECK_INT8(input);

    auto output = torch::empty_like(input);
    int size = input.numel();

    int blocks = get_blocks(size);


    kernel_quant_relu<<<blocks, kDefaultThreads>>>(
        input.data_ptr<int8_t>(),
        output.data_ptr<int8_t>(),
        zero_point,
        size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel_q_relu: %s\n", cudaGetErrorString(err));
    }

    return output;
}

