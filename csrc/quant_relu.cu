#include "kernel.hpp"

/**
 * @brief Kernel function for GMLQuantReLU
 *
 * @param input Input tensor (int8_t)
 * @param output (int8_t)
 * @param zero_point (int32_t)
 * @param size (int)
 */
template <typename T>
__global__ void kernel_quant_relu(
    const T* __restrict__ input,
    T* __restrict__ output,
    int32_t zero_point,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        T val = input[idx];
        T zp = static_cast<T>(zero_point);
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

    if (input.scalar_type() == torch::kUInt8) {
        kernel_quant_relu<uint8_t><<<blocks, kDefaultThreads>>>(
            input.data_ptr<uint8_t>(),
            output.data_ptr<uint8_t>(),
            zero_point,
            size
        );
    } else if (input.scalar_type() == torch::kInt8) {
        kernel_quant_relu<int8_t><<<blocks, kDefaultThreads>>>(
            input.data_ptr<int8_t>(),
            output.data_ptr<int8_t>(),
            zero_point,
            size
        );
    } else {
        TORCH_CHECK(false, "q_relu only supports uint8 or int8 tensors.");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel_q_relu: %s\n", cudaGetErrorString(err));
    }

    return output;
}

