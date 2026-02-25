#include "kernel.hpp"

/**
 * @brief square kernel (fp*)
 *
 * @param input Input tensor (fp*)
 * @param output Output tensor (fp*)
 * @param size Input tensor size (int32_t)
 */
template <typename T>
__global__ void kernel_square(
    const T* __restrict__ input,
    T* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

/**
 * @brief square for fp* input
 *
 * @param input Input tensor (fp*)
 * @return Output tensor of squared input (fp*)
 */
torch::Tensor square(const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_cuda(), "must be a CUDA tensor.");

    auto output = torch::empty_like(input);
    int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kernel_square", ([&] {
        kernel_square<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));
    return output;
}