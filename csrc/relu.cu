#include "kernel.hpp"

/**
 * @brief ReLU kernl (eltwise)
 * Logic: y = x > 0 ? x : 0
 * @param input Input tensor (fp*)
 * @param output Output tensor (fp*)
 * @param size Input tensor size (int)
 */
template <typename T>
__global__ void kernel_relu(
    const T* __restrict__ input,
    T* __restrict__ output,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        T val = input[idx];
        output[idx] = val > static_cast<T>(0) ? val : static_cast<T>(0);
    }
}

/**
 * @brief relu for input
 *
 * @param input Input tensor (fp*)
 * @return Output tensor of relu input (fp*)
 */
torch::Tensor relu(const torch::Tensor& input) {
    CHECK_CUDA(input);

    auto output = torch::empty_like(input);
    int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kernel_relu", ([&] {
        kernel_relu<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel_relu: %s\n", cudaGetErrorString(err));
    }

    return output;
}