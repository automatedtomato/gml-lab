#include "kernel.hpp"

/**
 * @brief Kernel function for GMLQuantLUT (uses shared memory)
 *
 * @param input Input tensor (int8_t)
 * @param lut Look-up table tensor (int8_t, strictly 256 elems)
 * @param output Output tensor
 */
__global__ void kernel_quant_lut(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ lut,
    int8_t* __restrict__ output,
    int size 
) {
  // Hold shared memory per block
  __shared__ int8_t shared_lut[256];

  for (int i = threadIdx.x; i < 256; i+= blockDim.x) {
    shared_lut[i] = lut[i];
  }

  // Wait for load completion of all threads
  __syncthreads();
  
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    int8_t val = input[idx];
    output[idx] = shared_lut[val + 128];
  }
}

/*
 * @brief Lut activation for int8 input
 */
torch::Tensor quant_lut(const torch::Tensor& input, const torch::Tensor& lut) {
  CHECK_CUDA(input);
  CHECK_INT8(input);
  CHECK_CUDA(lut);
  CHECK_INT8(lut);

  TORCH_CHECK(lut.numel() == 256, "LUT tensor must have exactly 256 elems");
  
  auto input_c = input.contiguous();
  auto output = torch::empty_like(input_c);
  auto size = input_c.numel();

  int blocks = get_blocks(size);

  kernel_quant_lut<<<blocks, kDefaultThreads>>>(
      input_c.data_ptr<int8_t>(),
      lut.data_ptr<int8_t>(),
      output.data_ptr<int8_t>(),
      size 
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in kernel_quant_lut: %s\n", cudaGetErrorString(err));
  }
  return output;
}
