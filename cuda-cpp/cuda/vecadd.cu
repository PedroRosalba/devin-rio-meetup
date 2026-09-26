#include "gpt2_cuda/kernels.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace gpt2 {
namespace cuda {

__global__ void vecadd_kernel(const float* a, const float* b, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

void vecadd_f32(const float* a, const float* b, float* y, int n) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  vecadd_kernel<<<grid, block>>>(a, b, y, n);
}

void vecadd_self_test() {
  const int n = 1024;
  float *d_a, *d_b, *d_y;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));
  float ha[1024], hb[1024], hy[1024];
  for (int i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i);
  }
  cudaMemcpy(d_a, ha, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, hb, n * sizeof(float), cudaMemcpyHostToDevice);
  vecadd_f32(d_a, d_b, d_y, n);
  cudaMemcpy(hy, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_y);
  for (int i = 0; i < n; ++i) {
    if (std::fabs(hy[i] - (ha[i] + hb[i])) > 1e-5f) {
      std::fprintf(stderr, "vecadd_self_test failed at %d\n", i);
      return;
    }
  }
  std::printf("vecadd_self_test OK\n");
}

}  // namespace cuda
}  // namespace gpt2
