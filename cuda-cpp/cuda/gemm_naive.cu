#include "gpt2_cuda/kernels.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace gpt2 {
namespace cuda {

__global__ void gemm_naive_kernel(const float* a, const float* b, float* c, int m, int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float sum = 0.f;
  for (int t = 0; t < k; ++t) sum += a[row * k + t] * b[t * n + col];
  c[row * n + col] = sum;
}

void gemm_naive_f32(const float* a, const float* b, float* c, int m, int k, int n) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  gemm_naive_kernel<<<grid, block>>>(a, b, c, m, k, n);
}

static void add_bias(float* c, const float* bias, int m, int n) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) c[i * n + j] += bias[j];
}

void gemm_self_test(const float* x, const float* w, const float* bias, const float* expected, int m, int k,
                    int n) {
  float *d_x, *d_w, *d_c;
  cudaMalloc(&d_x, m * k * sizeof(float));
  cudaMalloc(&d_w, k * n * sizeof(float));
  cudaMalloc(&d_c, m * n * sizeof(float));
  cudaMemcpy(d_x, x, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, k * n * sizeof(float), cudaMemcpyHostToDevice);
  gemm_naive_f32(d_x, d_w, d_c, m, k, n);
  std::vector<float> host(m * n);
  cudaMemcpy(host.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  add_bias(host.data(), bias, m, n);
  float max_err = 0.f;
  for (int i = 0; i < m * n; ++i) max_err = fmaxf(max_err, fabsf(host[i] - expected[i]));
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_c);
  if (max_err > 1e-3f) {
    std::fprintf(stderr, "gemm_self_test FAIL max_err=%f\n", max_err);
  } else {
    std::printf("gemm_self_test OK (max_err=%f)\n", max_err);
  }
}

}  // namespace cuda
}  // namespace gpt2
