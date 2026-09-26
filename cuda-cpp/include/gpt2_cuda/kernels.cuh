#pragma once

#include <cstddef>

namespace gpt2 {
namespace cuda {

// y[i] = a[i] + b[i]
void vecadd_f32(const float* a, const float* b, float* y, int n);

// C = A @ B, A [m,k], B [k,n] row-major, C [m,n] — Conv1D layout [in,out]
void gemm_naive_f32(const float* a, const float* b, float* c, int m, int k, int n);

void vecadd_self_test();
void gemm_self_test(const float* x, const float* w, const float* bias, const float* expected, int m, int k, int n);

}  // namespace cuda
}  // namespace gpt2
