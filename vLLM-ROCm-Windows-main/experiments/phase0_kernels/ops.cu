// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// Phase-0+ kernel hardening: two REAL kernels on gfx1100.
//  (1) rmsnorm  — a genuine LLM op (block reduction + normalize + weight), fp32.
//  (2) wmma_gemm — fp16 -> fp32 GEMM using rocWMMA 16x16x16 wave32 (the RDNA3 matrix path).
#include <torch/extension.h>
#include <rocwmma/rocwmma.hpp>

// ---------------- RMSNorm ----------------
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ out,
                               int hidden, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * hidden;
    float* outr = out + (size_t)row * hidden;
    float local = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = xr[i];
        local += v * v;
    }
    __shared__ float sdata[256];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] / hidden + eps);
    for (int i = tid; i < hidden; i += blockDim.x) outr[i] = xr[i] * inv * w[i];
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "cuda only");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "fp32 only");
    x = x.contiguous();
    w = w.contiguous();
    int rows = static_cast<int>(x.size(0));
    int hidden = static_cast<int>(x.size(1));
    auto out = torch::empty_like(x);
    rmsnorm_kernel<<<rows, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(),
                                  out.data_ptr<float>(), hidden, (float)eps);
    return out;
}

// ---------------- WMMA GEMM (rocWMMA) ----------------
using namespace rocwmma;

__global__ void wmma_gemm_kernel(const float16_t* a, const float16_t* b, float32_t* c,
                                 int M, int N, int K) {
    int colStart = blockIdx.x * 16;  // N tile
    int rowStart = blockIdx.y * 16;  // M tile
    fragment<accumulator, 16, 16, 16, float32_t> acc;
    fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        fragment<matrix_a, 16, 16, 16, float16_t, row_major> aFrag;
        fragment<matrix_b, 16, 16, 16, float16_t, row_major> bFrag;
        load_matrix_sync(aFrag, a + (size_t)rowStart * K + k, K);  // A: MxK row-major
        load_matrix_sync(bFrag, b + (size_t)k * N + colStart, N);  // B: KxN row-major
        mma_sync(acc, aFrag, bFrag, acc);
    }
    store_matrix_sync(c + (size_t)rowStart * N + colStart, acc, N, mem_row_major);
}

torch::Tensor wmma_gemm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cuda only");
    TORCH_CHECK(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf, "fp16 inputs");
    a = a.contiguous();
    b = b.contiguous();
    int M = static_cast<int>(a.size(0));
    int K = static_cast<int>(a.size(1));
    int N = static_cast<int>(b.size(1));
    TORCH_CHECK(K == b.size(0), "inner dims must match");
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "M,N,K must be multiples of 16");
    auto c = torch::zeros({M, N}, a.options().dtype(torch::kFloat32));
    dim3 grid(N / 16, M / 16);
    dim3 block(32);  // one wave32 per output tile on gfx1100
    wmma_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const float16_t*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const float16_t*>(b.data_ptr<at::Half>()),
        c.data_ptr<float>(), M, N, K);
    return c;
}

// ---------------- INT8 WMMA GEMM (iu8 -> i32): building block of W4A16/W8A8 ----------------
__global__ void wmma_igemm_kernel(const int8_t* a, const int8_t* b, int32_t* c,
                                  int M, int N, int K) {
    int colStart = blockIdx.x * 16;
    int rowStart = blockIdx.y * 16;
    fragment<accumulator, 16, 16, 16, int32_t> acc;
    fill_fragment(acc, 0);
    for (int k = 0; k < K; k += 16) {
        fragment<matrix_a, 16, 16, 16, int8_t, row_major> aFrag;
        fragment<matrix_b, 16, 16, 16, int8_t, row_major> bFrag;
        load_matrix_sync(aFrag, a + (size_t)rowStart * K + k, K);
        load_matrix_sync(bFrag, b + (size_t)k * N + colStart, N);
        mma_sync(acc, aFrag, bFrag, acc);
    }
    store_matrix_sync(c + (size_t)rowStart * N + colStart, acc, N, mem_row_major);
}

torch::Tensor wmma_igemm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cuda only");
    TORCH_CHECK(a.scalar_type() == torch::kChar && b.scalar_type() == torch::kChar, "int8 inputs");
    a = a.contiguous();
    b = b.contiguous();
    int M = static_cast<int>(a.size(0));
    int K = static_cast<int>(a.size(1));
    int N = static_cast<int>(b.size(1));
    TORCH_CHECK(K == b.size(0), "inner dims must match");
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "M,N,K must be multiples of 16");
    auto c = torch::zeros({M, N}, a.options().dtype(torch::kInt32));
    dim3 grid(N / 16, M / 16);
    dim3 block(32);
    wmma_igemm_kernel<<<grid, block>>>(
        reinterpret_cast<const int8_t*>(a.data_ptr<int8_t>()),
        reinterpret_cast<const int8_t*>(b.data_ptr<int8_t>()),
        c.data_ptr<int32_t>(), M, N, K);
    return c;
}
