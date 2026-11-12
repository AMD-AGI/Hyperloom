// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// Phase-0 Gate A: minimal HIP device kernel compiled by hipcc (clang) for gfx1100.
#include <torch/extension.h>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

torch::Tensor hip_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be on GPU");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "fp32 only");
    a = a.contiguous();
    b = b.contiguous();
    auto c = torch::empty_like(a);
    int n = static_cast<int>(a.numel());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                    c.data_ptr<float>(), n);
    return c;
}
