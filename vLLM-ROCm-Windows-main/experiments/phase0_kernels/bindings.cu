// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
#include <torch/extension.h>

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps);
torch::Tensor wmma_gemm(torch::Tensor a, torch::Tensor b);
torch::Tensor wmma_igemm(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "RMSNorm (fp32) on HIP");
    m.def("wmma_gemm", &wmma_gemm, "fp16->fp32 GEMM via rocWMMA (16x16x16, wave32)");
    m.def("wmma_igemm", &wmma_igemm, "int8->int32 GEMM via rocWMMA (iu8, 16x16x16, wave32)");
}
