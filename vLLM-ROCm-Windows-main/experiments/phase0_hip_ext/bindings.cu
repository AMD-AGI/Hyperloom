// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// Phase-0 Gate A: pybind binding in a separate translation unit (proves multi-file build).
// Kept as .cu so torch's hipify path assigns it a non-None hipified_path (see README).
#include <torch/extension.h>

torch::Tensor hip_add(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hip_add", &hip_add, "elementwise a+b on HIP (multi-file extension)");
}
