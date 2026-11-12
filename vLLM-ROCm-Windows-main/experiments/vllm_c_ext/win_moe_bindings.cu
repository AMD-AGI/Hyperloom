// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// Native _moe_C TORCH_LIBRARY for Windows ROCm: registers only the ROCm-available MoE helper
// ops (moe_align_block_size, batched_moe_align_block_size, moe_sum, topk_softmax) built from
// vLLM's csrc/moe/*.cu. The CUDA-only ops (moe_wna16_gemm, marlin_moe, cutlass, SM90 router
// GEMMs) are #ifndef USE_ROCM in the upstream sources, so they are intentionally excluded.
// Compiled as .cu (clang) like win_c_bindings.cu to dead-strip c10 helper symbols.
#include <torch/extension.h>
#include <torch/library.h>
#include <optional>
#include "moe/moe_ops.h"

TORCH_LIBRARY(_moe_C, m) {
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? bias) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, "
      "Tensor! sorted_token_ids, Tensor! experts_ids, Tensor! num_tokens_post_pad, "
      "Tensor? maybe_expert_map) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch, int block_size, "
      "Tensor expert_num_tokens, Tensor! sorted_token_ids, Tensor! experts_ids, "
      "Tensor! num_tokens_post_pad) -> ()");
  m.impl("batched_moe_align_block_size", torch::kCUDA, &batched_moe_align_block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}  // import triggers the TORCH_LIBRARY static init
