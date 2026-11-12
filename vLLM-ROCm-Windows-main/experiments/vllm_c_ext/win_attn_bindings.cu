// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// torch.ops._C.paged_attention_v1/v2 registration, built 1:1 from csrc/attention/ (the GENERIC
// wave32-clean paged attention, NOT csrc/rocm/attention.cu which is wave64+MFMA gfx9-only) on native
// Windows ROCm (gfx1100). Schemas copied verbatim from csrc/torch_bindings.cpp. torch::kCUDA (HIP
// presents as torch.cuda on this Windows torch-rocm build).
#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

// FRAGMENT (not TORCH_LIBRARY): coexists with vllm_win_C's TORCH_LIBRARY(_C) in the real model
// process (loading both under a plain TORCH_LIBRARY(_C) would be a "single library per namespace"
// conflict). Standalone (microbench, no vllm_win_C) the fragment still creates/registers into _C.
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step, int sliding_window) -> ()");
  m.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  m.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  m.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
