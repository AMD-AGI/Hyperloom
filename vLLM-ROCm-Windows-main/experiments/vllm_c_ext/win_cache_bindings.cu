// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// torch.ops._C_cache_ops.reshape_and_cache_flash registration, built 1:1 from csrc/cache_kernels.cu
// on native Windows ROCm (gfx1100). vLLM's v1 attention calls torch.ops._C_cache_ops.reshape_and_cache_flash
// for the per-token KV write; on ROCm it currently falls back to a Triton kernel. Building this lets the
// sliding TRITON_ATTN layers write KV via the native HIP kernel (one launch instead of a Triton launch),
// and it unblocks the native fp8-KV store (scaled_convert is already in the kernel, USE_ROCM-guarded).
// Schema copied verbatim from csrc/torch_bindings.cpp. Register under torch::kCUDA (HIP presents as
// torch.cuda on this Windows torch-rocm build -- kHIP would silently no-op).
#include <torch/extension.h>
#include <torch/library.h>

#include "cache.h"

TORCH_LIBRARY(_C_cache_ops, m) {
  // v0 PAGED layout: key_cache [num_blocks, num_kv_heads, head_size/x, block_size, x],
  // value_cache [num_blocks, num_kv_heads, head_size, block_size]. This is what the native
  // paged_attention_v1 reads (S5); reshape_and_cache_flash writes the FLASH layout instead.
  m.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  m.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  m.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  m.impl("reshape_and_cache_flash", torch::kCUDA, &reshape_and_cache_flash);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
