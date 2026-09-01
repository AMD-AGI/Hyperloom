---
title: aiter operator catalog — which aiter.ops.* API to call
kind: language
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1, mxfp4]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - https://github.com/ROCm/aiter
---

# AITER Operator Catalog

## TL;DR
The API surface of the aiter library: the exact `aiter.ops.*` entry point (and its signature) to call
per operator — attention (`mha_fwd`/`pa_fwd_asm`/`mla_*`), GEMM (`gemm_a8w8_*`/`gemm_a4w4`/`gemm_bf16`),
MoE (`fmoe_*`/`ck_moe_*`/`topk_*`), norm, activation, RoPE, quant, KV-cache, sampling, weight-shuffle.
Use this to pick the right aiter call for a task before optimizing; each op often has CK / ASM / HIP /
Triton variants with different shape constraints (see pitfalls). This is a REFERENCE catalog, not an
authoring guide — to change a kernel's internals see authoring_delegation.md.


## Attention

### Flash Attention (CK-based)
```python
from aiter.ops.mha import mha_fwd, fmha_v3_fwd, flash_attn_func, mha_batch_prefill

mha_fwd(q, k, v, dropout_p, softmax_scale, is_causal,
        window_size_left, window_size_right, sink_size,
        return_softmax_lse, return_dropout_randval,
        cu_seqlens_q=None, cu_seqlens_kv=None, out=None,
        bias=None, alibi_slopes=None,
        q_descale=None, k_descale=None, v_descale=None,
        sink_ptr=None, gen=None)
# Returns: (out, softmax_lse, dropout_mask, rng_state)

fmha_v3_fwd(q, k, v, dropout_p, softmax_scale, is_causal,
            window_size_left, window_size_right,
            return_softmax_lse, return_dropout_randval,
            how_v3_bf16_cvt,  # BF16 conversion mode: 0=rtne, 1=rtna, 2=rtz
            out=None, bias=None, alibi_slopes=None,
            q_descale=None, k_descale=None, v_descale=None, gen=None)
```

### Paged Attention (ASM/HIP)
```python
from aiter.ops.attention import pa_fwd_asm, paged_attention_v1, paged_attention_ragged

pa_fwd_asm(Q, K, V, block_tables, context_lens,
           block_tables_stride0, max_qlen=1,
           K_QScale=None, V_QScale=None, out_=None,
           qo_indptr=None, high_precision=1, kernelName=None)
# high_precision: 0=fast, 1=standard, 2=highest (fp8)

paged_attention_v1(out, workspace_buffer, query, key_cache, value_cache,
                   scale, block_tables, cu_query_lens, context_lens,
                   max_context_len, alibi_slopes, kv_cache_dtype, kv_cache_layout,
                   logits_soft_cap, k_scale, v_scale,
                   fp8_out_scale=None, partition_size=256, mtp=1, sliding_window=0)

paged_attention_ragged(out, workspace_buffer, query, key_cache, value_cache,
                       scale, kv_indptr, kv_page_indices, kv_last_page_lens,
                       block_size, max_num_partitions, alibi_slopes,
                       kv_cache_dtype, kv_cache_layout, logits_soft_cap,
                       k_scale, v_scale, fp8_out_scale=None, partition_size=256, mtp=1)
```

### Multi-Latent Attention (MLA)
```python
# High-level API (auto split-KV via get_meta_param, backend selection)
from aiter.mla import mla_decode_fwd, mla_prefill_fwd, mla_prefill_ps_fwd, mla_decode_fwd_v4_nm

mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
               max_seqlen_q, softmax_scale, logit_cap=0.0, ...)   # DeepSeek MLA decode
mla_decode_fwd_v4_nm(...)   # DeepSeek-V4 sparse MLA decode (FP8 Q, requires sink; gfx950/gfx1250)

# Low-level ASM stage wrappers
from aiter.ops.attention import mla_decode_stage1_asm_fwd, mla_prefill_asm_fwd

mla_decode_stage1_asm_fwd(Q, KV, qo_indptr, kv_indptr, kv_page_indices,
                          kv_last_page_lens, num_kv_splits_indptr,
                          work_meta_data, work_indptr, work_info_set,
                          max_seqlen_q, page_size, nhead_kv, softmax_scale,
                          splitData, splitLse, output, lse=None,
                          q_scale=None, kv_scale=None)

mla_prefill_ps_asm_fwd(Q, K, V, qo_indptr, kv_indptr, kv_page_indices,
                       work_indptr, work_info_set, max_seqlen_q,
                       softmax_scale, is_causal, splitData, splitLse, output,
                       q_scale=None, k_scale=None, v_scale=None)
```

## GEMM / Linear

### A8W8 (INT8 × INT8)
```python
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_ck, gemm_a8w8_asm

gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Out, bias=None, splitK=0)
# XQ: [M,K] int8, WQ: [N,K] int8, scales: [M,1] and [1,N] fp32

gemm_a8w8_asm(XQ, WQ, x_scale, w_scale, Out,
              kernelName="", bias=None, bpreshuffle=True, splitK=None)

# Block-scaled variants
gemm_a8w8_blockscale_ck(XQ, WQ, x_scale, w_scale, Out)
gemm_a8w8_blockscale_cktile(Out, XQ, WQ, x_scale, w_scale, isBpreshuffled=False)
flatmm_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, out)
```

### A4W4 (FP4 × FP4, MXFP4)
```python
from aiter.ops.gemm_op_a4w4 import gemm_a4w4, gemm_a4w4_asm

gemm_a4w4(A, B, A_scale, B_scale, bias=None, dtype=bf16, alpha=1.0, beta=0.0, bpreshuffle=True)
# A: [M, K//2] fp4x2, A_scale: [M, K//32] e8m0 (per-1x32 scaling)

gemm_a4w4_asm(A, B, A_scale, B_scale, out, ...)
```

### A16W16 / BF16
```python
from aiter.ops.gemm_op_a16w16 import gemm_bf16

gemm_bf16(A, B, Out, splitK=0, bias=None)
# A: [M,K] bf16, B: [N,K] bf16, Out: [M,N] bf16
```

### Batched GEMM
```python
batched_gemm_a8w8(XQ, WQ, x_scale, w_scale, Out, ...)  # [batch, M, K]
batched_gemm_bf16(A, B, Out, ...)
```

## Mixture of Experts (MoE)

### Gate Operations
```python
from aiter.ops.moe_op import topk_softmax, topk_sigmoid, moe_fused_gate

topk_softmax(topk_weights, topk_indices, token_expert_indices,
             gating_output, need_renorm, num_shared_experts=0,
             shared_expert_scoring_func="")

topk_sigmoid(topk_weights, topk_indices, gating_output)

moe_fused_gate(input, bias, topk_weights, topk_ids,
               num_expert_group, topk_group, topk,
               n_share_experts_fusion, routed_scaling_factor=1.0)
```

### MOE GEMM
```python
from aiter.ops.moe_op import fmoe, fmoe_int8_g1u0, fmoe_g1u1, fmoe_fp8_blockscale_g1u1

fmoe(out, input, gate, down, sorted_token_ids, sorted_weights,
     sorted_expert_ids, num_valid_ids, topk)

fmoe_g1u1(out, input, gate, down, sorted_token_ids, sorted_weights,
           sorted_expert_ids, num_valid_ids, topk,
           input_scale, fc1_scale, fc2_scale,
           kernelName="", fc2_smooth_scale=None,
           activation=ActivationType.Silu.value)

fmoe_fp8_blockscale_g1u1(out, input, gate, down, sorted_token_ids,
                          sorted_weights, sorted_expert_ids, num_valid_ids, topk,
                          input_scale, fc1_scale, fc2_scale,
                          kernelName="", fc_scale_blkn=128, fc_scale_blkk=128,
                          fc2_smooth_scale=None, activation=ActivationType.Silu.value,
                          block_size_M=32)

# CK 2-stage variant
ck_moe_stage1(hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids,
              num_valid_ids, out, topk, kernelName=None,
              w1_scale=None, a1_scale=None, block_m=32, ksplit=0,
              activation=ActivationType.Silu.value, quant_type=QuantType.No.value,
              sorted_weights=None)
```

### MOE Utilities
```python
moe_align_block_size(topk_ids, num_experts, block_size,
                     sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad)
moe_sum(input, output)
```

### Expert-Parallel Dispatch/Combine (cross-GPU seam — see `framework/mori/`)
Not an `aiter.ops.*` compute kernel — this is the **communicator seam** to MoRI-EP (or, intranode-only,
FlyDSL) for the cross-GPU all-to-all. Distinct module from the single-GPU ops above.
```python
from aiter.dist.device_communicators.all2all import MoriAll2AllManager, FlyDSLAll2AllManager

mgr = MoriAll2AllManager(cpu_group)          # wraps mori.ops.EpDispatchCombineOp
handle = mgr.get_handle(kwargs)              # kwargs: rank, num_ep_ranks, hidden dims, quant dtype, ...
# consumed via AiterCommunicator.dispatch()/.combine() in communicator_cuda.py
```

## Normalization

### RMSNorm
```python
from aiter.ops.rmsnorm import rms_norm, fused_add_rms_norm_cu

rms_norm(input, weight, epsilon, use_model_sensitive_rmsnorm=0)
# Returns: normalized tensor

fused_add_rms_norm_cu(input, residual_in, weight, epsilon)
# In-place: input = RMSNorm(input + residual_in)

rmsnorm2d_fwd_with_smoothquant(out, input, xscale, yscale, weight, epsilon,
                                use_model_sensitive_rmsnorm=0)

rmsnorm2d_fwd_with_add_smoothquant(out, input, residual_in, residual_out,
                                    xscale, yscale, weight, epsilon,
                                    out_before_quant=None, use_model_sensitive_rmsnorm=0)
```

### LayerNorm
```python
from aiter.ops.norm import layer_norm, layernorm2d_fwd

layer_norm(input, weight=None, bias=None, epsilon=1e-5, x_bias=None)

layernorm2d_fwd(input, weight, bias, epsilon=1e-5)

layernorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, bias, epsilon,
                          x_bias=None)

layernorm2d_fwd_with_smoothquant(out, input, xscale, yscale, weight, bias, epsilon)
```

### GroupNorm
```python
from aiter.ops.groupnorm import _groupnorm_run

_groupnorm_run(input, num_groups, weight, bias, eps)
```

## Activation
```python
from aiter.ops.activation import silu_and_mul, gelu_and_mul, gelu_tanh_and_mul

silu_and_mul(out, input, limit=0.0)   # out = SiLU(first_half) * second_half; limit>0 clamps (gpt-oss)
scaled_silu_and_mul(out, input, scale)
gelu_and_mul(out, input)
gelu_tanh_and_mul(out, input)
gelu_fast(out, input)
```

## Rotary Position Embedding (RoPE)
```python
from aiter.ops.rope import rope_fwd_impl, rope_cached_fwd_impl, rope_cached_positions_fwd_impl

rope_fwd_impl(output, input, freqs, rotate_style, reuse_freqs_front_part, nope_first)
# rotate_style: 0=NEOX (standard), 1=GPT-J (odd elements)

rope_cached_fwd_impl(output, input, cos, sin, rotate_style, reuse_freqs_front_part, nope_first)

rope_cached_positions_fwd_impl(output, input, cos, sin, positions,
                                rotate_style, reuse_freqs_front_part, nope_first)
# positions: [seq_len, batch] — per-token position indices

# 2-channel variants for dual streams
rope_2c_fwd_impl(output_x, output_y, input_x, input_y, freqs, ...)
rope_cached_2c_fwd_impl(output_x, output_y, input_x, input_y, cos, sin, ...)
```

## Quantization
```python
from aiter.ops.quant import pertoken_quant, per_tensor_quant, per_1x32_f4_quant

pertoken_quant(x, scale=None, x_scale=None, scale_dtype=fp32, quant_dtype=i8, dtypeMax=None)
# Returns: (quantized, scales)

per_tensor_quant(x, scale=None, scale_dtype=fp32, quant_dtype=i8)

per_1x32_f4_quant(x, scale=None, quant_dtype=fp4x2, shuffle=False, pack_dim=-1)
# MXFP4: per-1x32 block scaling with E8M0 scale factors

per_1x32_f8_scale_f8_quant(x, scale=None, quant_dtype=fp8, scale_type=fp32, shuffle=False)

# HIP-optimized variants
per_token_quant_hip(x, scale=None, quant_dtype=i8, num_rows=None, num_rows_factor=1)
per_group_quant_hip(x, group_size=128, ...)

# Fused smooth quantization
smoothquant_fwd(out, input, x_scale, y_scale)
moe_smoothquant_fwd(out, input, x_scale, topk_ids, y_scale)
```

## KV Cache Operations
```python
from aiter.ops.cache import reshape_and_cache, concat_and_cache_mla

reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                  kv_cache_dtype, k_scale=None, v_scale=None, asm_layout=False)

reshape_and_cache_with_pertoken_quant(key, value, key_cache, value_cache,
                                      k_dequant_scales, v_dequant_scales,
                                      slot_mapping, asm_layout)

reshape_and_cache_with_block_quant(key, value, key_cache, value_cache,
                                   k_dequant_scales, v_dequant_scales,
                                   slot_mapping, asm_layout)

concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)

swap_blocks(src, dst, block_mapping)
copy_blocks(key_caches, value_caches, block_mapping)
```

## Fused Operations
```python
from aiter.ops.fused_qk_norm_rope_cache_quant import (
    fused_qk_norm_rope_cache_quant_shuffle,
    fused_qk_norm_rope_cache_block_quant_shuffle,
    fused_qk_rope_concat_and_cache_mla,
)

fused_qk_norm_rope_cache_quant_shuffle(
    qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
    qw, kw, cos_sin_cache, is_neox_style, pos_ids,
    k_cache, v_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)

fused_qk_rope_concat_and_cache_mla(
    q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
    k_scale, q_scale, positions, cos_cache, sin_cache,
    is_neox, is_nope_first)
```

## Sampling
```python
top_k_renorm_probs(probs, maybe_top_k_arr, top_k_val)
top_p_sampling_from_probs(probs, indices, maybe_top_p_arr, top_p_val, deterministic=False)
top_k_top_p_sampling_from_probs(probs, indices, maybe_top_k_arr, top_k_val,
                                 maybe_top_p_arr, top_p_val, deterministic=False)
```

## Element-Wise Operations (CK-based)
```python
from aiter.ops.aiter_operator import add, sub, mul, div, sigmoid, tanh
add(input, other)    # Broadcasting supported
mul_(input, other)   # In-place variants with _ suffix
```

## Causal Convolution
```python
from aiter.ops.causal_conv1d_update import causal_conv1d_update

causal_conv1d_update(x, conv_state, weight, bias, out, use_silu,
                     cache_seqlens, conv_state_indices, pad_slot_id)
# Circular buffer mode when cache_seqlens is non-empty
```

## Utility — Weight Shuffle
```python
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_NK, shuffle_weight_a16w4

shuffle_weight(x, layout=(16,16), use_int4=False)
shuffle_weight_NK(x, inst_N, inst_K, use_int4=False)
shuffle_weight_a16w4(src, NLane, gate_up)
```

## Enums
```python
# Canonical location: aiter/ops/enum.py (bound from the C++ module_aiter_core enums).
from aiter.ops.enum import QuantType, ActivationType

QuantType.No           # no quantization
QuantType.per_Tensor
QuantType.per_Token
QuantType.per_1x32     # MX 1x32 block scaling (MXFP4 fp4x2 / MXFP8 fp8)
QuantType.per_1x128    # block-scaled
QuantType.per_128x128  # block-scaled (remapped to per_1x128 on some fused-MoE paths)
QuantType.per_256x128
QuantType.per_1024x128

ActivationType.No
ActivationType.Silu
ActivationType.Gelu
ActivationType.Gelu_Tanh
ActivationType.Swiglu
```
