# AITER Integration Patterns

## Quantization Flows

### FP8 Per-Token Quantization (Activation)
```python
from aiter.ops.quant import pertoken_quant

# Dynamic quantization: compute scale per token, quantize
x_fp8, x_scale = pertoken_quant(x_bf16, quant_dtype=torch.float8_e4m3fn)
# x_fp8: [M, K] fp8, x_scale: [M, 1] fp32
# Scale = max(|x_row|) / fp8_max
```

### MXFP4 Per-1x32 Block Quantization
```python
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f4_quant_for_dot_scaled

# Single tensor
x_fp4, x_scale = per_1x32_f4_quant(x, quant_dtype=torch.float4_e2m1fn_x2,
                                     shuffle=True, pack_dim=-1)
# x_fp4: [M, K//2] packed, x_scale: [M, K//32] e8m0

# Both operands for GEMM
lhs_fp4, lhs_s, rhs_fp4, rhs_s = per_1x32_f4_quant_for_dot_scaled(lhs, rhs)
```

### Smooth Quantization (Channel-wise)
```python
from aiter.ops.quant import smoothquant_fwd

# Apply channel-wise smooth scale + output quantization
smoothquant_fwd(out_int8, input_bf16, x_scale_per_channel, y_scale_per_token)
```

## GEMM Integration

### Standard A8W8 GEMM Flow
```python
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_ck
from aiter.ops.quant import pertoken_quant

# 1. Quantize activations dynamically
XQ, x_scale = pertoken_quant(X, quant_dtype=torch.int8)

# 2. Weights are pre-quantized offline
# WQ: [N, K] int8, w_scale: [1, N] fp32

# 3. GEMM
Out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Out, bias=bias)
```

### Weight Pre-Shuffle for ASM GEMM
```python
from aiter.ops.shuffle import shuffle_weight

# Pre-shuffle weights to (16,16) tile layout for ASM kernels
WQ_shuffled = shuffle_weight(WQ, layout=(16, 16))

# Use with bpreshuffle=True
gemm_a8w8_asm(XQ, WQ_shuffled, x_scale, w_scale, Out, bpreshuffle=True)
```

## MoE Integration

### Complete MoE Forward Pass
```python
from aiter.ops.moe_op import (
    topk_softmax, moe_align_block_size, fmoe_g1u1, moe_sum
)

# 1. Expert gating
topk_weights = torch.empty(num_tokens, topk, device='cuda')
topk_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device='cuda')
token_expert_indices = torch.empty_like(topk_indices)
topk_softmax(topk_weights, topk_indices, token_expert_indices,
             gating_output, need_renorm=True)

# 2. Block alignment (MANDATORY before fused MoE)
block_size = 32  # Must match kernel expectation
sorted_token_ids = torch.empty(num_tokens * topk + num_experts * block_size,
                                dtype=torch.int32, device='cuda')
experts_ids = torch.empty(num_experts, dtype=torch.int32, device='cuda')
token_nums = torch.empty(num_experts, dtype=torch.int32, device='cuda')
num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device='cuda')

moe_align_block_size(topk_indices, num_experts, block_size,
                     sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad)

# 3. Fused MoE GEMM (gate_up + activation + down projection)
out = torch.empty(num_tokens, hidden_dim, dtype=torch.bfloat16, device='cuda')
fmoe_g1u1(out, hidden_states, gate_weight, down_weight,
           sorted_token_ids, topk_weights, experts_ids, token_nums, topk,
           input_scale, fc1_scale, fc2_scale,
           activation=ActivationType.Silu.value)

# 4. Sum expert outputs (if topk > 1)
final_out = torch.empty(num_tokens, hidden_dim, dtype=torch.bfloat16, device='cuda')
moe_sum(out, final_out)
```

## KV Cache Integration

### Standard KV Cache Update
```python
from aiter.ops.cache import reshape_and_cache

# key, value: [batch, seq_len, num_heads, head_size]
# key_cache: [num_blocks, block_size, num_heads, head_size]
# slot_mapping: [batch * seq_len] — maps tokens to cache slots

reshape_and_cache(key, value, key_cache, value_cache,
                  slot_mapping, kv_cache_dtype="auto")
```

### FP8 KV Cache with Per-Token Quantization
```python
from aiter.ops.cache import reshape_and_cache_with_pertoken_quant

# Quantize K/V on the fly during caching
reshape_and_cache_with_pertoken_quant(
    key, value,
    key_cache_fp8, value_cache_fp8,
    k_dequant_scales, v_dequant_scales,  # Per-token scales
    slot_mapping, asm_layout=True)
```

### MLA (Multi-Latent Attention) Cache
```python
from aiter.ops.cache import concat_and_cache_mla

# Concatenate compressed KV and position encoding into cache
concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                     kv_cache_dtype="auto", scale=kv_scale)
```

### Fused QKNorm + RoPE + Cache + Quant
```python
from aiter.ops.fused_qk_norm_rope_cache_quant import (
    fused_qk_norm_rope_cache_quant_shuffle
)

# Single kernel: normalize Q/K → apply RoPE → cache K/V → quantize
fused_qk_norm_rope_cache_quant_shuffle(
    qkv,                    # [tokens, (q+k+v) * head_dim]
    num_heads_q, num_heads_k, num_heads_v, head_dim,
    eps=1e-5,
    qw=q_norm_weight,       # Q normalization weight
    kw=k_norm_weight,       # K normalization weight
    cos_sin_cache=cos_sin,  # Pre-computed RoPE cos/sin
    is_neox_style=True,
    pos_ids=positions,
    k_cache=key_cache,
    v_cache=value_cache,
    slot_mapping=slot_mapping,
    kv_cache_dtype="fp8",
    k_scale=k_scale,
    v_scale=v_scale,
)
```

## Attention Integration

### Decode with Paged Attention (ASM)
```python
from aiter.ops.attention import pa_fwd_asm

output = pa_fwd_asm(
    Q=query,                    # [num_seqs, num_heads, head_size]
    K=key_cache,                # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    V=value_cache,              # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables=block_tables,  # [num_seqs, max_blocks_per_seq]
    context_lens=context_lens,  # [num_seqs]
    block_tables_stride0=block_tables.stride(0),
    max_qlen=1,                 # Decode: always 1
    high_precision=1,           # 2 for FP8 KV cache
)
```

### Prefill with Flash Attention (CK)
```python
from aiter.ops.attention import mha_fwd

out, softmax_lse, _, _ = mha_fwd(
    q=query,    # [batch, seqlen_q, num_heads, head_size]
    k=key,      # [batch, seqlen_k, num_kv_heads, head_size]
    v=value,    # [batch, seqlen_k, num_kv_heads, head_size]
    dropout_p=0.0,
    softmax_scale=1.0 / math.sqrt(head_size),
    is_causal=True,
    window_size_left=-1, window_size_right=0,
    sink_size=0,
    return_softmax_lse=True,
    return_dropout_randval=False,
)
```

## Test & Benchmark Patterns

### Standard Operator Test
```python
from aiter.test_common import checkAllclose, perftest

@perftest(num_iters=101, num_warmup=2)
def test_gemm():
    XQ, x_scale = pertoken_quant(X, quant_dtype=torch.int8)
    Out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Out)
    return Out

result = test_gemm()
checkAllclose(result, reference, atol=1e-2, rtol=1e-2)
```

### Standard Tolerance by Dtype
| Dtype | atol | rtol | Notes |
|-------|------|------|-------|
| BF16 | 1e-2 | 1e-2 | Standard |
| FP16 | 1e-3 | 1e-3 | Higher precision |
| FP8 fwd | 0.3 | 0.1 | Wide tolerance |
| FP8 bwd | 1.0 | 0.1 | Very wide |
| INT8 | 1e-1 | 1e-1 | Quantization noise |
| FP8 cosine sim | ≥ 0.96 | — | Alternative check |

### Standard Test Shapes
```python
# GEMM: LLaMA 3 70B shapes
gemm_shapes = [
    (M, 10240, 8192) for M in [1, 128, 256, 4096]
]

# Attention
attn_configs = [
    (batch=1, seqlen_q=4096, nheads=32, d=128),
    (batch=4, seqlen_q=128, nheads=48, d=128),  # GQA
]

# MoE: Mixtral pattern
moe_shapes = [
    (tokens=4096, hidden=4096, inter=14336, experts=8, topk=2),
]
```

## Framework Integration

### vLLM / SGLang Integration Points
AITER operators are designed to be drop-in replacements in inference frameworks:

```python
# Replace vLLM's attention with AITER
from aiter.ops.attention import paged_attention_v1

# Replace vLLM's GEMM with AITER
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_ck

# Replace vLLM's cache ops
from aiter.ops.cache import reshape_and_cache
```

Key integration requirements:
1. KV cache layout must match: `[num_blocks, block_size, num_heads, head_size]`
2. Slot mapping must be pre-computed by the scheduler
3. Block tables must be contiguous int32 tensors
4. Quantization scales must be pre-computed for weight tensors

## Performance Comparison Template

```python
import torch
from aiter.test_common import perftest

@perftest(num_iters=100, num_warmup=25)
def bench_aiter():
    return aiter_op(...)

@perftest(num_iters=100, num_warmup=25)
def bench_baseline():
    return baseline_op(...)

aiter_result = bench_aiter()
baseline_result = bench_baseline()

# Report format matching AITER Fellow reporting
print(f"AITER: {aiter_result.median_ms:.3f} ms")
print(f"Baseline: {baseline_result.median_ms:.3f} ms")
print(f"Speedup: {baseline_result.median_ms / aiter_result.median_ms:.2f}x")
```
