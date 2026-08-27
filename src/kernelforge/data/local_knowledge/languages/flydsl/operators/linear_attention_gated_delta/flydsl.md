---
title: linear_attention_gated_delta on FlyDSL (gdr_decode) — SOTA card
kind: sota_card
operator: linear_attention_gated_delta
backend: flydsl
gens: [gfx950]
dtypes: [bf16, fp16]
regimes: [decode]
status: experimental
updated: 2026-06-09
---

# linear_attention_gated_delta × FlyDSL (gdr_decode)

## TL;DR
`flydsl_gdr_decode` is a FlyDSL **gated-delta-rule decode** kernel (gdr = gated-delta-rule) — the
single-step recurrent state update for gated DeltaNet / linear attention. It fuses sigmoid gating,
optional QK L2-norm, the delta-rule state update, and the output projection into one compiled kernel, with
a per-shape tuning table loaded from `gdr_decode_tuned.jsonl`.

**Status: experimental.** In `aiter/ops/flydsl/__init__.py` the line
`from .linear_attention_kernels import flydsl_gdr_decode` is **commented out**, so the symbol is present in
source but NOT wired into the default `aiter.ops.flydsl` import (and not in `__all__`). It must be imported
directly from `aiter.ops.flydsl.linear_attention_kernels`. The on-box tuning file is gfx950-only.

## SOTA implementation
`flydsl_gdr_decode` validates contiguity / 16-byte alignment / dtype, optionally permutes the state,
looks up the tuned config, builds the cached kernel, and launches. From
`/sgl-workspace/aiter/aiter/ops/flydsl/linear_attention_kernels.py`:

```python
def flydsl_gdr_decode(query, key, value, a, b, dt_bias, A_log, indices,
                      state, out, use_qk_l2norm, need_shuffle_state,
                      stream=torch.cuda.current_stream()):
    for input in [key, value, a, b, dt_bias, A_log, indices, out]:
        assert input.is_contiguous()
        assert input.data_ptr() % 16 == 0
    assert state.dtype in [torch.float, torch.bfloat16]
    ...
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    kwargs_ = get_default_kwargs(str(dtype), str(state.dtype), batch_size,
                                 seq_length, num_k_heads, num_v_heads,
                                 head_k_dim, head_v_dim)
    exe = create_shuffle_gdr_decode_kernel(..., state.stride(),
                                           use_qk_l2norm, **kwargs_)
```

The kernel builder `create_shuffle_gdr_decode_kernel` (`kernels/gdr_decode.py`, `lru_cache(maxsize=1024)`)
derives the warp tiling from `(NUM_WARPS, WARP_THREADS_K, NUM_BLOCKS_PER_V_DIM)`, asserts `WARP_SIZE==64`,
and emits the gating math (`g = -exp(A_log)·softplus(a+dt_bias)`, `beta = sigmoid(b)`) plus the recurrent
`h *= exp(g); v -= sum(h*k); v *= beta; h += k⊗v; o = sum(h*q)` update.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `flydsl_gdr_decode` (fused gated-delta decode) | `linear_attention_kernels.py` + `kernels/gdr_decode.py` | gfx950 (tuned table); bf16/fp16 in/out, fp32 or bf16 state | per-shape `duration` in `gdr_decode_tuned.jsonl` (autotune timings, not a comparative claim — e.g. ~4.4 µs for b=1,sq=1,kh2×128,vh8×128 on gfx950) | single-step GDN/linear-attn decode, experimental |

## Config space / knobs
From `get_default_kwargs` + `create_shuffle_gdr_decode_kernel` (`linear_attention_kernels.py`,
`kernels/gdr_decode.py`). The tuned table overrides the defaults per shape.

| param | range / typical | effect | default |
|---|---|---|---|
| `NUM_WARPS` | tuned (e.g. 4) | warps per block (BLOCK_THREADS = NUM_WARPS·64) | 4 |
| `WARP_THREADS_K` | tuned (e.g. 8, 16) | K-dim threads per warp (WARP_THREADS_V = 64/this) | 8 (default fn: WARP_THREADS_K=16) |
| `NUM_BLOCKS_PER_V_DIM` | tuned (e.g. 1, 4, 8) | V-dim block split | 1 |
| `use_qk_l2norm` | bool | apply L2-norm to Q,K in-kernel | caller |
| `need_shuffle_state` | bool | permute state `(0,1,3,2)` in/out around the kernel | caller |
| `softplus_beta` / `softplus_threshold` | 1.0 / 20.0 | softplus shape + numerical-stability cutoff | 1.0 / 20.0 |
| state dtype | fp32 / bf16 | fp32 → VALUES_PER_THREAD_K=4 (16B), else 8 | — |

## Numerics / parity
fp32 accumulation in the recurrence; SiLU/sigmoid/softplus done in fp32. `SCALE = 1/sqrt(head_k_dim)`
applied to Q. State (`h`) may be fp32 or bf16. Asserts: in/out tensors share `query.dtype` (bf16/fp16),
`A_log` ∈ {fp32, bf16}, `indices` int32, all 16-byte aligned. The on-box regression
(`test_flydsl_linear_attention.py`) compares against a Triton fused-sigmoid-gating delta-rule reference at
`atol=1e-3, rtol=1e-3` for both output and final state. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
**Not** exported from `aiter.ops.flydsl` (import line commented out). To use it:
```python
from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
```
The tuning table `gdr_decode_tuned.jsonl` is keyed on
`(dtype, state_dtype, arch, b, sq, num_k_heads, num_v_heads, head_k_dim, head_v_dim)` — only gfx950 rows are
present on-box, so off-table shapes fall back to the default kwargs. The kernel is `lru_cache`d on its full
config tuple. No env overlay.

## Pitfalls & anti-patterns
- **Experimental / not wired in**: don't assume `from aiter.ops.flydsl import flydsl_gdr_decode` works — it's
  commented out; import from the submodule and treat as unverified for production.
- All inputs must be contiguous and 16-byte aligned, and `state`/`A_log` permitted dtypes are narrow —
  violations hit asserts, not graceful fallback.
- Tuned table is gfx950-only; on gfx942 or untuned shapes you get default kwargs (likely suboptimal).
- `need_shuffle_state=True` does an extra permute+contiguous copy of the state on entry and exit.
- The `duration` values in the jsonl are autotune timings, not a head-to-head speedup vs another backend.

## How to verify
```bash
pytest -sv /sgl-workspace/aiter/aiter/ops/flydsl/test_flydsl_linear_attention.py
# parametrized bf16/fp16, b∈{1,2,128}, vs Triton GDN reference, atol/rtol 1e-3.
```

## Alternatives / cross-links
[[operators/linear_attention_gated_delta/backends/triton]] (sota: FLA reference + aiter on-box GDN port,
sglang default) · [[operators/linear_attention_gated_delta/backends/hip]] · [[operators/dense_gemm/backends/flydsl]]
(FlyDSL language) · language deep-dive `languages/flydsl/` (P1).
