---
title: kv_cache_quant on aiter — SOTA card
kind: sota_card
operator: kv_cache_quant
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [decode, prefill, both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - ROCm/aiter@b467ce342:csrc/kernels/cache_kernels.cu
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
---

# kv_cache_quant × aiter

## TL;DR
aiter owns the **fully-fused** KV store path. `fused_qk_norm_rope_cache_quant_shuffle` collapses
QK-norm + RoPE + KV-write and optional FP8 quant/layout shuffle into **one** kernel, with **per-tensor**
(`pts`) and per-**block** scale variants and an **mRoPE** variant for multimodal. The `auto` cache dtype
keeps BF16/FP16 K/V without quantization while retaining the same fused producer boundary. Underneath,
the plain store path is
`reshape_and_cache_flash_kernel` (per-tensor scale) and `reshape_and_cache_with_per_token_quant_kernel`
(per-token/per-head scale) in `cache_kernels.cu`. This is the production decode path on sglang/vLLM with
`--kv-cache-dtype fp8`; the read-side dequant lives in aiter's paged / MLA attention. Use it whenever you
run FP8 KV — but the **store shuffle layout must match the attention read** (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`),
and accuracy-gate KV+MLA combos (a gsm8k regression has been observed).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `fused_qk_norm_rope_cache_pts_quant_shuffle` | `aiter/ops/fused_qk_norm_rope_cache_quant.py:228` | gfx942/950, bf16/fp16/e4m3 | fused QK-norm+RoPE+write; optional per-tensor quant/shuffle | standard RoPE with BF16/FP16 or FP8 KV |
| `fused_qk_norm_rope_cache_block_quant_shuffle` | `:204` | gfx942/950 | per-block KV scale | accuracy-sensitive KV |
| `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | `aiter/ops/fused_qk_norm_mrope_cache_quant.py:10` | gfx942/950 | mRoPE (3D) + KV quant | multimodal |
| `reshape_and_cache_flash_kernel` (per-tensor, `opus::cast`) | `csrc/kernels/cache_kernels.cu:245` | gfx942/950 | `q = cast(kv * (1/k_scale))` | non-fused store, per-tensor |
| per-token/per-head amax store-quant | `cache_kernels.cu:405` | gfx942/950 | per-token/per-head amax scale | per-token KV quant |

### SOTA excerpt — per-tensor KV store-quant (`cache_kernels.cu:245`)
```cpp
const float inverted_kscale = 1 / (*k_scale);          // store 1/s → per-elem op is a multiply
const float inverted_vscale = 1 / (*v_scale);
for (int i = threadIdx.x; i < num_heads*head_size; i += blockDim.x) {
  if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {        // no quant
    key_cache[tgt]   = key[src];
    value_cache[tgt] = value[src];
  } else {                                              // FP8 KV
    key_cache[tgt]   = opus::cast<cache_t>(float(key[src])   * inverted_kscale);
    value_cache[tgt] = opus::cast<cache_t>(float(value[src]) * inverted_vscale);
  }
}
```

### SOTA excerpt — per-token / per-head amax scale (`cache_kernels.cu:405`)
```cpp
float k_max = wave_reduce(k_local_max, f_max_f32);     // per-token amax over the head dim
constexpr float k_pertoken_quant_scale_eps = 1e-12f;
float k_token_scale          = k_max / dtypeMax;       // dtypeMax → fnuz(224/240) vs ocp(448)
float k_token_scale_inverted = 1.0f / fmaxf(k_token_scale, k_pertoken_quant_scale_eps);
// store the scale at [num_blocks, num_heads, block_size] (asmLayout) for the FA read
k_dequant_scales[scale_idx] = k_token_scale;
```

## Config space / knobs
| knob | values | effect |
|---|---|---|
| `kv_cache_dtype` | "fp8" / "fp8_e4m3" / "auto" | quant on/off + dialect; `auto` preserves BF16/FP16 |
| scale granularity | per-tensor (`pts`) / per-block / per-token-per-head | `k_scale`/`v_scale` scalar vs block vs `[blocks,heads,block_size]` |
| `use_shuffle_layout` | True/False | paged layout for FA read coalescing; **must match the read** |
| `block_size` / `x` | paged params | KV page tiling; `x = 16 // elem_size` (5D shuffle layout) |
| `is_neox_style` / `rotary_dim` | bool / int | RoPE style |
| `slot_mapping` / `cu_q_len` | tensors | varlen / padded slots (slot −1 = skip) |
| mRoPE | mrope variant | multimodal positions |

## Measured performance
| config | metric | value @ hw / ver / date | source |
|---|---|---|---|
| fused QK-norm+RoPE+KV-quant vs separate ops | kernels | 1 fused launch vs ~4 separate passes | `fused_qk_norm_rope_cache_quant.py` |
| BF16 QK-norm+RoPE+KV-write, T16/Hq32/Hk8/D128 | latency | 7.84 us standalone on gfx950 | local exact-shape measurement |
| FP8 KV vs bf16 KV | capacity | ~2× KV bytes saved → larger batch/context | format (1 byte vs 2) |
| AITER MLA + FP8 KV | accuracy | gsm8k regression observed (#1455) — gate it | numerics.md |
| fused RMS fallback | threshold | `_FUSED_QK_FALLBACK_M = 16384` → rmsnorm2d_fwd | `fused_qk_norm_rope_cache_quant.py:174` |

> KV-quant gain is capacity/bandwidth, not FLOPs. Measure the max batch×context delta and gate accuracy;
> no vendor speedup asserted.

## Numerics / parity
- **Dialect:** fnuz on gfx942 / ocp on gfx950 — **must match the attention read dialect** or KV is read
  ~2× wrong. `dtypeMax` drives the scale (`k_max/dtypeMax`).
- Online softmax stays **fp32**; only the stored K/V are FP8.
- Per-token path adds `k_pertoken_quant_scale_eps = 1e-12` to avoid div-by-0; scales stored at
  `[num_blocks, num_heads, block_size]` (asmLayout) so the FA read indexes them per token.
- **Gate:** gsm8k. **AITER MLA+KV gsm8k regression observed (#1455)** — accuracy-gate KV+MLA combos →
  numerics.md.

## Integration (rebind seam)
- Live decode path: `aiter.ops.fused_qk_norm_rope_cache_*`; in vLLM via `--kv-cache-dtype fp8` plus
  `VLLM_ROCM_USE_AITER_MHA` / `VLLM_ROCM_USE_AITER_MLA`.
- The store shuffle layout **must** match `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` used on the FA read — the
  5D shuffle cache is `[B, Hv, page//x, D, x]` with `x = 16 // elem_size`. → ../../../reference/env_vars.

## Pitfalls & anti-patterns
- **Store/read shuffle-layout mismatch** → wrong KV (the most common KV-FP8 bug). Set
  `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` consistently on write and read.
- **FNUZ↔OCP mismatch** between store and attention read → ~2× error
  (../../../quantization/fnuz_vs_ocp).
- **Per-tensor vs per-block/per-token scale inconsistent** between write and read.
- mRoPE without the `fused_qk_norm_mrope_cache_quant` variant → wrong multimodal positions.
- Forgetting slot_mapping = −1 means "padded token, skip" — writing it corrupts a real slot.

## How to verify
- e2e gsm8k with and without FP8 KV; confirm the attention backend reads the **same** dialect + shuffle
  layout.
- Report the max batch / context gain at fixed memory.
- rocprofv3 → confirm the fused `fused_qk_norm_rope_cache_*` symbol (one launch) ran, not separate
  norm/rope/quant kernels.

## Alternatives / cross-links
vllm_kernels · hip · triton ·
[overview.md](overview.md) · [numerics.md](numerics.md) · [tuning.md](tuning.md) ·
operators/attention_decode_paged · operators/rope · operators/mrope ·
../../../quantization/kv_cache_quantization · ../../../quantization/fnuz_vs_ocp.

## Worked example
Enable fused FP8 KV store on a decode model in vLLM on gfx942:
```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1                 # FA read must speak the same path
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1       # store + read must agree
vllm serve <model> --kv-cache-dtype fp8          # → fused_qk_norm_rope_cache_pts_quant_shuffle
```
The QK-norm + RoPE + KV-write + FP8 quant + shuffle now run in one aiter kernel; the FA read dequants with
the matching per-tensor `k_scale`/`v_scale` in the shuffled layout. Gate gsm8k before shipping — especially
with MLA (#1455).

## Sources
- aiter fused QK-norm+RoPE+KV-quant (pts/block/mrope):
  `ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_rope_cache_quant.py`
  (`fused_qk_norm_rope_cache_quant_shuffle:39`, `..._block_quant_shuffle:204`, `..._pts_quant_shuffle:228`,
  `_FUSED_QK_FALLBACK_M:174`), `aiter/ops/fused_qk_norm_mrope_cache_quant.py`
  (`fused_qk_norm_mrope_3d_cache_pts_quant_shuffle:10`).
- KV store-quant kernels (per-tensor `opus::cast` + per-token/per-head amax):
  `ROCm/aiter@b467ce342:csrc/kernels/cache_kernels.cu`
  (`reshape_and_cache_flash_kernel:245`, per-token amax + `k_pertoken_quant_scale_eps` at `:405`).
- KV shuffle / kv-cache-dtype env (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`):
  https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
