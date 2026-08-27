---
title: attention_prefill_fmha on Triton — SOTA card
kind: sota_card
operator: attention_prefill_fmha
backend: triton
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
---

# attention_prefill_fmha × Triton

## TL;DR (one-line decision)
> Triton FA is the **editable** prefill attention and the **most feature-rich** path (fp32 + fp8 +
> arbitrary head dim + ALiBi + softcap + sliding-window + sinks + varlen/paged). It is the
> `--attention-backend triton` swap target and the FA backend whose kernels come from **`ROCm/aiter`**.
> It usually **loses ~1.5× to CK-Tile / TileLang** on plain bf16 prefill (TileLang FA ≈1.53× Triton on
> MI300X), so reach for it when you need (a) a feature CK can't do for your shape, or (b) an editable
> `@triton.jit` kernel to modify and e2e-gate.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| aiter Triton unified attention | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/unified_attention.py` | gfx90a/942/950; fp16/bf16/fp32/fp8; **causal GQA, softcap, SWA, alibi, sinks** | one kernel for chunked-prefill + decode; feature ceiling | mixed batches, SWA/softcap/sinks |
| aiter Triton FA (`mha.py`, `mha_v3.py`) | same repo | fp8 / head_dim>256 / ALiBi | ~0.65× of TileLang FA fwd on MI300X (TileLang 1.53× Triton, vendor) | fp8 / large d / editable |
| FA-ROCm Triton backend (same aiter kernels) | `Dao-AILab/flash-attention` (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) | as above | FA-2/FA-v3 interface | drop-in `flash_attn_func` |

**Real code** — `unified_attention` packs GQA into `BLOCK_M` and computes `BLOCK_Q` per kv-head
(`unified_attention.py`):
```python
num_queries_per_kv = num_query_heads // num_kv_heads
BLOCK_M = (16 if num_queries_per_kv <= 16
           else triton.next_power_of_2(num_queries_per_kv))
BLOCK_Q = BLOCK_M // num_queries_per_kv          # query rows per program
if max_seqlen_q >= 256:                            # large-prefill config
    BLOCK_M = 64 if arch.is_rdna else 128
    num_stages_2d, num_warps = 1, 4
```
Softcap is computed in the inner loop as a numerically-stable `tanh` via `exp2`
(`_triton_kernels/attention/unified_attention.py`):
```python
def apply_softcap(S, x):
    Sdiv = S / x
    p1, p2 = tl.math.exp2(Sdiv), tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)             # == x*tanh(S/x)
# ... USE_SOFTCAP: S = apply_softcap(S, softcap) * RCP_LN2
# causal: seq_mask = seq_offset < context_len + query_pos + 1
# SWA:    (context_len + query_pos - seq_offset) < SLIDING_WINDOW
# fp8:    qk_scale *= q_descale; qk_scale *= k_descale; one_over_L = v_descale / L
```

## Config space / knobs
| param | range / values | effect | default (large prefill) |
|---|---|---|---|
| `BLOCK_M` | 64 / 128 (16 if GQA ratio ≤16) | Q-rows tile | 128 (64 RDNA) |
| `BLOCK_N`/`TILE_SIZE` | 16 / 32 / 64 | KV tile (64 CDNA, 16 RDNA, 32 gfx1201) | 64 |
| `num_warps` | 4 / 8 (wave64!) | parallelism | 4 |
| `num_stages` | **1** for fused FA (≤2 if d>128) | SW pipeline depth | 1 |
| `waves_per_eu` | 1–3 (CDNA), 6 RDNA | occupancy | 2 (CDNA) |
| `matrix_instr_nonkdim` | 16 | MFMA 16×16 | 16 |
| `kpack` | 2 (gfx942) | LDS pack | 2 |
| `schedule_hint` | `attention`/`memory-bound-attention` | scheduler | attention |

FA-ROCm pins a config without autotune via
`FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":128,"BLOCK_N":64,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":8}'`,
or `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` for one-time warmup search. See
`triton_levers/knobs.md` and [tuning.md](tuning.md).

## Numerics / parity
fp32 online-softmax accumulate (same as FA-2); P/O stored bf16. fp8 uses `q_descale`/`k_descale` applied
to `qk_scale` *before* softmax and `v_descale` folded into `1/L` after — **FNUZ on gfx942, OCP on
gfx950** (wrong dialect off ~2×). Softcap uses `exp2`-based tanh (above). Cross-backend bf16 argmax
tie-flips are benign. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **sglang:** `--attention-backend triton` (and `--prefill-attention-backend triton`).
- **vLLM:** Triton FA is the historical ROCm default (`VLLM_USE_TRITON_FLASH_ATTN`); on V1 select
  `TRITON_ATTN` explicitly (the legacy flag may be ignored).
- The `@triton.jit` body is the **Tier-C rewrite seam** — edit, autotune, e2e-gate.
- **Verify it engaged:** rocprofv3 Top-N shows Triton names (`_attn_fwd_*` / `kernel_unified_attention_*`).

## Pitfalls & anti-patterns
- Carrying `num_warps=8` + `num_stages=3` from NVIDIA → VGPR spill / worse pipelining; FA wants
  `num_stages=1`.
- **`unified_attention` asserts `causal=True`** (`assert causal, "Only causal attention is supported"`) —
  not a general bidirectional FMHA; use `flash_attn_func` for non-causal.
- Sliding-window was historically WIP in FA-ROCm Triton → check your version, or use CK for SWA.
- Shared-mem/register-pressure compile failures on some models (Phi3V LDS overflow) → fall back to CK.
- Don't expect to beat CK-Tile/TileLang on plain bf16; the win is features or editability.

## Worked example
Gemma-2 (softcap=50, GQA, SWA on alternating layers) prefill, bf16:
1. Gemma-2 needs **softcap + SWA** → Triton unified attention is the natural fit (CK softcap support is
   spottier). Serve `--attention-backend triton`.
2. Confirm `kernel_unified_attention_*` in rocprofv3.
3. Bench TTFT; accept the ~1.5× bf16 gap vs CK as the price of softcap+SWA in one kernel.

## How to verify (bench + oracle)
`AMDGCN_ENABLE_DUMP=1` → want `global_load_dwordx4`, `ds_read_b128`, `v_mfma_*16x16`, no `scratch_`.
Isolated FMHA bench vs CK at the same `(B,H,sq,sk,d,causal,dtype)`; greedy temp=0 parity over ≥10 prompts.

## Alternatives / cross-links
./ck.md · ./tilelang.md · ./aiter.md · ./asm.md · ../../attention_decode_paged/backends/triton.md ·
../../mla_attention/backends/triton.md · `triton_levers/` · ../overview.md · ../tuning.md.

## Sources
- aiter Triton kernels (on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py`, `_triton_kernels/.../unified_attention.py`: BLOCK_M/BLOCK_Q GQA packing, softcap exp2-tanh, causal/SWA/alibi/fp8).
- FA-ROCm Triton backend = aiter kernels, config-json / autotune env: https://github.com/Dao-AILab/flash-attention
- Triton AMD knobs (num_stages=1 FA, schedule_hint, wave64): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- TileLang FA 1.53× Triton (relative ranking, vendor): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
