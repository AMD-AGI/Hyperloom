---
title: layernorm — tuning
kind: technique
operator: layernorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/norm.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
---

# layernorm — tuning

Same bandwidth-bound playbook as rmsnorm (read rmsnorm/tuning.md first), plus
the cost of a **second statistic** (mean) and a **bias** add. The lever set is identical; the only new
decision is **two-pass vs Welford** for computing μ and σ² in one row sweep.

## 1. Two-pass vs one-pass Welford
- **Two-pass** (what aiter's Triton impl does): pass A computes `μ = Σx/N`, pass B computes
  `σ² = Σ(x−μ)²/N`. Re-reads the row from registers/LDS (not HBM if N fits), so on-chip it's cheap; for
  blocked N>block it re-reads from HBM (2× x traffic).
- **One-pass Welford**: streams once, updating running `(count, mean, M2)` — one HBM read even for huge N.
  Numerically the most stable. Use it when **N > block** (blocked path) to avoid the second HBM read.
- For real hidden dims (N ≤ 5120 ≤ 32768-bf16-block) the row fits → two-pass on-chip is simplest and
  fastest; Welford only pays off in the blocked regime or training with very large N.

## 2. The aiter Triton heuristics (verified)
```python
BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))   # MAX_FUSED_SIZE = 65536 // elt_size
USE_BLOCKED = N > BLOCK_SIZE                            # → two-pass-over-HBM blocked kernel
num_warps = min(max(BLOCK_SIZE // 256, 1), 8)          # 2–4 typical; cap 8
# backward dwdb reduction tiled: BLOCK_SIZE_M=dwdb_block_m, BLOCK_SIZE_N=dwdb_block_n
# blocked path clamps BLOCK_SIZE to 2048 (fp32) / 4096 (else) to bound LDS
```
Grid: row-per-program (prefill) or persistent `min(M, num_sms)` (decode).

## 3. Knob table
| knob | layernorm setting | note |
|---|---|---|
| `num_warps` | 2–4 (`min(max(BLOCK//256,1),8)`) | memory-bound; 8 spills |
| `BLOCK_SIZE` | next_pow2(N), clamp 2048/4096 in blocked | full wave reduce, 128-bit loads |
| pass scheme | two-pass on-chip (N≤block) / Welford (N>block) | avoid 2nd HBM read |
| grid | row-per-prog / `min(M,num_sms)` | fill 304 CUs |
| `waves_per_eu` | 3–4 | VGPR-light |
| `cache_modifier` | `.cg` on x | read-once |

## 4. Backward (training)
Saves `mean,rstd` from forward; `dγ,dβ` need a **cross-row reduction** → a second tiled reduce kernel
(`BLOCK_SIZE_M=128, BLOCK_SIZE_N=64` in aiter). Keeping `dγ,dβ` in registers scales only to ~8k hidden
(flash-attn note: register spill beyond 8k) — for larger N accumulate partials to global/LDS.

## 5. Vectorized I/O + bias
Read x as `float4`/`__half2`; `γ,β` are small (N elements) → cache in LDS once per block and reuse across
rows in the persistent loop. Bias-add `+β` is free (in the epilogue, in-register).

## Sources
- aiter Triton heuristics (BLOCK_SIZE clamp, num_warps, two-pass, blocked, dwdb tiles): `aiter/ops/triton/normalization/norm.py`.
- Memory-bound knobs + 128-bit loads: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- Two-pass / register-spill-beyond-8k / Welford: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html, https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py.
