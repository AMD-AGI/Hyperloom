---
title: sliding_window_attention — tuning
kind: technique
operator: sliding_window_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# sliding_window_attention — tuning

## The one lever that defines SWA: block-skipping
SWA is only faster than full FMHA if the kernel **does not iterate KV blocks fully outside the window**.
For a query block at rows `[m0, m0+BLOCK_M)` and a key block at cols `[n0, n0+BLOCK_N)`:
- **Skip** the key block entirely if `n0 > m0 + BLOCK_M - 1 + window_right` (future, beyond right edge)
  **or** `n0 + BLOCK_N - 1 < m0 - window_left` (too old, fell out of the left window).
- **Partial-mask** only the diagonal-adjacent blocks straddling an edge.
Implement this as the loop bound on the KV `for` loop, not as a post-multiply mask. A Triton kernel that
masks-but-still-loads every block has full-attention cost and zero SWA benefit (the #1 SWA mistake).

## Knobs (Triton FA on AMD — see triton_amd knobs.md)
- `BLOCK_M / BLOCK_N`: 128×64 or 128×128 are good FA defaults on gfx942. Keep `BLOCK_N` ≤ window so the
  band spans only a few KV blocks.
- `matrix_instr_nonkdim=16` (mfma_16x16) — preferred for attention dots.
- `num_warps`: 4 (start); attention is VGPR-heavy — going to 8 risks scratch spill (3–5× slower).
- `num_stages=1` for the fused two-GEMM FA body (stream pipeliner; >1 crushes occupancy on 64 KB LDS).
- `waves_per_eu`: 2–3; `schedule_hint=attention` (or `memory-bound-attention` for decode-skinny).
- `knobs.amd.use_buffer_ops=ON` for the masked edge loads (bounds-checked `buffer_load`).

## Knobs (CK-Tile FMHA — see composable_kernel fmha_template.md)
- Codegen the SWA instance via the FMHA `generate.py` with the `window_generic` mask + `left/right`.
- Block tile `M/N PerBlock`, `M/N PerXDL` as in full FMHA; aim ≥1024 workgroups across 304 CUs.
- The mask object (`make_generic_attention_mask_coordinates_from_lr_window`) already encodes the
  early-out tile range, so CK gets block-skipping for free once you select `window_generic`.

## Decode SWA
- Truncate the paged-KV scan to the last `window` pages. The decode kernel's KV loop bound is
  `min(seq_len, window)` — and the **KV cache itself can evict** pages older than the window (huge HBM
  saving for long-context SWA models). Verify the server actually shrinks the cache, not just the scan.

## Autotune sketch (Triton)
Sweep `(BLOCK_M, BLOCK_N) ∈ {(128,64),(128,128),(64,64)}`, `num_warps ∈ {4,8}`,
`waves_per_eu ∈ {2,3}`, `num_stages=1`. Key the autotune on `(seq, window, head_dim, causal)`.
Re-tune per window — a config tuned at window=1024 is not optimal at window=4096.

## Verify the speedup is real
`window << seq` should give wall-clock roughly proportional to `window`, not `seq`. If doubling `seq` at
fixed `window` doubles prefill time, block-skipping is broken — inspect the KV loop bound / mask.

## Sources
- MI300X attention tuning (2-GEMM fusion, OPTIMIZE_EPILOGUE, ≥1024 grid): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Triton AMD knobs (num_stages, waves_per_eu, schedule_hint, buffer_ops): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- CK window mask early-out: `ROCm/composable_kernel:example/ck_tile/01_fmha/mask.hpp` (on-box).
