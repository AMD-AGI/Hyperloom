---
title: batched_gemm on aiter — SOTA card
kind: sota_card
operator: batched_gemm
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:aiter/ops/batched_gemm_op_bf16.py
  - ROCm/aiter@b467ce342:csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py
  - https://github.com/ROCm/aiter
---

# batched_gemm × aiter

## TL;DR
> aiter has **two batched-GEMM paths**: (1) the `tuned_gemm` dense engine flattens a 3-D `A` to 2-D
> (`gemm_a16w16`, `batched=True`) and dispatches per shape from the dense DB; (2) a **dedicated CK
> strided-batched kernel** `batched_gemm_bf16` (`aiter/ops/batched_gemm_op_bf16.py`) with its **own** DB
> keyed on `(gfx, cu_num, B, M, N, K)`, tuned by `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` and
> deployed via `AITER_CONFIG_BF16_BATCHED_GEMM`. To improve batched GEMM, tune the matching DB. Most
> attention batched matmuls never reach either path (FMHA fuses them) — see [overview.md](overview.md).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `tuned_gemm` dense engine (3-D→2-D flatten) | `aiter/tuned_gemm.py:356-395` (`gemm_a16w16`, `batched=True`) | gfx942/950; bf16, fp8 scaled | inherits the dense-GEMM tuning mechanism (**+2.23% e2e** on the dense path, Qwen3.5-27B/sglang, MI300X, 2026-06-08) | batched linear where per-token flatten is valid |
| **dedicated CK strided-batched** `batched_gemm_bf16` | `aiter/ops/batched_gemm_op_bf16.py` (`batched_gemm_bf16`, `get_CKBatchedGEMM_config`) + `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` | gfx942/950; bf16 (a8w8 sibling: `batched_gemm_op_a8w8.py`) | own `(gfx,cu_num,B,M,N,K)` DB; auto split-K via `compute_batched_gemm_SplitK` | true strided-batched (per-batch distinct A and B) |

## Config space / knobs
- **Dense flatten path**: capture `AITER_TUNE_GEMM=1` (bias must match live) → dense untuned CSV; tune via
  `csrc/gemm_a16w16/gemm_a16w16_tune.py`; deploy `AITER_CONFIG_GEMM_BF16`. Lookup key (10-tuple):
  `(gfx, cu_num, padded_M, N, K, bias, dtype, otype, scaleAB, bpreshuffle)`.
- **Dedicated CK batched path**: tune via `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` (a8w8:
  `csrc/ck_batched_gemm_a8w8/`); deploy `AITER_CONFIG_BF16_BATCHED_GEMM` (a8w8:
  `AITER_CONFIG_A8W8_BATCHED_GEMM`). Lookup key: `(gfx, cu_num, B, M, N, K)` (legacy CSVs fall back to
  `(cu_num, B, M, N, K)`).
- Prove engagement: `AITER_LOG_TUNED_CONFIG=1` → `is tuned on cu_num` for the dense path.

## Numerics / parity
Same-dtype solution swap → parity-safe per batch; gradlib gates `err_ratio<0.05`. See
[numerics.md](numerics.md).

## Integration (rebind seam)
Live call site `aiter.tuned_gemm:gemm_a16w16` / `tgemm.mm`. No package edit needed — env-overlay CSV.

## Pitfalls & anti-patterns
- bias mismatch (tuned bias=true vs live bias=false) → 100% lookup miss, 0 engagement.
- Expecting it to speed up attention's internal matmuls — those are fused in FMHA, not here.
- TunableOp / `HIPBLASLT_TUNING_FILE` hook a path aiter bypasses → 0 engagement.

## How to verify
`grep -c 'is tuned on cu_num' <server.log>` > 0, then same-session A/B, accept iff delta>0.5% AND
non-overlap AND parity holds.

## Alternatives / cross-links
hipblaslt (executed strided-batched) · ck · triton ·
asm · hip · [overview.md](overview.md).
Dense equivalent: operators/dense_gemm/backends/aiter.md.

## Sources
- On-box: `aiter/tuned_gemm.py` (flatten path), `aiter/ops/batched_gemm_op_bf16.py` +
  `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py` (dedicated CK batched) — `ROCm/aiter@b467ce342`.
- +2.23% dense-path validation (shared flatten engine): perf_knowledge e2e run 2026-06-08.
- aiter engine: https://github.com/ROCm/aiter.
