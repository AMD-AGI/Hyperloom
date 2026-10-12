---
title: aiter dispatch & rebind seam — how a kernel actually engages (sglang/vLLM)
kind: language
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:aiter/tuned_gemm.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
---

# aiter dispatch & rebind seam

## TL;DR
aiter is a **dispatcher**: it resolves a per-shape key to a `libtype` and calls the winning executor
(`hipblaslt` / `asm` / `skinny` / `triton` / `torch` / `flydsl` / `opus`). To make a change
"engage" you must hit the **live seam** the framework actually calls — otherwise an optimization is a
silent no-op. This card is the map of those seams and the env gates that turn them on.

## GEMM dispatch (aiter.tuned_gemm)
`get_GEMM_A16W16_config()` looks up the 10-tuple key (leading `gfx`; see [tuning_db.md](tuning_db.md)) and
`solMap` routes the `libtype`:
```python
solMap = {"torch": torch_gemm, "hipblaslt": hipb_gemm, "skinny": skinny_gemm,
          "asm": asm_gemm, "triton": triton_gemm, "flydsl": flydsl_gemm, "opus": opus_gemm}
```
- No matching row → default fallback (`tuned_gemm.py:255-288`): `hipblaslt`/`asm` when `bpreshuffle`,
  `skinny` (solidx 2) for small-M default shapes (`is_skinny_default_shape`), else `torch`. An un-tuned
  shape is **not broken**, just un-optimized. A `flydsl` row is dropped if FlyDSL isn't installed (falls
  through to the next granularity/default).
- `opus` is a split-K GEMM/MoE-stage2 backend (`gfx942`/`gfx950`/`gfx1250`); its launcher needs a
  per-stream fp32 workspace warmed **before** HIP-graph capture (see `aiter/ops/opus/*`,
  `csrc/opus_gemm/*` — this is aiter-internal, no separate card).
- Live call sites: `aiter.tuned_gemm:gemm_a16w16`, `tgemm.mm` — sglang/vLLM `LinearMethod` route here.

## SGLang seam
- Master gate: **`SGLANG_USE_AITER=1`** — without it, `UnquantizedLinearMethod` runs `F.linear`/hipBLASLt
  default and aiter is never consulted.
- Attention/MoE/MLA have their own `SGLANG_*` flags (e.g. `SGLANG_ROCM_FUSED_DECODE_MLA`,
  `SGLANG_AITER_MLA_PERSIST`).
- To engage an authored kernel: add a tuned CSV row (`libtype`) or a call-site rebind, then **e2e-gate**.

## vLLM seam (custom-op registration)
`vllm/_aiter_ops.py` wraps aiter kernels as **`torch.ops` custom ops** via `direct_register_custom_op`
(with fake/meta impls). This is what keeps hand-tuned aiter kernels **opaque through `torch.compile`** —
Inductor fuses *around* them instead of decomposing them into generated Triton.
- Master: **`VLLM_ROCM_USE_AITER=1`** (default 0); sub-flags `_LINEAR/_MOE/_RMSNORM/_MLA/_MHA` (default 1
  once master on), `_FP4BMM=0` on gfx942 (crash), `_TRITON_GEMM`, `_TRITON_ROPE`, …
- Registered ops examples: `rocm_aiter_ck_moe`, `rocm_aiter_fmoe_fp8_blockscale_g1u1`,
  `rocm_aiter_asm_moe`, `rocm_aiter_topk_softmax`, `_rocm_aiter_mla_decode_fwd`, `_rocm_aiter_w8a8_gemm`.
- **Register as a custom op → survives `torch.compile`; don't → Inductor regenerates it** (losing the
  hand-tuned kernel). ROCm fusion passes (`rocm_aiter_fusion.py`) fuse aiter op chains (rms+quant).

## The rebind decision (Amdahl gate)
An authored/replacement kernel only helps if it reaches the live seam AND moves e2e:
1. Pick the seam (aiter CSV `libtype` row, or a `LinearMethod`/custom-op rebind).
2. Prove **engagement** (`AITER_LOG_TUNED_CONFIG=1` → `is tuned on cu_num`; or rocprofv3 shows the kernel
   ran, not a Triton fallback).
3. **e2e-gate**: keep only if `pct_gpu_time × speedup` clears the noise band. An isolated 1.47× that never
   engages, or engages but is Amdahl-tiny, is a reject.

## Pitfalls
- **Isolated win ≠ e2e win**: an authored kernel measured 0.99–1.47× isolated still lost e2e to the aiter
  env path (didn't enter the stack). Always gate through the real seam.
- **Image/ABI mismatch**: `VLLM_ROCM_USE_AITER=1` / `SGLANG_USE_AITER=1` with no matching aiter in the
  image → import/runtime failure or silent wrong results. Don't ad-hoc pip-install aiter.
- **Coverage gaps**: aiter tunes CDNA4 first; a missing gfx942 shape falls back to generic Triton (several×
  slower) — watch traces; `AITER_ONLINE_TUNE=1` to retry on `wrong! device_gemm`.
- **Env sprawl**: 13+ `VLLM_ROCM_USE_AITER_*` vars; a config-based op-priority system is proposed (vLLM
  #33163) — expect the surface to change.

## Sources
- GEMM dispatch + solMap + fallback: `ROCm/aiter@b467ce342:aiter/tuned_gemm.py`.
- vLLM custom-op registration / torch.compile opacity: https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py ; https://docs.vllm.ai/en/stable/design/custom_op/
- vLLM aiter env gates: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
