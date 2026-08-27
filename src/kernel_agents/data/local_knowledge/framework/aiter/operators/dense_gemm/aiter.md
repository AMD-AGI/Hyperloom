---
title: dense_gemm on aiter — SOTA card
kind: sota_card
operator: dense_gemm
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:csrc/gemm_a16w16/gemm_a16w16_tune.py
  - ROCm/aiter@b467ce342:gradlib/gradlib/GemmTuner.py
  - ROCm/aiter@b467ce342:aiter/jit/core.py
  - ROCm/aiter@b467ce342:aiter/ops/gemm_op_common.py
  - https://github.com/ROCm/aiter
---

# dense_gemm × aiter

## TL;DR
On sglang/vllm, **aiter is the live dense-GEMM path** (`aiter.tuned_gemm:gemm_a16w16` → dispatches per
shape to hipBLASLt `Cijk_*` / asm / skinny / triton / flydsl / opus from a tuned CSV DB). To improve dense
GEMM you tune **aiter's per-shape DB**: capture real shapes (`AITER_TUNE_GEMM=1`), tune with the
multi-backend tuner `csrc/gemm_a16w16/gemm_a16w16_tune.py` (hipBLASLt opt-in via `--with-hipblaslt`), deploy
by env (`AITER_CONFIG_GEMM_BF16`). This is the **only** GEMM lever that actually engages the serving path —
TunableOp and `HIPBLASLT_TUNING_FILE` hook PyTorch dispatch, which aiter bypasses entirely. Measured
**+2.23% e2e on Qwen3.5-27B / sglang** from DB tuning alone.

## SOTA implementation
The live dispatcher resolves a **10-tuple** key (leading `gfx`) against a CSV and calls the winning library.
Real code, on-box (`aiter/tuned_gemm.py`, `ROCm/aiter@b467ce342`):

```python
# get_GEMM_A16W16_config(): try exact M, then two padded_M granularities; key is gfx-first
gfx, cu_num = get_gfx(), get_cu_num()
for gl in [None, 0, 1]:
    padded_M = M if gl is None else get_padded_m(M, N, K, gl)
    config = cfg.get(
        (gfx, cu_num, padded_M, N, K, bias, str(dtype), str(otype), scaleAB, bpreshuffle),
        None,
    )
    if config is not None:
        if config["libtype"] == "flydsl":          # flydsl needs the pkg AND a known kernel name
            if not is_flydsl_available() or get_flydsl_splitk_hgemm_kernel_params(...) is None:
                config = None; continue
        if AITER_LOG_TUNED_CONFIG:
            logger.info(f"... is tuned on cu_num = {cu_num} ... libtype is {config['libtype']}")
        return config
```

`solMap` then routes `libtype` to the executor: `{"torch", "hipblaslt", "skinny", "asm", "triton",
"flydsl", "opus"}`. So this card subsumes "which library kernel" — the multi-backend tuner
(`csrc/gemm_a16w16/gemm_a16w16_tune.py`) races them per shape and writes the winner.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `tuned_gemm` DB (per-shape libtype+solidx) | `aiter/tuned_gemm.py` + `csrc/gemm_a16w16/gemm_a16w16_tune.py` (multi-backend) / `gradlib` (hipBLASLt) | gfx942/950, bf16/fp16 (fp8 via scaled variants) | **+2.23% e2e** stacked on `--attention-backend triton` (1548.9 → 1583.5 tok/s, 5-rep non-overlapping, **246 `is tuned on cu_num` hits**) @ MI300X gfx942, sglang 0.5.11 / aiter, 2026-06-08 | the live serving GEMM path on sglang/vllm |
| hipBLASLt solution dispatched by aiter | `hipb_gemm` → `hipb_mm` | gfx908–950; bf16/fp16/fp8 | ~708 TFLOPS bf16 (M4096,N4864,K32896) ≈54% of 1307 theo. peak @ MI300X, ROCm 6.3, AMD Labs Feb-2025 | most prefill/large-M shapes |
| skinny `wvSpltK` / `wv_splitk_small` | `skinny_gemm` → `ops.wvSpltK` | gfx942/950; bf16/fp16 | bandwidth-bound decode win, see skinny card | decode M=1..8 (`is_skinny_default_shape`) |

## Config space / knobs
The flow is **capture → tune → deploy**; each stage is an env/CLI lever:

| param | range / values | effect | default |
|---|---|---|---|
| `AITER_TUNE_GEMM` | 0 / 1 | when 1, every live `gemm_a16w16` call appends its real shape to `aiter/configs/bf16_untuned_gemm.csv` | 0 |
| `gemm_a16w16_tune.py --indtype` | f32/f16/bf16/fp8 | input dtype for the tuning run | (from CSV) |
| `gemm_a16w16_tune.py --mp` | 1..#GPUs | parallel tuning processes (all visible GPUs) | 1 |
| `--errRatio` | float | per-solution accuracy gate; solutions above are dropped before argmin-time selection | **0.05** |
| `--libtype` | all/asm/opus/flydsl/triton/torch/skinny/hipblaslt | restrict which backends to race (default `all`; hipblaslt needs `--with-hipblaslt`) | all |
| `--with-hipblaslt` | flag | also race hipBLASLt (delegates to gradlib; off by default) | off |
| `--all_bias` | flag | tune both bias=true/false regardless of capture | off |
| `AITER_CONFIG_GEMM_BF16` | path(s), `:`-joined | deploy the tuned CSV (no package edit); multiple files merge via `AITER_CONFIGS` | `aiter/configs/bf16_tuned_gemm.csv` |
| `AITER_LOG_TUNED_CONFIG` | 0 / 1 | emit `is tuned on cu_num` log line per hit — the engagement proof | 0 |

**`get_padded_m` bucketing** bounds tune time and improves hit rate. It is exposed in Python from
`aiter/ops/gemm_op_common.py` (compiled op `module_gemm_common`/`getPaddedM`; C++ impl in
`csrc/py_itfs_cu/gemm_common.cu`): `gl=0` (fine) rounds M up to 16 (M≤256) / 32 (≤1024) / 64 (≤4096) / 128
else; `gl=1` (coarse) is `nextPow2(M)` (capped at 8192 for M>8192,N>4096). The lookup tries exact M, then
`gl=0`, then `gl=1`, so a tuned bucket covers a *range* of live M.

## Numerics / parity
Same-math bf16/fp16 algorithm swap → parity-safe. gradlib gates each solution at tune time with
`rtol=atol=5e-2` (bf16) / `1e-2` (else) and keeps only `err_ratio < errRatio` (0.05). Cross-solution
accumulation-order flips are possible but tiny; add a downstream task-accuracy gate for fp8 scaled variants
([numerics.md](numerics.md)).

## Integration (rebind seam)
Live call site: `aiter.tuned_gemm:gemm_a16w16` and `tgemm.mm` (sglang/vllm `LinearMethod` route here).
The lookup key is `(gfx, cu_num, padded_M, N, K, bias, dtype, otype, scaleAB, bpreshuffle)` — **every field
must match the live calls**. `gfx` is `get_gfx()`, `cu_num` is `get_cu_num()`, `bias` is `bias is not None`,
`scaleAB` is `scale_a/scale_b is not None`, `bpreshuffle` is `B.is_shuffled`. Deploy = point
`AITER_CONFIG_GEMM_BF16` at your CSV; no code change.

## Pitfalls & anti-patterns
- ⚠ **bias mismatch = 0 engagement**: tuning with synthesized `bias=true` shapes while live calls are
  `bias=false` → every lookup misses → tuned CSV does nothing (verified failure, 2026-06-07). Always capture
  live (`AITER_TUNE_GEMM=1`), never hand-author the untuned CSV.
- **TunableOp / `HIPBLASLT_TUNING_FILE` are dead ends on sglang**: they hook PyTorch's `addmm` dispatch;
  aiter calls `hipb_mm` directly, so neither is consulted on the live path → 0 engagement.
- **Fork-storm risk**: the hipBLASLt path (`--with-hipblaslt`) races ~1365 solutions/shape and across large
  prefill shapes spawns hundreds of `rocm_agent_enumerator` subprocesses and can OOM the host. Bucket-reduce
  big M (rely on `get_padded_m`), cap `--mp`, and restrict `--libtype` while iterating.
- **`flydsl` rows silently drop** if FlyDSL isn't installed (`is_flydsl_available()` false) or the tuned
  kernel name is not in the current catalog — the loop sets `config=None` and falls through to the next
  granularity/default. Verify FlyDSL before trusting flydsl rows.
- Default fallback when no row matches (`tuned_gemm.py`): `hipblaslt`/`asm` when `bpreshuffle`, `skinny`
  (solidx 2) for small-M default shapes, else `torch` — i.e. an un-tuned shape is *not* broken, just
  un-optimized.

## How to verify (worked example)
```bash
# 1) capture live shapes (warm the server, run real traffic)
EXTRA_ENV="AITER_TUNE_GEMM=1" <launch sglang server>     # appends to bf16_untuned_gemm.csv

# 2) tune across all GPUs, accuracy-gated (multi-backend; add --with-hipblaslt to also race hipBLASLt)
python csrc/gemm_a16w16/gemm_a16w16_tune.py --indtype bf16 --mp 8 \
    -i aiter/configs/bf16_untuned_gemm.csv -o /tmp/qwen_tuned.csv --errRatio 0.05 --libtype all

# 3) deploy + prove engagement
EXTRA_ENV="AITER_CONFIG_GEMM_BF16=/tmp/qwen_tuned.csv AITER_LOG_TUNED_CONFIG=1" <launch server>
grep -c 'is tuned on cu_num' server.log        # must be > 0  (we saw 246)

# 4) same-session 2-launch A/B (ref=current vs cand=+CSV); accept iff
#    delta > 0.5% AND cand_min > ref_max AND parity holds  ->  1548.9 -> 1583.5 tok/s (+2.23%)
```

## CDNA4 ceiling (context)
aiter is the live serving lever, but on **MI355X (gfx950)** the absolute dense-GEMM ceiling is now set by
DSL / HIP-C++ kernels, not aiter's library dispatch: **Gluon FP16 1489 TFLOPS @ 98.75% MFMA eff** and
**HipKittens BF16 1610 TFLOPS** (M=N=K=8192) are the current bars (AMD/HK-measured). For FP8 the HIP/C++
8-wave ping-pong kernel hits 3204 @ 8192, *beating* hipBLASLt with no asm. These are reference ceilings —
the live aiter path still resolves to hipBLASLt/asm/skinny per shape. See
operators/dense_gemm/backends/gluon, operators/dense_gemm/backends/hipkittens.

## Alternatives / cross-links
operators/dense_gemm/backends/hipblaslt (executed kernels) · operators/dense_gemm/backends/flydsl
(authorable, mixed precision) · operators/dense_gemm/backends/gluon (FP16 1489 ceiling) ·
operators/dense_gemm/backends/hipkittens (BF16 1610, academic SOTA) ·
operators/dense_gemm/backends/triton (fused/odd shapes) ·
operators/dense_gemm/backends/asm (peak) · operators/dense_gemm/backends/ck ·
operators/dense_gemm/backends/opus (split-K GEMM) ·
operators/skinny_gemv_decode/backends/aiter (decode M=1..8) ·
operators/scaled_quant_gemm/backends/aiter (fp8/fp4) · kernel_workflow/gemm_tuning_workflow ·
operators/dense_gemm/overview.

## Sources
- On-box source: `aiter/tuned_gemm.py`, `csrc/gemm_a16w16/gemm_a16w16_tune.py` (multi-backend tuner),
  `gradlib/gradlib/GemmTuner.py` (hipBLASLt-only), `aiter/jit/core.py` (`AITER_CONFIGS`),
  `aiter/ops/gemm_op_common.py` + `csrc/py_itfs_cu/gemm_common.cu` (`get_padded_m`) (= `ROCm/aiter@b467ce342`).
- Measured +2.23% / 246 engagement hits: perf_knowledge e2e validation run `e2e_Qwen-Qwen3.5-27B_20260607_193315`, 2026-06-08.
- hipBLASLt bf16 ~708 TFLOPS / 1307 theo. peak (MI300X, ROCm 6.3, Feb-2025): https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html ; ~45% sustained utilization: https://arxiv.org/pdf/2510.27583
- aiter as central engine: https://github.com/ROCm/aiter
- CDNA4 ceilings — Gluon FP16 1489@98.75%: AMD Gluon GEMM tutorial; HipKittens BF16 1610: arXiv 2511.08083; HIP/C++ 8-wave ping-pong FP8 3204@8192 (>hipBLASLt): AMD cdna4-gemm-kernels blog.
