---
title: scaled_quant_gemm on aiter — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: aiter
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:aiter/ops/gemm_op_a8w8.py
  - ROCm/aiter@b467ce342:aiter/ops/gemm_op_a4w4.py
  - ROCm/aiter@b467ce342:aiter/jit/core.py
  - ROCm/aiter@b467ce342:csrc/ck_gemm_a8w8/gemm_a8w8_tune.py
  - https://github.com/ROCm/aiter
---

# scaled_quant_gemm × aiter

## TL;DR
> On sglang/vllm, **aiter is the live scaled-GEMM path** — fp8 `gemm_a8w8` / block-scale variants, and A4W4
> (fp4) via FlyDSL→CK/asm. The per-shape tuned DB selects the fastest scaled impl. To improve fp8/fp4 GEMM on
> the serving path you tune aiter's DB with the **`scaleAB` key set** — same capture/tune/deploy mechanism as
> dense GEMM, but a separate CSV per quant format (`AITER_CONFIG_GEMM_A8W8`, `_A4W4`, `_A8W8_BLOCKSCALE`, …).

## SOTA implementation
For 16-bit-in/quant-out shapes the dispatcher is the same `gemm_a16w16` 10-tuple (leading `gfx`, with
`scaleAB=True` when `scale_a/scale_b` is passed). Dedicated quant entry points have their own CSV lookups.
From `aiter/ops/gemm_op_a4w4.py` (`ROCm/aiter@b467ce342`) — A4W4 routes CK-blockscale vs asm
by `kernelName`, and **rejects gfx942**:

```python
if gfx_arch in ["gfx942"]:
    raise RuntimeError("A4W4 GEMM kernel is not supported on gfx942 ...")
ck_config = get_GEMM_config(m, n, k)            # padded_m lookup, own CSV
if ck_config is not None and kernelName.find("_ZN") == -1:
    return gemm_a4w4_blockscale(A.view(m, k//2), B, A_scale, B_scale, out, splitK=splitK)[:m]
gemm_a4w4_asm(A.view(m, k//2), B, A_scale, B_scale, out, kernelName, bias, alpha, beta, bpreshuffle, ...)
```

A4W4 packs 4-bit operands (`A:[M,K/2] f4x2`), uses **E8M0 per-32 block scales** (`A_scale:[M,K/32]`), and
pads output M to a multiple of 32 (`(m+31)//32*32`).

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter scaled GEMM (`gemm_a8w8` / fp8 block-scale, A4W4 via FlyDSL→CK/asm) | `aiter/ops/gemm_op_a8w8.py`, `gemm_op_a4w4.py` (+ `tuned_gemm` scaled key) | gfx942 fp8 FNUZ; gfx950 fp8/mxfp4 | no first-party isolated number reproduced — selected per shape from tuned DB; fp8 dense ceiling ~1223.6 TFLOPS (hipBLASLt, MI300X, Feb-2025) | live fp8/fp4 serving GEMM |

**CDNA4 (gfx950) ceiling context:** aiter is the live lever, but the absolute FP8 bar on MI355X is set
elsewhere — hipBLASLt FP8 ~3130 TFLOPS @ 8192, *beaten* by a HIP/C++ **8-wave ping-pong** kernel at
**3204** (no asm); the **4-wave interleave** variant is the successor. Gluon reaches **BF8 3257 TFLOPS
@ 99.72%** and **MXFP4 5255 @ 92.41%**. Both scheduling patterns originate from **HipKittens** (arXiv
2511.08083). See operators/scaled_quant_gemm/tuning, operators/scaled_quant_gemm/backends/asm,
optimization/mfma_scheduling, and the Gluon/HipKittens cards.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| 10-tuple `scaleAB` | True for scaled | must be **True** to hit scaled rows; mismatch = miss | from call |
| `bpreshuffle` | bool | pre-shuffled weights (own `_BPRESHUFFLE` CSV) | False |
| quant format | fp8 tensor / fp8 block / A4W4 | picks which CSV + kernel family | per-call |
| `AITER_CONFIG_GEMM_A8W8` | path(s) | deploy tuned fp8 per-tensor CSV | `configs/a8w8_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` | path(s) | fp8 block-scale CSV | `configs/a8w8_blockscale_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` | path(s) | preshuffled fp8 CSV | `configs/a8w8_bpreshuffle_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` | path(s) | fp8 blockscale + preshuffle CSV | `configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv` |
| `AITER_CONFIG_GEMM_A4W4` | path(s) | fp4 block-scale CSV | `configs/a4w4_blockscale_tuned_gemm.csv` |
| `AITER_LOG_TUNED_CONFIG` | 0/1 | engagement log | 0 |

Capture/tune/deploy follow the dense recipe (operators/dense_gemm/backends/aiter) but each quant format has
its **own CK tuner + untuned CSV**: `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py` (fp8/int8 per-tensor),
`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`, `csrc/ck_gemm_a8w8_bpreshuffle/...`, and
`csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` (fp4). A4W4 needs FlyDSL installed for the fast
path (else CK blockscale).

## Numerics / parity
- fp8 **FNUZ** (gfx942) vs **OCP** (gfx950); **scale-after-dot**, fp32 accumulate. A4W4 uses E8M0 per-32
  microscaling (OCP MXFP4).
- gradlib gates each solution on `err_ratio < 0.05`; add a **task-accuracy gate** (the byte-parity gate does
  not apply to quant) → [numerics.md](numerics.md).

## Integration (rebind seam)
Live call sites: `gemm_a8w8` / `gemm_a8w8_blockscale*` / `gemm_a4w4` (sglang/vllm fp8/fp4 `LinearMethod`).
Deploy by the matching `AITER_CONFIG_GEMM_*` env (no package edit). Verify via `grep 'is tuned on cu_num'`
and the scaled kernel name in a rocprof trace.

## Pitfalls & anti-patterns
- **scaleAB / bias / gfx mismatch = 0 engagement**: capture live (don't synthesize) so the 10-tuple key matches.
- gfx942 has **no native block-scaled MFMA** → fp8 here is tensor/coarse-scaled, not 32-elem MXFP; and
  **A4W4 raises a RuntimeError on gfx942** — fp4 is gfx950-only.
- A4W4 expects E8M0 scales shaped `[*, K/32]` and output M padded to 32 — wrong scale shape misindexes
  silently.
- Each quant format has its **own CSV env** — pointing only `AITER_CONFIG_GEMM_BF16` does nothing for fp8/fp4.

## How to verify (worked example)
```bash
EXTRA_ENV="AITER_TUNE_GEMM=1" <launch fp8 server>                       # capture scaled shapes
python csrc/ck_gemm_a8w8/gemm_a8w8_tune.py --mp 8 \
    -i aiter/configs/a8w8_untuned_gemm.csv -o /tmp/fp8_tuned.csv --errRatio 0.05
EXTRA_ENV="AITER_CONFIG_GEMM_A8W8=/tmp/fp8_tuned.csv AITER_LOG_TUNED_CONFIG=1" <launch>
grep -c 'is tuned on cu_num' server.log        # > 0
# A/B (ref vs +CSV): accept on non-overlapping latency win AND task-accuracy gate held
```

## Alternatives / cross-links
operators/scaled_quant_gemm/backends/triton · operators/scaled_quant_gemm/backends/asm ·
operators/dense_gemm/backends/aiter (same mechanism, bf16) ·
operators/grouped_gemm_moe/backends/aiter (A4W4 MoE) · quantization/block_scaling_mxfp ·
quantization/fnuz_vs_ocp · operators/scaled_quant_gemm/overview

## Sources
- On-box: `aiter/ops/gemm_op_a8w8.py`, `gemm_op_a4w4.py`, `tuned_gemm.py`, `aiter/jit/core.py` (config env
  names), and the per-format CK tuners `csrc/ck_gemm_a8w8*/`, `csrc/ck_gemm_a4w4_blockscale/` — `ROCm/aiter@b467ce342`.
- AITER scaled GEMM: https://github.com/ROCm/aiter
- Matrix Core (fp8 FNUZ vs OCP, block scaling): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- fp8 1223.6 TFLOPS ceiling (MI300X, ROCm 6.3): https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
- CDNA4 FP8 bars — 8-wave ping-pong 3204@8192 (>hipBLASLt 3130): AMD cdna4-gemm-kernels blog; 4-wave interleave: 4wave-fp8gemm blog; Gluon BF8 3257@99.72% / MXFP4 5255@92.41%: Gluon GEMM tutorial; scheduling origin: arXiv 2511.08083.
