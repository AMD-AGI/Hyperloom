---
title: dense_gemm on FlyDSL — SOTA card
kind: sota_card
operator: dense_gemm
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1, mxfp4]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
---

# dense_gemm × FlyDSL

## TL;DR
FlyDSL is aiter's **Python kernel DSL with instruction-level control** — the productivity middle ground
between Triton (too opaque for peak) and raw asm (too slow to iterate). It's the authoring backend AMD used
to beat the stock fused-MoE path on Kimi-K2.5, and aiter's FusedMoE uses FlyDSL for mixed precision (A4W4).
**When FlyDSL is absent, aiter silently falls back to CK** — so verify `is_flydsl_available()`. Deploy is
the same env path as [[operators/dense_gemm/backends/aiter]] (a tuned CSV row with `libtype=flydsl`).

## SOTA implementation
FlyDSL is reached only through the aiter dispatcher, and only when installed. From
`/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`):

```python
if config["libtype"] == "flydsl":
    if is_flydsl_available():
        flydsl_config = aiter.ops.flydsl.gemm_kernels.get_flydsl_splitk_hgemm_kernel_params(
            config["kernelName"])
        if flydsl_config is None:
            config = None          # named kernel not found -> fall through
    else:
        config = None              # FlyDSL not installed -> next granularity / default
```

The executor (`flydsl_gemm`) asserts **no scaling** for the hgemm path and fuses bias only when dtypes match
(`bias.dtype == inp.dtype and otype in {None, inp.dtype}`), else it adds bias in a follow-up cast.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FlyDSL split-K hgemm | `aiter/ops/flydsl/gemm_kernels.py::flydsl_hgemm` (+ `get_flydsl_splitk_hgemm_kernel_params`) | gfx942/950; bf16, fp8, A4W4/mxfp4 (MoE) | Kimi-K2.5 fused-MoE (FlyDSL): up to **+162% throughput, −69% TPOT, −65% TTFT** with SGLang+AITER, vendor-reported 2025 | mixed-precision / MoE GEMM; new shapes needing fast iteration to near-asm perf |

## Config space / knobs
From the on-box `flydsl_hgemm` signature (`aiter/ops/flydsl/gemm_kernels.py`). A tuned CSV row carries
`kernelName`; `get_flydsl_splitk_hgemm_kernel_params` decodes it into these:

| param | range / typical | effect | default |
|---|---|---|---|
| `tile_m` / `tile_n` / `tile_k` | 64–256 / 64–256 / 32–128 | per-workgroup output + K tile | 128 / 128 / 64 |
| `split_k` | 1–16 | K-dim split across CUs (skinny/deep-K) | 1 |
| `block_m_warps` / `block_n_warps` | 1–4 / 1–8 | warp grid inside a block | 1 / 4 |
| `n_tile_repeat` | 1–4 | N tiles per workgroup iteration | 1 |
| `persistent_n_tiles` | 1–N | persistent-kernel N tiling | 1 |
| `waves_per_eu` | 0–4 | occupancy hint (0 = compiler-chosen) | 0 |
| `b_to_lds` / `b_to_lds_unroll` | bool / 0–8 | stage B through LDS + unroll | False / 0 |
| `b_preshuffle` | bool | consume pre-shuffled weights (set by `B.is_shuffled`) | True |
| `c_to_lds` | bool | stage C through LDS before store | False |
| `stages` | 1–4 | software-pipeline depth | FIXED_STAGE (2) |
| `async_copy` | bool | use async global→LDS copies | False |
| `kernel_family` | HGEMM / SMALL_M | choose the small-M kernel for decode | HGEMM |

## Numerics / parity
hgemm with **fp32 accumulate**; bias fused when dtype matches (else cast-then-add). The hgemm path
**rejects scale tensors** (`assert scale_a is None and ...`) — scaled/A4W4 GEMM goes through the MoE/quant
FlyDSL paths, gated on task accuracy ([../numerics.md](../numerics.md)). bf16 hgemm is parity-safe vs library.

## Integration (rebind seam)
Reached through `aiter.tuned_gemm`: a CSV row with `libtype=flydsl` + a `kernelName` that
`get_flydsl_splitk_hgemm_kernel_params` resolves AND `is_flydsl_available()` true. Deploy = same env path as
the dense aiter card (`AITER_CONFIG_GEMM_BF16=<csv>`). No standalone env-overlay for FlyDSL by itself.

## Pitfalls & anti-patterns
- **FlyDSL is optional**; if not installed, aiter silently uses CK (correct, slower for low-bit/MoE) — always
  verify `is_flydsl_available()` before trusting flydsl CSV rows.
- A flydsl row whose `kernelName` isn't decodable returns `None` → the dispatcher falls through to the next
  `padded_M` granularity / default, so a typo in the CSV silently disables the row.
- Instruction-level control = many more knobs than Triton; do **not** hand-tune — rely on aiter's per-shape
  DB / gradlib autotune to fill `kernelName`.
- hgemm path can't take scales — passing `scale_a/scale_b` raises an assert; use the quant FlyDSL/MoE path.

## How to verify (worked example)
```bash
python -c "from aiter.ops.flydsl.utils import is_flydsl_available; print(is_flydsl_available())"
# isolated bench of one tuned kernelName vs hipBLASLt on the same (M,N,K)
python gradlib/gradlib/gemm_tuner.py --indtype bf16 --libtype flydsl -i shapes.csv -o /tmp/fly.csv
# e2e: deploy via aiter env + confirm libtype is flydsl in the log
AITER_CONFIG_GEMM_BF16=/tmp/fly.csv AITER_LOG_TUNED_CONFIG=1 <launch> ; grep 'libtype is flydsl' server.log
```

## Alternatives / cross-links
[[operators/dense_gemm/backends/aiter]] (dispatch + deploy) · [[operators/dense_gemm/backends/triton]]
(easier to author, lower ceiling) · [[operators/dense_gemm/backends/hipblaslt]] (library default) ·
[[operators/dense_gemm/backends/ck]] (the fallback) · [[operators/grouped_gemm_moe/backends/aiter]]
(FlyDSL MoE) · language deep-dive `languages/flydsl/` (P1) · authoring how-to: [[languages/flydsl/authoring_gemm_levers]] (tiling / LDS / MFMA-loop / epilogue when writing a GEMM `@flyc.kernel`) + [[languages/flydsl/authoring_tile_programming]].
