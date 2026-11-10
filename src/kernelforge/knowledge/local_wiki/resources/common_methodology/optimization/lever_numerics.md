---
title: numerics — fp32 accumulation, online softmax, Welford, the OCP fp8 trap
kind: lever
lever: numerics
gens: [gfx950]
bottleneck: none — this is a correctness gate on every other lever
updated: 2026-08-28
---

# Numerics

**This is not a speed lever. It is the gate every speed lever must pass.** A faster kernel that is
wrong is not a result. Read this before shipping any fast path, and re-read it whenever you change a
dtype.

## Route here when
- You changed a dtype, a scale, or an accumulation order.
- Accuracy regressed, or you see NaN/inf at long context.
- You are about to accept an autotuned config (`lever_autotune.md` gates at `err_ratio < 0.05`).
- You imported a quantized checkpoint from anywhere.

## The four invariants

### 1. Accumulate in FP32 (or INT32) — always
MFMA already accumulates in FP32 in AGPRs even for BF16/FP16/FP8 inputs. **Keep it there.**
Down-converting inside the K-loop buys no speed — the hardware accumulator width is fixed — and costs
accuracy for free. Cast to the output dtype in the epilogue only.

The same rule covers long reductions: softmax denominators, norm sums, logsumexp, `amax`. BF16
accumulation of a long sum loses bits quickly and drifts.

### 2. Online (streaming) softmax for attention
Stream K-blocks holding a running max `m` and denominator `l`; rescale the partial output by
`exp(m_old − m_new)` per block. Avoids `exp(large)` overflow and needs no second pass. This is the
basis of flash-style attention — not an optimization, a correctness requirement at long context.

### 3. Welford for norms
Single-pass mean/variance with a stable running update. The naive `E[x²] − E[x]²` suffers catastrophic
cancellation. FP32 accumulators, always.

### 4. Gate against an FP32 oracle
Every fast path gets compared to an FP32 reference before it counts. `err_ratio < 0.05` is the GEMM
tuning gate. For quantized paths, gate on a **task metric**, not `allclose` — see below.

## The fp8 trap on gfx950: it is OCP, not FNUZ

| | gfx950 (OCP) | Older CDNA (FNUZ) |
|---|---|---|
| E4M3 | **E4M3FN**: bias **7**, max **±448**, ±0, NaN, **no inf** | bias **8**, max **±240**, no inf, single zero, NaN = `0x80` |
| E5M2 | bias 15, max ±57344, **with ±inf** | bias 16, max ±57344, no inf |
| Helper | `__amd_fp8_*` (`hip_ext_ocp.h`) | `__hip_fp8_*` (`hip_fp8.h`) |

**A checkpoint quantized against FNUZ must be converted, never reinterpreted.** Different bias and
different saturation point: bit-copying FNUZ into a gfx950 MFMA produces silently wrong numbers — no
error, no NaN, just drift. Check the producing framework's fp8 flavour before trusting any downloaded
quantized model.

Also gone: **TF32 was removed on gfx950.** Code paths that assumed it must fall back to BF16 or FP32.

## MXFP block scaling (gfx950)

A block of **32 consecutive elements** shares one **E8M0** scale (8-bit, exponent-only,
value `2^(scale−127)`; `127` = no scaling; `E=255` reserved for NaN). The scaled MFMA applies it after
the dot product, before accumulation:

```
v_mfma_scale_f32_32x32x64_f8f6f4(A, B, C, Atype, Btype, opsel_a, scale_a, opsel_b, scale_b)
// type codes: 0=E4M3  1=E5M2  2=E2M3(fp6)  3=E3M2(bf6)  4=E2M1(fp4)
```

The **scale layout is part of correctness** — confirm it with
`amd_matrix_instruction_calculator --architecture cdna4 --detail-instruction` before wiring scales up.
FP6 runs at the FP4 rate, so FP6 is often the better accuracy/speed point than FP4.

**Subnormals are fully supported on gfx950** — no flush-to-zero workarounds needed.

## Quantization hygiene

- **Scale granularity**: per-tensor → per-channel → per-block (MXFP, 32 elements). Pick the finest the
  kernel can afford. A per-tensor scale on a heavy-tailed activation clips.
- Compute `amax` in **FP32**, derive the scale, clamp to the format max before the cast.
- Fuse `amax` + quant into the producing pass to avoid an extra read (`lever_fusion.md`) — but keep the
  `amax` reduction itself in FP32.
- Dynamic activations: recompute the scale per tensor per step. Weights: calibrate offline.

## Verify

| Check | How | Pass |
|---|---|---|
| GEMM fast path | max relative error vs FP32 reference | `err_ratio < 0.05` |
| Quantized path | **task accuracy metric**, not `allclose` | within your task's tolerance |
| Attention long context | logsumexp stability | no NaN/inf at max sequence length |
| fp8 flavour | round-trip a tensor through the cast, check max/rel error against FP32 | bias and saturation match **OCP** |
| Accumulator | ISA dump — no down-convert inside the K-loop | FP32 through to the epilogue |

MMA-Sim (arXiv 2511.10909) is a bit-accurate reference model if you need to predict MFMA
conversion/accumulation behaviour exactly.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Results silently ~wrong, no NaN | FNUZ bits fed to an OCP MFMA | convert, do not bit-copy |
| Drift or NaN at long context | BF16 accumulation of the softmax denom or norm sum | FP32 accumulate |
| Accuracy loss with no speed gain | down-converted the accumulator in-loop | keep FP32 through K |
| Won't lower / compile error on a dtype | assumed TF32 exists | it was removed — BF16 or FP32 |
| Quantized model clips | per-tensor scale on a wide-range tensor | per-channel or MXFP block scale |
| Autotune picked a "faster" wrong kernel | no FP32 oracle gate | enforce `err_ratio < 0.05` |
| Scaled MFMA gives garbage | scale layout mismatch | check with the matrix calculator first |

## Deeper
`hardware/mi350_dtypes.md` (format zoo, OCP detail, FP6/FP4, MXFP + E8M0 block scaling) ·
`lever_autotune.md` (where the `err_ratio` gate is enforced) · `lever_mfma_sched.md`
