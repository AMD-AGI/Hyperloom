---
title: scaled_quant_gemm on asm — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: asm
gens: [gfx950, gfx942]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp6_e2m3]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - ROCm/aiter@a6bb4993:aiter/ops/gemm_op_a4w4.py
---

# scaled_quant_gemm × asm

## TL;DR
> The near-peak low-bit GEMM path: hand-tuned MFMA assembly (or Gluon, which lowers to it) that overlaps the
> **scale pipeline** with the **tile pipeline** perfectly. Gluon reached **BF8 99.72%** and **MXFP4 92.41%**
> efficiency on gfx950 — this is where you go for the last 10–20% over Triton/CK. Author via Gluon/aiter, not
> raw asm from scratch. In aiter it's a raced/blockscale candidate (`gemm_a4w4_asm`, fp8 blockscale asm).

## SOTA implementation
aiter's A4W4 dispatcher falls to the **asm** kernel when the CK-blockscale lookup doesn't claim the shape
(mangled-name check). From `/sgl-workspace/aiter/aiter/ops/gemm_op_a4w4.py` (`ROCm/aiter@a6bb4993`):

```python
gemm_a4w4_asm(
    A.view(m, k // 2), B, A_scale, B_scale, out, kernelName,
    bias, alpha, beta, bpreshuffle, log2_k_split=splitK)
```

`A:[M,K/2] f4x2`, `B:[N,K/2] f4x2`, **E8M0 per-32 block scales** `A_scale:[M,K/32]`, `B_scale:[N,K/32]`,
output M padded to a multiple of 32, optional `log2_k_split`. The Gluon tutorial kernel is the open
reference for how the scale GR→LDS→MFMA staging is overlapped.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Gluon scaled-MFMA GEMM (lowers to asm) | https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html | gfx950; bf8, mxfp4 | **BF8 3257 TFLOPS (99.72% eff), MXFP4 5255 TFLOPS (92.41% eff)** @ MI350/MI355 gfx950 | absolute peak low-bit GEMM |
| HIP/C++ **8-wave ping-pong** FP8 (no asm) | AMD CDNA4 GEMM blog (cdna4-gemm-kernels) | gfx950 fp8 | **2680 TFLOPS @ 4096, 3204 @ 8192** — *beats* hipBLASLt 3130 (MI355X, ROCm 7.1) | near-peak FP8 without writing asm |
| HIP/C++ **4-wave interleave** FP8 (successor) | AMD 4-wave FP8 GEMM blog (4wave-fp8gemm) | gfx950 fp8 | one wave/SIMD, full 512-VGPR budget, 128×128 tile; no `#pragma unroll`, robust across ROCm (HK 4-wave 3327 TFLOPS) | robustness/perf successor to ping-pong |
| aiter asm scaled GEMM (`gemm_a4w4_asm`, fp8 blockscale asm) | `aiter/ops/gemm_op_a4w4.py`, `gemm_op_a8w8.py` | gfx950 fp4/fp8; gfx942 fp8 FNUZ | selected per shape by aiter DB | live serving small-M / blockscale fp4/fp8 |

The **8-wave ping-pong** and **4-wave interleave** scheduling patterns both originate from **HipKittens**
(arXiv 2511.08083) and were adopted into AMD's own CDNA4 GEMM blogs. NVIDIA-style wave specialization
underperforms on CDNA (static register allocation starves producers → ~80% peak BF16) — use ping-pong/
interleave. See optimization/mfma_scheduling.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| scaled-MFMA instr | `mfma_scale_f32_16x16x128_f8f6f4` / `32x32x64` | matrix-core shape for f8f6f4 | per-kernel |
| `Atype/Btype` codes | 0=E4M3,1=E5M2,2=E2M3,3=E3M2,4=E2M1 | operand format | — |
| scale staging | Global→LDS re-layout→LDS read | overlap with A/B tile pipeline (the hard part) | — |
| operand packing | 256-bit | fp4 32-elem = 128b → pad to 256b operand | — |
| K alignment | multiple of 32 | scale block granularity | — |
| `log2_k_split` | 0..n | K split across CUs | 0 |
| `bpreshuffle` | bool | pre-shuffled weights | True |

## Numerics / parity
- **E8M0** per-32 microscaling, **scale-after-dot**, fp32 accumulate; accuracy gate vs bf16
  ([numerics.md](numerics.md)). gfx942 fp8 is FNUZ (no native block scale); gfx950 is OCP MXFP.

## Integration (rebind seam)
Via aiter dispatch (`gemm_a4w4` falls to `gemm_a4w4_asm`; fp8 blockscale asm picked by small `padded_M` /
scaled key) or a Gluon-compiled op called directly. Verify the asm scaled kernel name in a rocprof trace.

## Pitfalls & anti-patterns
- asm is **gfx-specific**; the scaled-MFMA instrs are **gfx950-only** — gfx942 has no native block-scaled MFMA
  (and aiter A4W4 raises on gfx942).
- The **scale pipeline is the hard part** — a stall in the Global→LDS→MFMA staging caps you well below the
  Gluon ceiling; this is why Triton/CK trail asm here.
- Wrong E8M0 scale shape (`[*, K/32]`) or K not multiple of 32 → silent misindex.
- Forgetting 256-bit operand padding for fp4 scaled MFMA.

## How to verify (worked example)
```bash
# microbench vs the Gluon ceiling on the same shape
rocprofv3 --stats -- python bench_a4w4_asm.py     # confirm scaled asm kernel name
# TFLOP/s = 2*M*N*K / t  -> compare to 5255 (MXFP4) / 3257 (BF8) ceilings
# accuracy: dequant reference + downstream eval (lm-eval / LAMBADA)
```

## Alternatives / cross-links
operators/scaled_quant_gemm/backends/triton (parity ref) · operators/scaled_quant_gemm/backends/aiter
(live dispatch) · operators/dense_gemm/backends/asm (bf16 asm + scaled-MFMA knobs) ·
quantization/block_scaling_mxfp · quantization/fnuz_vs_ocp ·
operators/scaled_quant_gemm/overview

## Sources
- Gluon GEMM tutorial (asm-level, BF8 99.72% / MXFP4 92.41%): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- CDNA4 ISA (scaled MFMA): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- aiter asm A4W4 dispatch (E8M0 scales, M-pad-32): `/sgl-workspace/aiter/aiter/ops/gemm_op_a4w4.py` (`ROCm/aiter@a6bb4993`).
- HIP/C++ 8-wave ping-pong FP8 2680@4096 / 3204@8192 (>hipBLASLt 3130, no asm): AMD CDNA4 GEMM blog (https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html).
- 4-wave interleave successor (1 wave/SIMD, full 512 VGPR, 128×128, no `#pragma unroll`); HK 4-wave 3327 TFLOPS: AMD 4-wave blog (https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html) + arXiv 2511.08083.
