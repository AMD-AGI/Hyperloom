---
title: CDNA4 / MI350 (gfx950) — FP4 / FP6 / MXFP microscaling
kind: hardware
gens: [gfx950]
dtypes: [fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4]
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350 (gfx950) — FP4 / FP6 / MXFP microscaling

> The new low-bit numerics on CDNA4. Matrix-core instruction detail in
> [matrix_core_blockscale.md](matrix_core_blockscale.md); cross-gen format facts in
> [../shared/dtype_numerics.md](../shared/dtype_numerics.md). Numbers in brackets are the sources at the end.

## TL;DR
> CDNA4 adds **FP6 (E2M3/E3M2)** and **FP4 (E2M1)**, and **MXFP** block scaling where a **32-element
> block** shares one **E8M0** (8-bit, exponent-only) scale [1][3]. MXFP6 and MXFP4 both run at **10 PF** on
> MI355X [4]. Per AMD, the finer-grained scale lets reduced-precision formats serve a much wider variety of
> tensors than a single per-tensor scale [4].

## Concepts

### The low-bit formats
| Format | Bits | Layout | Max (per element) | Use |
|---|---|---|---|---|
| FP8 E4M3 | 8 | 4-exp/3-mant, bias 7 | ±448 (OCP) [1] | activations/weights |
| FP8 E5M2 | 8 | 5/2, bias 15 | ±57344 [1] | wide range |
| **FP6 E2M3** | 6 | 2/3, bias 1 | ±7.5 [1] | more mantissa |
| **FP6 E3M2** | 6 | 3/2, bias 3 | ±28 [1] | wider range |
| **FP4 E2M1** | 4 | 2/1, bias 1 | ±6 [1] | most compact |

FP4 packs **2 values per byte** (`__amd_fp4x2_storage_t`, alias `uint8_t`); helpers
`__amd_extract_fp4`, `__amd_create_fp4x2` from `hip_ext_ocp.h` [2][6]. Addressing granularity is 8 bits [2].

### MXFP = microscaling (OCP MX spec)
- A **block of 32 consecutive elements** along K shares **one E8M0 scale** [1][3].
- **E8M0**: 8 bits, exponent only, value `2^(scale-127)`; `scale=127` ⇒ ×1 (no scaling); `E=255`
  reserved for NaN; range `2^-127 … 2^127` [2][3].
- `MXFP8` = 32× FP8 + 1 E8M0; `MXFP6` = 32× FP6 + 1 E8M0; `MXFP4` = 32× FP4 + 1 E8M0 [3].
- Effective bits/element = element_bits + 8/32 = element_bits + 0.25 (the scale amortizes over 32; derived).

### Why block scaling matters
A per-tensor scale has to cover the whole tensor's range; a scale shared by a block of 32 elements follows
the local range. AMD's white paper: the finer granularity lets reduced-precision formats be used on a much
wider variety of tensors in AI workloads, and opens the way to MXFP6 and MXFP4 [4].

### How the hardware applies the scale
The block-scaled MFMA (`v_mfma_scale_f32_*_f8f6f4`) takes the A/B operands **plus** their E8M0 scale
operands (Ax/Bx) and applies the scale **after the dot product, before accumulation** [2]. Type codes:
`0=E4M3, 1=E5M2, 2=E2M3, 3=E3M2, 4=E2M1` [1][2]. A and B types/scales are independent → mix FP4 weights
with FP6/FP8 activations [1]. Layout (32×32×64): A 32×64, **Ax 32×2**, B 64×32, **Bx 2×32**, C 32×32;
per-thread 32 A / 1 Ax / 32 B / 1 Bx / 16 C [2].

### Throughput (MI355X)
| Precision | CDNA4 peak [4][5] | vs FP32 matrix (derived) |
|---|---|---|
| FP16/BF16 | 2.5 PF | 16× |
| FP8 (OCP) | 5 PF | 32× |
| MXFP8 | 5 PF | 32× |
| MXFP6 | **10 PF** | 64× |
| MXFP4 | **10 PF** | 64× |

MXFP6 runs at the MXFP4 rate, not between FP8 and FP4 [4]: in the matrix core, choosing FP6 over FP4
costs **accuracy headroom, not FLOPs**. FP6 still moves 1.5× the bytes of FP4, which matters wherever the
kernel is bandwidth-bound (derived).

## The levers
1. **MXFP4 weight-only** for the largest weight tensors; **MXFP6** when FP4 is too lossy (same matrix rate).
2. **Mix A/B types** — FP4 weights × FP6/FP8 activations as the accuracy gate dictates.
3. **Block-scale (32-elem E8M0)** rather than per-tensor scale on wide-range tensors.
4. **Always gate on task accuracy**, not byte parity, for any FP4/FP6 path.

## Pitfalls
- **Per-tensor FP4** — the whole tensor shares one scale; use MXFP block scaling.
- **Wrong Ax/Bx scale layout** → silent corruption; follow the ISA's scale layout [1].
- **Choosing FP4 over FP6 "for speed"** — same 10 PF matrix rate; FP6 gives more mantissa or range.
- **Assuming MXFP on CDNA3** — CDNA4-only [7].

## Verify
- Round-trip weights through the MXFP4/6 cast; measure per-block error and end-task accuracy vs FP16.
- Scale placement: the scale data layouts for 16×16 (K=128) and 32×32 (K=64) outputs in the CDNA4 ISA
  §7.2.1 [1], and the worked FP8 and FP4 layouts in [2]. (AMD's Matrix Instruction Calculator covers
  CDNA1–CDNA3, not CDNA4 [8].)

## Sources
1. AMD, *CDNA4 Instruction Set Architecture Reference Guide*, 5 Aug 2025: §6.7.1 (scaled conversions: an
   E8M0 bias of 127 shared by 32 values along K), §7.1.4 (F8F6F4 rules: CBSZ/BLGP type codes),
   §7.2.1 (MFMA with block exponent scaling, scale layouts), §7.3 Table 30 (small float formats).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
2. ROCm Blogs, *Matrix Core Programming on AMD CDNA3 and CDNA4 architecture* (E8M0, scaled-MFMA
   intrinsic, type codes, layouts, FP4 storage helpers).
   <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
3. Open Compute Project, *OCP Microscaling Formats (MX) Specification* v1.0 (block size 32, E8M0 scale,
   MXFP8/MXFP6/MXFP4).
   <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
4. AMD, *Introducing AMD CDNA 4 Architecture* (white paper): micro-scaling, Table 1, Table 2.
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
5. ROCm documentation, *AMD Instinct MI350 Series microarchitecture* (peak theoretical performance).
   <https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html>
6. HIP documentation, *Low precision floating point types* (`__amd_fp4x2_storage_t` and the other
   `__amd_*` types, exclusive to gfx950).
   <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html>
7. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (MXFP8/MXFP6/MXFP4:
   no on MI300X/MI325X, yes on MI350X/MI355X).
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
8. AMD Matrix Instruction Calculator (supported architectures: CDNA1–CDNA3, RDNA3, RDNA4).
   <https://github.com/ROCm/amd_matrix_instruction_calculator>
