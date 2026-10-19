---
title: MI350X / MI355X — chip orientation, cheat sheet, peaks
kind: hardware
topic: overview
gens: [gfx950]
updated: 2026-08-28
---

# MI350X / MI355X (gfx950) — orientation

**Start here.** One screen of constants, then the peak tables. Every other card in this folder
assumes these numbers.

## The one-screen cheat sheet

| Fact | Value | Why it matters |
|---|---|---|
| ISA target | **`gfx950`** (`--offload-arch=gfx950`) | calculator keyword `cdna4` |
| Wavefront | **64 lanes** | all shuffle/ballot/reduction math is mod 64 |
| Active CUs | **256** = 8 XCD × 32 | grid still wants ≥1024 workgroups |
| XCDs | **8**, on **2 I/O dies** | L2 is per-XCD, not global |
| SIMDs / CU | **4** | occupancy is computed per-SIMD |
| Wave slots | 8/SIMD → **32/CU** | hard cap |
| Registers | **512 × 4 B per SIMD**, 16-granule | + ≤256 AGPR, unified pool |
| LDS | **160 KiB/CU**, **64 banks**, **256 B/clk** | 2.5× capacity, 2× banks vs CDNA3 |
| Direct global→LDS | **128 b/lane** (1/2/4/12/16 DWORD) | 4× wider than CDNA3 |
| L2 | **per-XCD** | cross-XCD reuse is not an L2 hit |
| Infinity Cache | **256 MiB** (MALL/L3), device-shared | the first shared level |
| HBM3E | **288 GB**, **8.0 TB/s** | 8 stacks × 36 GB (12-Hi) |
| Matrix cores | **1024** (256 CU × 4) | 2× the per-CU rate of CDNA3 |
| FP8 encoding | **OCP** (E4M3FN / E5M2) | **not FNUZ** — re-cast checkpoints |
| TF32 | **removed** | fall back to BF16 or FP32 |
| Process | TSMC **N3P** (XCD) + N6 (IOD) | 185 B transistors |
| TDP | **1000 W** (MI350X air) / **1400 W** (MI355X liquid) | same compute, different sustained clock |
| Engine clock | up to ~**2400 MHz** (MI355X) | basis of the peak math |

## Peak throughput

Per OAM, vendor-reported. **All theoretical** — see the reality check below.

| Computation | Peak | vs FP32 | vs CDNA3 |
|---|---|---|---|
| FP16 / BF16 matrix | **2.5 PFLOP/s** | 16× | **2×** (1307 → 2500) |
| FP8 (OCP) matrix | **5 PFLOP/s** | 32× | **2×** (2615 → 5000) |
| FP6 matrix | **10 PFLOP/s** | 64× | new |
| FP4 matrix | **10 PFLOP/s** | 64× | new |
| MXFP8 / 6 / 4 | matches the element rate | — | new |
| INT8 | ~5 POPS | 32× | 2× |
| FP32 matrix | 157.3 TFLOP/s | 1× | ~same |
| FP64 vector | 78.6 TFLOP/s | 0.5× | vector ~same, **matrix halved** |
| TF32 | **removed** | — | gone |

**FP6 and FP4 share the 10 PF rate.** Choosing FP6 over FP4 costs accuracy headroom, not throughput —
so prefer FP6 whenever FP4 is too lossy.

## Roofline ridge points

`ridge = peak ÷ 8.0 TB/s`. Left of it → bandwidth-bound; right → compute-bound.

| dtype | ridge |
|---|---|
| FP16 / BF16 | **≈ 312 FLOP/byte** |
| FP8 | ≈ 625 FLOP/byte |
| FP6 / FP4 | ≈ 1250 FLOP/byte |
| FP32 | ≈ 20 FLOP/byte |

The FP16 ridge is **higher than CDNA3's ≈247** because the matrix core doubled while bandwidth grew
less. Practical reading: **more kernels are bandwidth-bound on this part than on MI300X.** A kernel that
was borderline compute-bound before may now sit left of the ridge — re-classify ports, do not carry the
verdict over.

## Sustained reality — the bar is not peak

Tuned GEMM sustains **~45–55% of theoretical matrix peak**. That is a software-maturity ceiling, not a
hardware defect. Never quote peak as achievable; the real bar is the best tuned library kernel for
that shape. Record measurements as `value @ MI355X gfx950, ROCm <ver>, <lib>@<ver>, <date>`.

## Package topology

8 XCDs (TSMC N3P) hybrid-bonded onto **2 I/O dies** (N6) — CDNA3 used 4 IODs. Each IOD connects
4 HBM3E stacks (36 GB, 12-Hi) → 288 GB total. Infinity Fabric plus the 256 MiB Infinity Cache form the
device-shared coherence layer; **L2 stays per-XCD**. Inter-package: 4th-gen Infinity Fabric,
**1075 GB/s** bidirectional aggregate per card, 8-GPU fully connected.

## The deltas that break ported kernels

Four things silently change behaviour when moving a working MI300X kernel here:

1. **FP8 FNUZ → OCP** — different bias and saturation. Bit-copying corrupts silently → `mi350_dtypes.md`
2. **LDS 32 → 64 banks** — any inherited swizzle is unverified → `mi350_lds.md`
3. **304 → 256 CUs** and **64 → 160 KiB LDS** — occupancy and grid math both move → `mi350_execution.md`
4. **TF32 removed** — the code path does not exist → `mi350_isa.md`

## Verify
- `rocminfo` / `amd-smi static` → `gfx950`, 256 CU, 288 GB.
- `rocprof-compute` for occupancy (against the 160 KiB LDS limit), L2/L3 hit rates, HBM BW, matrix
  utilization.
- Treat `amd_matrix_instruction_calculator --architecture cdna4` as authoritative over any table here.

## Scope
This folder covers **gfx950 only** (MI350X / MI355X). CDNA1–CDNA3 are not documented here; if you are
targeting MI300X the numbers above are wrong for you — use AMD's CDNA3 material.

## Related
`mi350_execution.md` · `mi350_matrix_core.md` · `mi350_dtypes.md` · `mi350_lds.md` ·
`mi350_memory.md` · `mi350_chiplet.md` · `mi350_isa.md` · `mi350_clocks.md`
