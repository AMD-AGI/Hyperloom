---
name: hardware
description: AMD Instinct hardware facts for kernel authoring, rooflines and communication — MI350X/MI355X (gfx950, CDNA4) and MI300X/MI325X (gfx942, CDNA3) — with the spec peaks, how to measure what a node achieves, the memory hierarchy, LDS banks, MFMA and block-scaled matrix instructions, the gfx950 ISA additions, XGMI/collective bandwidth, and where each fact comes from. Use when writing or tuning a kernel, pricing a direction against its floor, writing a kernel plan, reading a roofline, or deciding whether a kernel is compute- or bandwidth-bound.
---

# Hardware: AMD Instinct for kernel authors

The facts a campaign needs when it prices a direction, writes a kernel plan or reads a roofline.
`reference/` holds the detail. It is KernelForge's hardware knowledge base
(`local_knowledge/hardware/`, AMD, MIT licence: `reference/LICENSE-KernelForge`), imported unchanged
on 2026-10-20.

KernelForge's separate gfx950 summary (`knowledge_base/shared/gpu_arch_gfx950.md`) was not imported:
it says "304 CUs", "32 banks" and "similar to MI300X's 5.3 TB/s", all wrong for gfx950.

## The quick facts

| | MI350X / MI355X (gfx950, CDNA4) | MI300X (gfx942, CDNA3) | MI325X (gfx942) |
| --- | --- | --- | --- |
| CUs | **256** (8 XCDs × 32) | 304 (8 × 38) | 304 |
| Wavefront, SIMDs/CU | wave64, 4 | wave64, 4 | wave64, 4 |
| VGPR / AGPR | 512 × 4 B per SIMD / ≤256 | same | same |
| LDS | **160 KiB/CU, 64 banks, 256 B/clk** | 64 KiB, 32 banks, 128 B/clk | as MI300X |
| Direct global→LDS | up to 128 b/lane (`global_load_lds_dwordx4`) | 32 b/lane | as MI300X |
| L2 | 4 MB per XCD (not device-wide), 16-way | 4 MiB per XCD | as MI300X |
| Infinity Cache | 256 MiB, device-shared | 256 MiB | 256 MiB |
| HBM | **288 GB HBM3E, 8.0 TB/s** | 192 GB HBM3, 5.325 TB/s | 256 GB HBM3E, 6.0 TB/s |
| Matrix BF16/FP16 | **2.5 PF** | 1.307 PF | as MI300X |
| Matrix FP8 | **5 PF**, OCP e4m3fn/e5m2 | 2.6 PF, **FNUZ** | as MI300X |
| Matrix FP6 / FP4, MX block-scaled | **10 PF** (FP6 at the FP4 rate), E8M0 scales per 32 | none | none |
| TF32 | **removed** | 654 TF | as MI300X |
| Power, clock | MI350X 1000 W air; MI355X 1400 W liquid, ≤2.4 GHz | 750 W, 2.1 GHz | 1000 W |
| GPU↔GPU | Infinity Fabric, 8 fully connected, 1075 GB/s per card | 7 links per GPU | as MI300X |
| FP16 ridge (spec) | ≈ 312 FLOP/B | ≈ 247 FLOP/B | — |

**Official sources for the gfx950 column** (checked 2026-10-20):
- [ROCm: AMD Instinct MI350 Series microarchitecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html):
  8 XCDs of 32 active CUs (256), 160 KB LDS per CU, 4 MB L2 per XCD, 256 MB Infinity Cache,
  8 TB/s and 288 GB HBM3E as "peak theoretical", 7 Infinity Fabric links at 38.4 Gbps.
- [AMD Instinct MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
  and its [datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf);
  [MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html).
- [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html):
  why graph capture removes most per-launch host overhead.
- The LDS bank count (64) is KernelForge's (`reference/cdna4_mi350/memory.md`); the ROCm page
  doesn't state it.

**gfx950 vs gfx942:**
- MI350X and MI355X are the same chip. MI355X holds higher clocks under load. Compute-bound peaks
  differ only by the clock; bandwidth-bound work is identical.
- Detail: `reference/cdna4_mi350/arch.md`, `peak_tables.md`, `memory.md`, `clocks_power.md`;
  `reference/cdna3_mi300/*`; `reference/shared/hbm_infinity_fabric.md`.

## Measured, not spec

Spec peaks are ceilings. A floor in a kernel plan, a dossier or a roofline uses a number measured
on the node, with its source: a streaming read for HBM bandwidth (not `torch.sum`, which reads
low), RCCL's tests for collectives, a graph-captured loop for launch cost.

**How to read and record measured numbers:**
- **The measured ridge is further right than the spec ridge.** Achieved bandwidth is below the
  spec, so even more kernels are bandwidth-bound than the spec suggests.
- **Record a new measurement** as `value @ GPU, gfx, ROCm and library versions, node, method,
  date`, in the dossier that uses it.
- **When a measurement and the spec disagree,** give both with their sources. Plans use the
  measured one.

## Where to read, by question

| Question | Read |
| --- | --- |
| Is this kernel compute- or bandwidth-bound; what is its floor? | the quick facts, then `reference/shared/hbm_infinity_fabric.md`; count bytes first |
| MFMA shapes, cycles, operand layouts | `reference/shared/matrix_core_mfma_smfmac.md`, `reference/cdna4_mi350/matrix_core_blockscale.md` |
| FP4 / FP6 / MXFP, E8M0 scales, `v_mfma_scale_f32_*_f8f6f4` | `reference/cdna4_mi350/fp4_fp6_microscaling.md` |
| LDS sizing, 64-bank conflicts, padding, read-with-transpose | `reference/shared/memory_model_lds_bank.md`, `reference/cdna4_mi350/memory.md` |
| Occupancy, VGPR/AGPR budgets, wave slots | `reference/shared/wavefront_simd_vgpr_agpr.md`, `reference/cdna3_mi300/occupancy.md` |
| L2 per XCD, workgroup-to-XCD placement, swizzles | `reference/shared/l2_xcd_swizzle.md`, `reference/cdna3_mi300/xcd_chiplet.md` |
| New gfx950 instructions, hazards | `reference/cdna4_mi350/isa_notes.md` |
| Numerics of each dtype (OCP vs FNUZ, rounding) | `reference/shared/dtype_numerics.md` |
| Clocks, power capping, measurement warm-up | `reference/cdna4_mi350/clocks_power.md` |
| Everything, by task or symptom | `reference/INDEX.md` |

## Communication (what KernelForge does not cover)

- **Topology:** 8 GPUs per node, fully connected by Infinity Fabric (xGMI). Measure collective
  bandwidth on the node.
  - Small messages are latency-bound: bus bandwidth stays flat across small message sizes.
  - Decode-size all-reduces cost latency, not bandwidth. Fusing them (with RMSNorm, with the next
    GEMM) or removing them is the lever, not a faster algorithm.
- **Quantized all-reduce (quick-reduce):** the recipes use INT4/INT6/INT8, and it changes numerics.
  It has size thresholds:
  - the default minimum for bf16 at TP4 is 16 MB, so decode-size messages never take it unless the
    minimum is lowered;
  - prefill needs the maximum raised.
  - Knobs: `../serving-playbook/knobs.json`.
- **Overlap:** two-batch overlap and multi-stream overlap need more ROCm hardware queues
  (`GPU_MAX_HW_QUEUES`, 5 in the recipes), or the streams serialize.
- **Across nodes:** MoRI (expert-parallel dispatch/combine and KV transfer over RDMA), with its
  queue-pair and dispatch-cap knobs. See `../serving-playbook/`.

## Rules

- **Never price a direction or a kernel plan against the spec alone.** Use a measured number, or
  measure. A plan whose ceiling comes from the spec says so.
- **Porting from gfx942 to gfx950:**
  - FNUZ becomes OCP FP8;
  - LDS goes from 32 to 64 banks, so re-pad the swizzles;
  - 304 CUs become 256, so redo the grid and wave arithmetic;
  - TF32 is gone.
- **Every fact here carries its source.** When one turns out wrong, fix it here and in the place
  that cited it. The MI350X dossier's "304 CUs" carries a correction note dated 2026-10-20.
