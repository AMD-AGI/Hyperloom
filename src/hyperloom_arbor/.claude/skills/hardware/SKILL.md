---
name: hardware
description: AMD Instinct hardware facts for kernel authoring, rooflines and communication — MI350X/MI355X (gfx950, CDNA4) and MI300X/MI325X (gfx942, CDNA3) — with the spec peaks, how to measure what a node achieves, the memory hierarchy, LDS banks, MFMA and block-scaled matrix instructions, the gfx950 ISA additions, Infinity Fabric links and collectives, and the public source of each fact. Use when writing or tuning a kernel, pricing a direction against its floor, writing a kernel plan, reading a roofline, or deciding whether a kernel is compute- or bandwidth-bound.
---

# Hardware: AMD Instinct for kernel authors

The facts a campaign needs when it prices a direction, writes a kernel plan or reads a roofline.
`reference/` holds the detail. It began as KernelForge's hardware knowledge base
(`local_knowledge/hardware/`, AMD, MIT licence: `reference/LICENSE-KernelForge`). Its cards have since
been checked against public AMD sources: every fact kept cites one, and facts without a public source
were cut.

KernelForge's separate gfx950 summary (`knowledge_base/shared/gpu_arch_gfx950.md`) was not imported:
it says "304 CUs", "32 banks" and "similar to MI300X's 5.3 TB/s", all wrong for gfx950 [1][2][3].

**When the campaign loads `knowledge:search`, `reference/` is behind it.** Reading its pages is denied. Find, get or
ask instead (`python3 -m arbor.knowledge find|get|ask`, source `hardware-reference`). This file stays as it is: it
is the most trusted, and short.

## The quick facts

| | MI350X / MI355X (gfx950, CDNA4) | MI300X (gfx942, CDNA3) | MI325X (gfx942) |
| --- | --- | --- | --- |
| CUs | **256** (8 XCDs × 32) [2] | 304 (8 × 38) [4] | 304 [4] |
| Wavefront, SIMDs/CU | wave64 [1], 4 [4][7] | wave64 [5], 4 [4] | wave64, 4 |
| VGPR / AccVGPR | ≤256 + ≤256 per wave [1]; 512 KiB file per CU [6] | same [5][6] | same |
| LDS | **160 KiB/CU, 64 banks** [1], **256 B/clk** [2] | 64 KiB, 32 banks [5], 128 B/clk [2] | as MI300X |
| Direct global→LDS | up to 128 b/lane (`global_load_lds_dwordx4`) [1] | 32 b/lane [5] | as MI300X |
| L2 | 4 MB per XCD (not device-wide), 16-way [2] | 4 MB per XCD [4] | as MI300X |
| Infinity Cache | 256 MB, device-shared [2] | 256 MB [4] | 256 MB [4] |
| HBM | **288 GB HBM3E, 8.0 TB/s** [2] | 192 GB HBM3, 5.3 TB/s [4] | 256 GB HBM3E, 6.0 TB/s [4] |
| Matrix BF16/FP16 | **2.5 PF** (MI355X), 2.3 PF (MI350X) [2] | 1.307 PF [8] | as MI300X [4] |
| Matrix FP8 | **5 PF** (MI355X), 4.6 PF (MI350X) [2]; OCP e4m3fn/e5m2 [9] | 2.6 PF [8], **FNUZ** [9] | as MI300X [4] |
| Matrix MXFP6 / MXFP4 | **10 PF** (MI355X), 9.2 PF (MI350X), MXFP6 at the MXFP4 rate [2]; E8M0 scale per 32 [1] | none [4] | none [4] |
| TF32 | **removed** (software via BF16) [2] | 653.7 TF [8] | as MI300X [4] |
| Power, clock | MI350X 1000 W air, 2.2 GHz; MI355X 1400 W liquid, 2.4 GHz [2] | 750 W, 2.1 GHz [4] | 1000 W, 2.1 GHz [4] |
| GPU↔GPU | Infinity Fabric, 8 fully connected; 7 links × 144 GB/s = 1,008 GB/s per GPU [2][10] (1,075 GB/s in [2]'s Figure 5 and [4]) | 7 links per GPU [8] | as MI300X |
| FP16 ridge (spec) | ≈ 312 FLOP/B (2.5 PF / 8 TB/s) | ≈ 247 FLOP/B (1307 TF / 5.3 TB/s) | — |

**Sources:**
1. [AMD CDNA4 Instruction Set Architecture Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
   (5 Aug 2025): §2.2.1 and §11.1, LDS of 160 kB in 64 banks of 640 dwords; §3.6.4, VGPRs; §9.1.9
   and §10, loads to LDS up to `DWORDX4`; §7.2.1, one E8M0 scale per 32 K-values.
2. [Introducing AMD CDNA 4 Architecture](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
   (white paper): 8 XCDs of 32 active CUs, LDS 160 KB at 256 bytes per clock (double CDNA3's), L2 and
   Infinity Cache, 288 GB HBM3E at 8 TB/s, Table 1 and Table 2 (peaks, clocks, power, links).
3. [ROCm: AMD Instinct MI350 Series microarchitecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html):
   256 CUs, 160 KB LDS, 8 TB/s and 288 GB as "peak theoretical", 7 Infinity Fabric links at 36 Gbps.
4. [ROCm: MI300 Series / MI350 Series workload optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html):
   comparison tables for MI300X, MI325X, MI350X and MI355X (CUs, clocks, power, caches, memory, peaks,
   MXFP support, 38.4 Gbps links); 4 SIMDs per CU.
5. [AMD Instinct MI300 / CDNA3 Instruction Set Architecture Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf):
   LDS of 64 kB in 32 banks; loads to LDS up to a dword.
6. [ROCm: AMD GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html):
   VGPR file 512 KiB per CU, LDS 160 KiB on gfx950.
7. [ROCm Blogs: Occupancy Math on the AMD MI355X GPU (CDNA4)](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html).
8. [ROCm: AMD Instinct MI300 Series microarchitecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html):
   peak table (1307.4 TF FP16, 2614.9 TF FP8, 653.7 TF TF32), seven Infinity Fabric links per GPU.
9. [HIP: Low precision floating point types](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html):
   FNUZ on gfx94x, OCP on CDNA4.
10. [AMD Instinct MI355X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf):
    7 × 144 GB/s Infinity Fabric links; the MI355X sustains higher performance under prolonged load.
11. [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html):
    a graph is launched with a single call, with less overhead than launching its work call by call.
12. C. Ambati, T. Diep, [AMD MI300X GPU Performance Analysis](https://arxiv.org/abs/2510.27583),
    arXiv:2510.27583: about 45% of peak FLOPs on average across FP8, BF16 and FP16, and about 81% of the
    5.3 TB/s peak bandwidth, on MI300X.
13. [ROCm: environment variables](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html)
    (`GPU_MAX_HW_QUEUES`).
14. [ROCm Blogs: QuickReduce](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html).
15. [MORI](https://github.com/ROCm/mori) (ROCm): MORI-EP dispatch/combine, MORI-IO for KV-cache transfer.

**gfx950 vs gfx942:**
- MI350X and MI355X are the same chip (256 CUs, 1,024 matrix cores) [2]. The MI355X's peak clock is
  2.4 GHz against 2.2 [2], and AMD says its power budget sustains higher performance under prolonged load
  [10]. Compute-bound peaks differ by the clock; the peak memory bandwidth is the same 8 TB/s [2].
- Detail: `reference/cdna4_mi350/arch.md`, `peak_tables.md`, `memory.md`, `clocks_power.md`;
  `reference/cdna3_mi300/*`; `reference/shared/hbm_infinity_fabric.md`.

## Measured, not spec

Spec peaks are ceilings. A floor in a kernel plan, a dossier or a roofline uses a number measured
on the node, with its source:
- **HBM bandwidth:** a streaming read, cold and hot (through the Infinity Cache), at the sizes the
  kernel touches.
- **Collectives:** RCCL's tests on the node, bus bandwidth across the message sizes that run.
- **Launch cost:** a graph-captured loop, per added launch [11].
- **Matrix throughput:** sustained rates sit well below the peak (about 45% on MI300X [12]; AMD
  publishes no sustained figure for CDNA4).

**How to read and record measured numbers:**
- **A measured ridge uses measured numbers on both axes.** Both sit below the spec (on MI300X about 81%
  of peak bandwidth and 45% of peak FLOPs [12]), so the measured ridge can fall on either side of the
  spec ridge.
- **Record a new measurement** as `value @ GPU, gfx, ROCm and library versions, node, method,
  date`, in the dossier that uses it.
- **When a measurement and the spec disagree,** give both with their sources. Plans use the
  measured one.

## Where to read, by question

| Question | Read |
| --- | --- |
| Is this kernel compute- or bandwidth-bound; what is its floor? | the quick facts and "Measured, not spec" above, then `reference/shared/hbm_infinity_fabric.md`; count bytes first |
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

- **Topology:** 8 GPUs per node, fully connected by Infinity Fabric (xGMI) [2][3]. Measure collective
  bandwidth on the node, across the message sizes that run.
- **Quantized all-reduce (QuickReduce):** vLLM's `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP|INT8|INT6|INT4`
  quantizes the all-reduce payload, so the INT modes change numerics [14]. vLLM picks it only where it
  beats its other all-reduce paths, above a message-size minimum; small messages stay on the custom
  all-reduce [14]. Read the thresholds of the vLLM version that runs. See `../serving-playbook/`.
- **Overlap:** HIP maps streams beyond `GPU_MAX_HW_QUEUES` hardware queues (default 4) onto the same
  queues round-robin [13], so overlap schemes with more streams than queues share them.
- **Across nodes:** MoRI (expert-parallel dispatch/combine and KV-cache transfer over RDMA) [15]. See
  `../serving-playbook/`.

## Rules

- **Never price a direction or a kernel plan against the spec alone.** Use a measured number, or
  measure. A plan whose ceiling comes from the spec says so.
- **Porting from gfx942 to gfx950:**
  - FNUZ becomes OCP FP8 [9];
  - LDS goes from 32 to 64 banks, so re-pad the swizzles [1][5];
  - 304 CUs become 256, so redo the grid and wave arithmetic [4];
  - TF32 is gone [2].
- **Every fact here carries its source.** When one turns out wrong, fix it here and in the place
  that cited it, with a correction note.
