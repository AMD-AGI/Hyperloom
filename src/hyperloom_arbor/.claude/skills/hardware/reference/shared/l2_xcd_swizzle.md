---
title: L2 / XCD locality, tile swizzle & coalescing (CDNA cross-gen)
kind: hardware
gens: [gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# L2 / XCD locality, tile swizzle & coalescing

> Numbers in brackets point to the Sources at the end.

## TL;DR
> On chiplet CDNA (MI300X/MI350X) **L2 is per-XCD**, not global [2][8]. Two perf rules fall out:
> (1) make GEMM tile counts and grid dims **multiples of 8** so work distributes evenly across the
> 8 XCDs and reuse hits the same XCD's L2 [1]; (2) avoid the **512 B-stride "Tagram" hotspot** on the
> TN GEMM case [1]. Always coalesce to cache-line-aligned `dwordx4` loads [1][7].

## Concepts

### Why "multiple of 8"
- MI300X/MI325X have **8 XCDs**; MI350X/355X have **8 XCDs** too [1]. In SPX mode the hardware
  distributes workgroups **round-robin** across the XCDs [3][4]. AMD's guidance: use multiples of the
  XCD count (8) to maximize L2 cache efficiency, and put workgroups that share data on the same XCD
  [1]. It follows that a non-multiple leaves some XCDs idle in the tail and scatters reuse across
  XCDs, where it misses to the Infinity Cache [2].
- The same guide asks for **≥1024 workgroups** on MI300X [1] (≈3.4 per CU on 304 CUs): it fills the
  device and gives the scheduler tail slack.

### Grid/tile rules of thumb
| Rule | Value | Why |
|---|---|---|
| Workgroups per launch | **≥ 1024** [1] | fill 304 CUs (MI300X) / 256 CUs (MI350X) [1] + tail hiding |
| Tile M/N | **multiple of 8** [1] | even XCD distribution, L2 reuse, no straggler XCD |
| Threads/block | ≤ 1024 (16 waves) [6] | a workgroup runs on one CU [6]; occupancy is per-SIMD |
| MFMA shape | 16×16 over 32×32 on MI300X [1] | see [matrix_core_mfma_smfmac.md](matrix_core_mfma_smfmac.md) |

### The 512 B-stride "Tagram" hotspot
On MI300, a GEMM whose matrix stride is a **multiple of 512 bytes** can cause **Tagram channel
hotspotting**, a significant performance drop that is worst for **TN** transpose cases; it shows up as
higher VMEM instruction latency [1]. Mitigation [1]:
- **Pad the stride** so it is not a multiple of 512 B — e.g. for TN F16 GEMM set `lda = ldb = K + 128`
  when `K % 256 == 0`.

### `OPTIMIZE_EPILOGUE`
A Triton-on-ROCm environment variable [5]: `OPTIMIZE_EPILOGUE=1` stores the MFMA results **in the
MFMA layout directly**, removing the `convert_layout` in the epilogue. The default (0) converts the
results to a blocked layout through LDS (padded against bank conflicts), so the global store can be
`global_store_dwordx4` — at the cost of extra LDS use that can reduce occupancy. AMD's guide says to
turn it on in most cases: it costs global store efficiency, but usually little kernel time [5].
Verify on the target shape.

### Coalescing (feeds everything above)
- Have consecutive lanes access consecutive addresses, aligned to cache-line boundaries [7]; make sure
  loops emit `global_load_dwordx4` (16 B/lane) [1], so each wave fills full 128 B cache lines [2] with
  fewer instructions.
- `buffer_load`/`buffer_store` with a descriptor (V#) are range-checked: out-of-range reads return
  zero and writes are dropped [6] → cheaper guards in tiled GEMM than branchy `global_load`.

## The levers
1. **Make tile/grid counts multiples of 8** for even XCD spread + L2 reuse [1].
2. **Launch ≥1024 workgroups** [1].
3. **Break 512 B strides** (pad the stride) to dodge the Tagram hotspot on TN GEMM [1].
4. **`OPTIMIZE_EPILOGUE=1`** to skip the epilogue LDS round-trip (verify store coalescing on your
   shape) [5].
5. **`global_load_dwordx4`, cache-line aligned**; use `buffer_*` for cheap bounds checks [1][6].
6. **Keep an operand's reuse on one XCD** (tile schedule) so it stays in that XCD's 4 MiB L2 [1][2].
7. **Consider CPX/NPS partitioning** to make L2/HBM XCD-local for many-small-kernel workloads [3][4]
   → see each gen's `xcd_chiplet.md` / `arch.md`.

## Pitfalls
- **Non-multiple-of-8 grids** leave straggler XCDs and scatter L2 reuse [1].
- **TN GEMM at a 512 B stride** hits the Tagram hotspot; the symptom is higher VMEM latency at
  specific N/K [1].
- **Assuming L2 is global** — cross-XCD reuse does **not** hit the 4 MiB L2; it falls to the 256 MiB
  Infinity Cache [2].

## Verify
- `rocprof-compute`: L2 hit rate and VMEM latency per shape; look for cliffs at 512 B-stride N/K; XCD
  load balance.
- A/B a GEMM with tile dims rounded to a multiple of 8 vs not, at fixed (M,N,K).
- Toggle `OPTIMIZE_EPILOGUE` and compare TFLOP/s + store-coalescing counters.

## Sources
1. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (Tensile tips: clock differences, WorkGroupMapping, GEMM stride issues; Triton: overall GPU resource utilization) — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
2. AMD, *AMD CDNA 3 Architecture* white paper (memory section) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
3. ROCm Blogs, *Deep dive into the MI300 compute and memory partition modes* (workgroup scheduling) — https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
4. AMD GPU driver documentation, *AMD Instinct MI300X GPU Partitioning Overview* — https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html
5. ROCm 6.1.1 documentation, *Optimizing Triton kernels* (`OPTIMIZE_EPILOGUE`) — https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
6. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §4.3 work-groups, §9.1.5 range checking) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
7. HIP documentation, *Hardware implementation* (memory coalescing) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
8. AMD, *AMD CDNA 4 Architecture* white paper (L2 per XCD) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
