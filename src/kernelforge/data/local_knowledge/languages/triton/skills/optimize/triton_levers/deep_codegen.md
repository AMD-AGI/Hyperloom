---
title: Triton on AMD — codegen deep dive (TritonGPU → MFMA → AMDGCN)
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8]
regimes: [both]
status: competitive
updated: 2026-08-23
sources:
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45
---

# Triton on AMD — codegen deep dive

How `@triton.jit` becomes `v_mfma_*` + `ds_*_b128` + `global_load_dwordx4`. This is the model you need
to predict why a knob helps and to read the ISA (see [isa_verify.md](isa_verify.md)).

## 1. `tl.dot` → MFMA (the heart of GEMM/attention)
`tl.dot(a, b, acc)` lowers, in the **TritonAMDGPU** dialect, to a `dot` op carrying an **MFMA layout**,
then to a sequence of `v_mfma_f32_*` instructions. The chosen MFMA shape is governed by
`matrix_instr_nonkdim` (16 or 32) and the input dtype:

| `tl.dot` input | MFMA instruction (gfx942) | K/instr | recommended `BLOCK_K` |
|---|---|---|---|
| fp16 / bf16 | `v_mfma_f32_16x16x16` (nonkdim=16) | 16 | 32–64 |
| fp16 / bf16 | `v_mfma_f32_32x32x8` (nonkdim=32) | 8 | 32–64 |
| fp8 (fnuz) | `v_mfma_f32_16x16x32_fp8_fp8` | 32 | 64–128 |
| fp8 (fnuz) | `v_mfma_f32_32x32x16_fp8_fp8` | 16 | 64–128 |
| int8 | `v_mfma_i32_16x16x32_i8` | 32 | 64–128 |

On **gfx950 (CDNA4)** the dtype set widens to OCP fp8/bf8 and block-scaled `mfma_scale_f32_*_f8f6f4`
(MXFP8/6/4); see [pitfalls.md](pitfalls.md) for the FNUZ↔OCP split.

Each MFMA is **wavefront-wide**: the 64 lanes collectively hold the A/B/C tiles. You never write the
instruction by hand in Triton, but the layout it picks dictates VGPR/AGPR pressure (→ occupancy). The
MFMA accumulator lives in **AGPRs**; the epilogue `convert_layout` (reblock from MFMA layout to the
output blocked layout) is an LDS round-trip — `OPTIMIZE_EPILOGUE=1` drops it.

**Why 16x16 ≥ 32x32 on MI300X (AMD guidance):** `nonkdim=32` produces a larger per-wave accumulator
→ more AGPR/VGPR pressure → can force occupancy down or spill, and gives the scheduler coarser
granularity (fewer chances to hide load latency). Pick 16 unless 32 measurably wins for your shape.

## 2. Block layouts and `convert_layout`
TTGIR assigns each tensor a **layout** (blocked, MFMA/dot-operand, slice). Mismatched producer/consumer
layouts force a `convert_layout` op, which lowers to an **LDS round-trip** (`ds_write` then `ds_read`
with a different swizzle). The two layout costs that matter:
- **Epilogue convert** (MFMA-layout acc → blocked store layout): killed by `OPTIMIZE_EPILOGUE=1`.
- **dot-operand convert** (loaded blocked tile → MFMA dot-operand layout): unavoidable for GEMM, but
  its LDS swizzle is tuned by `kpack` (wider `ds_read_b128`) and tile shape.

`grep "triton_gpu.shared"` in `MLIR_ENABLE_DUMP` output to see the LDS bytes each layout reserves.

## 3. The stream pipeliner (`num_stages` on AMD)
AMD does **not** use NVIDIA's `cp.async` + mbarrier pipeline. Instead the TritonAMDGPU
**stream-pipeliner** pass (`add_schedule_loops` / `add_pipeline`) software-pipelines the K-loop:
prefetch the next K-tile's global loads into LDS while the current tile feeds the MFMAs. `num_stages`
= number of in-flight LDS-staged tiles.

| Pattern | `num_stages` | Why |
|---|---|---|
| single GEMM | **1–2** | extra stages buffer more loads in LDS → crush occupancy on 64 KB LDS |
| two fused GEMMs (Flash-Attention) | **1** | two dots + softmax already saturate LDS/regs |
| GEMM + non-GEMM epilogue | 2 | |
| no-GEMM (elementwise/reduction) | 1 | |

**The pass only fires on a schedulable loop.** It rewrites `scf.for` (Triton `for` / `tl.range`) with a
loop-invariant bound and staged loads whose addresses are affine in the induction variable. A `while`
whose exit test reads memory, or a load addressed by another load in the same iteration, is left
unpipelined — silently, and `knobs.amd.use_async_copy` is lost with it. See
`local_knowledge/common_methodology/optimization/loop_form_and_pipelining.md` for the rewrite
(shape-static range + mask) and what it costs.

`num_stages>1` is the prerequisite for **block ping-pong** (`knobs.amd.use_block_pingpong`): two warp
groups alternate so one issues MFMA while the other issues VMEM/DS. Defaults: `HIPOptions.num_stages`
is **2** upstream.

## 4. Global loads: `global_load_dwordx4`, buffer loads, async-copy
- **Wide loads:** a well-formed kernel emits **128-bit** `global_load_dwordx4` in the K-loop (4 fp32 /
  8 fp16 per lane). Narrow `global_load_dword` means poor vectorization — bump tile/`kpack`,
  add `__restrict`-equivalent contiguity (Triton infers from masks/strides).
- **Buffer loads:** `buffer_load` uses a 128-bit resource descriptor with **hardware bounds checking**
  — OOB lanes return 0, removing the predication branch from masked tail loads. Critically, on many
  builds **buffer ops are NOT the default**; enable `knobs.amd.use_buffer_ops` (a.k.a. the
  "buffer-loads-not-default" issue) for masked GEMM/attention tails. Verify `buffer_load_dwordx4`
  appears.
- **Async copy (direct-to-LDS):** `knobs.amd.use_async_copy` emits `global_load_lds` /
  `buffer_load ... lds` (skip VGPR staging → frees registers). **Default on gfx950**, experimental on
  gfx942. Lowers to `s_wait_asynccnt`-gated loads. For the underlying HIP ISA see the HIP LDS/async
  card (`languages/hip/skills/optimize/hip_levers/lds_async.md`).

## 5. LDS swizzle & `kpack`
Triton inserts a **swizzled** shared layout so 64-lane waves hit distinct LDS banks (32 banks × 4 B,
128 B/clk on CDNA3). `kpack=2` packs 2 K-slices per LDS read → emits `ds_read_b128` instead of two
`ds_read_b64`, halving LDS instruction count. `kpack=2` is a near-universal win for fp16/bf16 GEMM
with `BLOCK_K≥64` on **gfx942**; it is **deprecated/forced to 1 on gfx950** (the backend warns).

## 6. LLVM/AMDGCN stage attributes
The LLVM-IR stage attaches:
- `"amdgpu-waves-per-eu"="N"` from `waves_per_eu` → backend trims VGPRs to fit N waves/EU.
- `"amdgpu-flat-work-group-size"` from `num_warps·64`.
- denormal-fp-math flags.
The register allocator then sets `.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size` (LDS), and
`.private_segment_fixed_size` (**scratch — must be 0**; nonzero = spilling to HBM).

## 7. Worked: an annotated K-loop you can predict the ISA of
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)     # -> AGPR accumulator (MFMA layout)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None,:] < K-k*BLOCK_K, other=0.0)  # -> global_load_dwordx4
    b = tl.load(b_ptrs, mask=offs_k[:,None] < K-k*BLOCK_K, other=0.0)
    acc = tl.dot(a, b, acc)                          # -> ds_read_b128 (kpack=2) + v_mfma_f32_16x16x16
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
c = acc.to(c_ptr.dtype.element_ty)                   # OPTIMIZE_EPILOGUE=1 -> no convert_layout
```
With `matrix_instr_nonkdim=16, kpack=2, num_stages=2, OPTIMIZE_EPILOGUE=1` you should see, in the inner
loop: `global_load_dwordx4` (×N), `ds_read_b128`, dense `v_mfma_f32_16x16x16`, no `v_accvgpr_*`, and
`.private_segment_fixed_size: 0`. Anything else → retune (see [isa_verify.md](isa_verify.md)).

## Sources
- Triton AMD backend `HIPOptions` / stream-pipeliner / knobs.amd.*: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- Matrix Core programming CDNA3/CDNA4 (MFMA shapes, AGPR accumulators, block-scaled f8f6f4): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- A Deep Dive Into AMD Triton Compilation (TTIR→TTGIR→TritonAMDGPU→AMDGCN, convert_layout): https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45
- OPTIMIZE_EPILOGUE / ds_read_b128 / global_load_dwordx4: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
