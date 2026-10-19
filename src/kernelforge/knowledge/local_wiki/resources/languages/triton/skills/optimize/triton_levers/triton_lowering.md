---
title: Triton on AMD — the compilation model you need to predict a knob's effect
kind: language
lever: triton_lowering
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45
---

# How Triton becomes AMDGCN

## Route here when
- A knob moved performance and you want to know the mechanism, not just the number.
- You are about to open the disassembly and want to know what "correct" looks like.
- `num_stages` does nothing no matter what you set it to.
- You need to explain why a kernel that works on NVIDIA is slow here.

Everything described below happens in the `TritonAMDGPU` stage of the pipeline.

## 1. `tl.dot` becomes MFMA
`tl.dot(a, b, acc)` first becomes a `dot` operation tagged with an **MFMA layout**, and that lowers to a
run of `v_mfma_f32_*` instructions. Which instruction depends on the input dtype and on
`matrix_instr_nonkdim`.

| Input to `tl.dot` | Instruction on gfx950 | K per instruction | `BLOCK_K` worth trying |
|---|---|---:|---|
| fp16 / bf16 | `v_mfma_f32_16x16x32` (nonkdim=16) | 32 | 32–64 |
| fp16 / bf16 | `v_mfma_f32_32x32x16` (nonkdim=32) | 16 | 32–64 |
| fp8 (OCP) | `v_mfma_f32_16x16x128_f8f6f4` | 128 | 64–128 |
| fp8 (OCP) | `v_mfma_f32_32x32x64_f8f6f4` | 64 | 64–128 |
| MXFP8 / 6 / 4 | `v_mfma_scale_f32_*_f8f6f4` | — | block-scaled, E8M0 scales |
| int8 | `v_mfma_i32_16x16x64_i8` | 64 | 64–128 |

Every MFMA is **wavefront-wide**: all 64 lanes jointly hold A, B and C. You never emit one by hand, but
the layout the compiler picks determines VGPR and AGPR pressure, and therefore occupancy — which is why
a shape choice this far down the stack shows up in your throughput.

The accumulator lives in **AGPRs**. Getting it back out to a storable arrangement means a
`convert_layout`, and on AMD that is a round trip through LDS. `OPTIMIZE_EPILOGUE=1` removes it.

**On preferring nonkdim=16.** Two independent reasons, and most write-ups only give the second:

1. A 32×32 accumulator occupies **16 C registers per lane against 16×16's 4**. That is register
   pressure you pay in occupancy, or in spills.
2. It schedules coarsely — fewer instruction boundaries at which the compiler can hide load latency.

On top of both, it draws more power and so clocks lower. Start at 16; move to 32 only when a
measurement says so.

## 2. Layouts, and what a mismatch costs
In TTGIR every tensor carries a **layout** — blocked, MFMA / dot-operand, or slice. When a producer's
layout does not match what the consumer wants, the compiler inserts a `convert_layout`, which becomes a
`ds_write` followed by a `ds_read` under a different swizzle. An LDS round trip, in other words.

Two of these show up in a GEMM, and they are not equally fixable:

| Conversion | What it costs | What to do |
|---|---|---|
| Epilogue: MFMA accumulator → blocked store layout | one LDS round trip per output tile | `OPTIMIZE_EPILOGUE=1` eliminates it outright |
| dot-operand: loaded blocked tile → MFMA operand layout | unavoidable — GEMM needs it | you cannot remove it; the LDS swizzle it uses is shaped by your tile choice |

To see how much LDS each layout is reserving, dump the IR and look at the shared-memory allocations:

```bash
MLIR_ENABLE_DUMP=1 python your_kernel.py 2>&1 | grep "triton_gpu.shared"
```

## 3. The stream pipeliner — what `num_stages` actually drives
**AMD does not use NVIDIA's `cp.async` plus mbarrier machinery.** The TritonAMDGPU **stream-pipeliner**
pass (`add_schedule_loops`, `add_pipeline`) software-pipelines the K-loop: while the current K-tile
feeds the matrix core, the next tile's global loads are already moving into LDS. `num_stages` is how
many LDS-staged tiles are allowed in flight.

| Kernel shape | `num_stages` | Reasoning |
|---|---|---|
| one GEMM | **1–2** | each extra stage buys prefetch depth at the cost of LDS |
| two fused GEMMs (Flash-Attention) | **1** | two dots plus softmax already consume the LDS and register budget |
| GEMM with a non-GEMM epilogue | 2 | |
| no GEMM at all (elementwise, reduction) | 1 | nothing to pipeline |

gfx950's **160 KiB of LDS** — two and a half times gfx942 — raises the ceiling. A third stage that did
not fit before may fit now, so re-tune rather than carrying a stage count over.

> **The pass only runs on a loop it can schedule, and it says nothing when it can't.** It rewrites an
> `scf.for` (a Triton `for` or `tl.range`) whose trip count is **loop-invariant** and whose staged loads
> have addresses **affine in the induction variable**. A `while` loop whose exit condition reads memory,
> or a load whose address comes from another load in the same iteration, is left **unpipelined,
> silently** — and `knobs.amd.use_async_copy` goes with it.
>
> **This is why a flat `num_stages` sweep is a diagnosis, not a result.** If 1, 2 and 3 all measure the
> same, the pipeliner never fired. The fix is to change the loop into a static-range-plus-mask form —
> see `../../../../common_methodology/optimization/lever_loop_form.md`.

`num_stages > 1` is also the precondition for **block ping-pong**
(`knobs.amd.use_block_pingpong`), where two warp groups alternate so one issues MFMA while the other
issues memory traffic.

## 4. What the global loads compile to
| ISA form | What it is | How you get it |
|---|---|---|
| `global_load_dwordx4` | a **128-bit** load — the one you want | contiguous, aligned, well-formed kernel |
| `global_load_dword` | scalar; vectorization failed | grow the tile; check that the mask and strides really are contiguous |
| `buffer_load` | 128-bit through a descriptor with **hardware bounds checking** — out-of-range lanes return zero, no predication branch needed | `knobs.amd.use_buffer_ops`, **which is not on by default in many builds** |
| `global_load_lds` / `buffer_load ... lds` | asynchronous, straight into LDS, skipping VGPR staging — frees registers | `knobs.amd.use_async_copy`, **default on gfx950**, experimental on gfx942 |

The buffer-ops default is worth checking explicitly. **If the disassembly shows `global_load_dword`
surrounded by `v_cmp` predication on your masked tail loads, you are on the slow path and a single flag
away from the fast one.** Nothing in the output warns you.

gfx950 also widens direct-to-LDS to **128 bits per lane** and adds **read-with-transpose** `ds` loads.

## 5. LDS swizzling
Triton lays shared memory out with a **swizzle** so that a 64-lane wave touches 64 distinct banks. On
gfx950 that means **64 banks of 4 B at 256 B/clock** — double gfx942 on both counts.

`kpack=2` used to matter on gfx942: it packed two K-slices into one LDS read so the compiler could emit
`ds_read_b128` instead of a pair of `ds_read_b64`. It is **deprecated and forced to 1 on gfx950**,
where you should be getting `ds_read_b128` without asking.

## 6. What the LLVM stage adds
Before AMDGCN, the LLVM-IR stage attaches:

- `"amdgpu-waves-per-eu"="N"` from your `waves_per_eu` — the backend then trims VGPR usage to make N
  waves fit
- `"amdgpu-flat-work-group-size"` from `num_warps · 64`
- denormal handling flags

Register allocation then fixes `.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size` (LDS), and
`.private_segment_fixed_size`. **That last one must be 0.** Anything else means the kernel is spilling
to HBM, and no amount of tile tuning will compensate.

## 7. A loop whose ISA you can predict
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)     # AGPR accumulator, MFMA layout
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)   # global_load_dwordx4
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    acc = tl.dot(a, b, acc)                        # ds_read_b128 + v_mfma_f32_16x16x32
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
c = acc.to(c_ptr.dtype.element_ty)                 # with OPTIMIZE_EPILOGUE=1, no convert_layout
```

Compiled with `matrix_instr_nonkdim=16`, `num_stages=2` and `OPTIMIZE_EPILOGUE=1`, the inner loop
should contain: several `global_load_dwordx4`, `ds_read_b128`, a dense run of `v_mfma_f32_16x16x32`,
**no `v_accvgpr_*` moves**, and `.private_segment_fixed_size: 0`.

Anything else is a finding. Take it to `triton_isa_check.md`.

## Failure modes
| What you see | What it means | Where to go |
|---|---|---|
| `num_stages` sweep is completely flat | the pipeliner never fired on this loop | `lever_loop_form.md` — restructure the loop |
| `global_load_dword` plus `v_cmp` on masked loads | buffer ops are off in this build | enable `knobs.amd.use_buffer_ops` |
| `.private_segment_fixed_size` nonzero | spilling to scratch | cut the tile or the stage count before tuning anything else |
| Runs of `v_accvgpr_*` in the hot loop | accumulator pressure at this tile size | shrink the tile; also check LLVM #131954 |
| Two `ds_read_b64` where you expected `ds_read_b128` | on gfx942 this was `kpack`; on gfx950 it means something else | look at the tile shape and the swizzle |
| Fast standalone, unchanged end to end | the kernel is not on the dispatch path | verify what actually ran, not what you compiled |

## Sources
- AMD backend `HIPOptions`, the stream-pipeliner passes, and the `knobs.amd.*` surface:
  https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- MFMA shapes, AGPR accumulators, block-scaled f8f6f4 (Matrix Core programming, CDNA3/CDNA4):
  https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- TTIR → TTGIR → TritonAMDGPU → AMDGCN, and where `convert_layout` comes from:
  https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45
- `OPTIMIZE_EPILOGUE`, `ds_read_b128`, `global_load_dwordx4`:
  https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
