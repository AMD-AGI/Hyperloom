---
title: AMDGCN ABI — kernel descriptor, AQL dispatch, register conventions
kind: api_reference
gens: [gfx942, gfx950]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/amd_clr.html
---

# AMDGCN ABI & kernel descriptor

Where a hand-written `.co` / `.s` kernel meets the runtime: the code-object ABI (kernel descriptor,
argument metadata, register/EXEC/SGPR conventions) and the HSA AQL dispatch packet. This is the "how a
raw kernel is launched and reads its args" API — the layer above the instruction stream in
[cdna4_isa_reference.md](cdna4_isa_reference.md).

## The measured ground-truth lives in IntelliKit
The load-bearing ABI docs are in the vendored IntelliKit reference (MI355X-measured):
- **`../skills/optimize/asm_levers/intellikit/instructions/kernel_descriptor.md`** — `accum_offset`,
  `.args` metadata, AGPR aliasing — the launch-failure bugs in one place.
- **`../skills/optimize/asm_levers/intellikit/instructions/hsa_aql_dispatch.md`** — the AQL dispatch
  packet (grid/workgroup dims, kernarg pointer, private/group segment sizes).
- **`../skills/optimize/asm_levers/intellikit/instructions/inter_wg_barrier.md`** — cross-workgroup sync.
- ISA/ABI register model (VGPR/AGPR/SGPR/EXEC/SCC/M0/HW_ID) — see
  [cdna4_isa_reference.md](cdna4_isa_reference.md) and its `cdna4-isa/Ch2_program_org.txt`,
  `Ch3_kernel_state.txt`.

## Key ABI facts (CDNA)
- **Kernel descriptor** (64 B) declares `.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size` (LDS),
  `.private_segment_fixed_size` (scratch — must be 0 for a spill-free kernel), `accum_offset` (AGPR base),
  and kernarg size. A wrong `accum_offset` / AGPR aliasing → launch failure or garbage.
- **Register file**: unified VGPR+AGPR physical file; with one wave/SIMD it splits **256 VGPR + 256 AGPR**.
  MFMA `ACC`/`ACC_CD` bits select VGPR(0) vs AGPR(1) per matrix operand.
- **Scalar state**: 104 SGPRs/wave (VCC aliases 106–107); `EXEC` 64-bit lane mask; `SCC`; `M0` (LDS
  offset / GPR-indexing base).
- **Async memory**: correctness is via `s_waitcnt vmcnt/lgkmcnt` (≤N outstanding, NOT N instructions);
  CDNA4 adds `q_waitcnt` / `s_wait_asynccnt` for the async-load queue.

## Toolchain (disassemble / inspect a .co)
```bash
# emit ISA from source
/opt/rocm/bin/amdclang++ -x hip --offload-device-only --offload-arch=gfx950 -O3 -S kern.cpp -o kern.s
# disassemble a compiled code object
roc-obj-ls kernel.hsaco
llvm-objdump -d --arch=amdgcn kernel.hsaco | less
# round-trip validate bit-identical before editing (IntelliKit METHODOLOGY)
```

## Sources
- AMDGPU code-object ABI (kernel descriptor, kernarg, register usage attrs, s_waitcnt): https://llvm.org/docs/AMDGPUUsage.html
- Measured kernel-descriptor / AQL dispatch ground truth: `../skills/optimize/asm_levers/intellikit/instructions/`
- CDNA ISA program organization / kernel state: [cdna4_isa_reference.md](cdna4_isa_reference.md)
