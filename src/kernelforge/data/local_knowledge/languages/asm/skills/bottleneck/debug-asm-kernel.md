---
name: debug-asm-kernel
description: >
  Diagnose hand-written AMDGCN assembly (.s / inline asm / HSACO) on CDNA3/CDNA4:
  NOP hazards and silent corruption, s_waitcnt vmcnt/lgkmcnt FIFO ordering, AGPR
  aliasing / accum_offset launch failures, inline-asm defeating SchedGroupMask,
  early-clobber/memory-clobber rules, VGPR/AGPR spill occupancy cliffs, and the
  disassemble→round-trip→one-change→profile workflow. Use when a hand-asm kernel is
  wrong, fails to launch, or underperforms. Usage: /debug-asm-kernel
allowed-tools: Read Bash Grep Glob
---

# Debug AMDGCN ASM kernel

Diagnostic workflow for the lowest level (raw `.s`, inline `asm volatile`, HSACO) on MI300X gfx942 /
MI350/MI355X gfx950. The per-instruction ground truth lives in
[../optimize/asm_levers/intellikit/](../optimize/asm_levers/intellikit/) (start with
`intellikit/instructions/nop_hazard_summary.md`, `intellikit/guides/debugging-playbook.md`); the
authoring model is in [../optimize/asm_levers/asm_decision.md](../optimize/asm_levers/asm_decision.md).

## Step 0: the IntelliKit workflow (never guess)
Disassemble a working `.co` → **round-trip validate bit-identical** → make **one targeted change** →
profile. This is the METHODOLOGY; skipping the round-trip is how silent corruption slips in.
```bash
/opt/rocm/bin/amdclang++ -x hip --offload-device-only --offload-arch=gfx942 -O3 -S kern.cpp -o kern.s
grep -E 'v_mfma|s_waitcnt|accvgpr|ds_read|buffer_load|scratch_|s_nop' kern.s
```

## Step 1: classify the symptom
| Symptom | Likely cause | Go to |
|---|---|---|
| Intermittent wrong bits, no error | missing NOP after a hazard instruction | §2 |
| Loads return stale/garbage data | `s_waitcnt` FIFO ordering wrong | §3 |
| Kernel won't launch / AGPR garbage | `accum_offset` / AGPR aliasing in kernel descriptor | §4 |
| MFMA loop slow despite hand-scheduling | inline-asm MFMA defeats SchedGroupMask | §5 |
| Compiler reorders / drops your asm | missing clobbers / not one block / not volatile | §5 |
| Perf collapses to small-tile class | VGPR/AGPR spill (`scratch_` in ISA) | §6 |

## 2. NOP hazards (the #1 silent-corruption source)
CDNA has instruction-pair hazards that require inserted `s_nop`s or specific spacing. The complete table
is `intellikit/instructions/nop_hazard_summary.md` — the single most important asm reference. A missing
NOP is not an error; it's wrong bits. Verify hazards for every MFMA→read, DPP, and cross-lane sequence.

## 3. s_waitcnt FIFO ordering
CDNA memory is asynchronous; `s_waitcnt vmcnt(N)`/`lgkmcnt(N)` mean "wait until ≤N **outstanding**", NOT
"wait N instructions". vmcnt tracks VMEM (`buffer_*`/`global_*`); lgkmcnt tracks LDS/SMEM (`ds_*`,
`s_load`). Reads issued before the matching count drains return stale data. CDNA4 adds `q_waitcnt` for the
async-load queue. Details: `intellikit/instructions/s_waitcnt.md`,
`intellikit/guides/memory-coherence-formats.md`.

## 4. Launch failures (kernel descriptor)
`accum_offset`, `.args` metadata, and AGPR aliasing are the classic launch-failure bugs — all in
`intellikit/instructions/kernel_descriptor.md`. If AGPRs alias VGPRs or `accum_offset` is wrong, the
kernel either fails to launch or produces AGPR garbage.

## 5. Inline asm rules & scheduling
- **Do NOT hand-write MFMA in inline asm** — it's not recognized by `SchedGroupMask`, defeating the SW
  pipeliner. Use `__builtin_amdgcn_mfma_*` + `sched_group_barrier` to *guide* the compiler
  (see hip `hip_levers/hip_builtins.md`).
- Inline `asm volatile` for ordered sequences needs **early-clobber `"=&v"`**, a `"memory"` clobber, and
  **one asm block** (splitting lets the compiler reorder). Missing these → dropped/reordered instructions.

## 6. Occupancy / spills
With one wave/SIMD the 512 registers split **256 VGPR + 256 AGPR**. Spilling past the budget emits
`scratch_` (HBM) and collapses occupancy — the #1 cause of MFMA kernels underperforming. Use
`intellikit/tools/scripts/vgpr_liveness.py` to find dead register windows and lift occupancy; budget per
`intellikit/guides/register-allocation.md`. 16×16 MFMA usually clocks higher than 32×32 (power).

## 7. When to stay at intrinsics instead
Hand-asm pays only when disassembly proves the compiler's schedule is suboptimal AND the kernel is hot
enough to amortize maintenance. For MFMA loops prefer intrinsics + `sched_group_barrier`; drop to raw
`.s` only for a proven peak micro-kernel (the AITER-style last-few-percent case).
