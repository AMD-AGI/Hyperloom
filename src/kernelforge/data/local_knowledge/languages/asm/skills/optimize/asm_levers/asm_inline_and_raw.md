---
title: asm — inline asm, raw .s, s_waitcnt overlap, SMFMAC
kind: language
lever: asm_inline_and_raw
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
  - https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://arxiv.org/abs/2511.08083
  - https://github.com/ROCm/aiter
---

# Inline asm and raw `.s`

Sub-levels 2 and 3 (`asm_decision.md`). This is where aiter's fastest paths live: a hand-scheduled
interleave of `buffer_load` / `ds_read` / `v_mfma` that out-schedules LLVM for one specific hot kernel.

## Route here when
- Disassembly **proves** the compiler's interleave is bad, and the kernel is hot enough to justify it.
- You need a latency probe or a specific encoding the compiler will not emit.
- You are implementing SMFMAC (structured sparsity).

**Never put MFMA itself in inline asm.** `SchedGroupMask` only recognizes *intrinsic* MFMA; hand-writing
`v_mfma` in an `asm volatile` block blinds the software pipeliner and you lose more than you gain. Keep
MFMA as the builtin and hand-schedule only the surrounding loads.

## `s_waitcnt` — the mechanism the whole level rests on

CDNA memory is asynchronous; hardware tracks *outstanding* operations in counters. `s_waitcnt` blocks
until a counter drops to a given value:

> **`s_waitcnt <counter>(N)` means "wait until ≤ N outstanding" — NOT "wait N instructions."**

| Counter | Tracks | Typical use |
|---|---|---|
| **`vmcnt(N)`** | VMEM (`buffer_load` / `global_load`) | wait until ≤ N global loads pending |
| **`lgkmcnt(N)`** | LDS (`ds_*`) + scalar (`s_load`) + messages | wait until ≤ N LDS/scalar pending |
| **`q_waitcnt`** (gfx950) | async load queue | direct-to-LDS overlap |
| `expcnt(N)` | exports | graphics; rare in compute |

Omitted fields default to **max** ("don't wait"). `s_waitcnt vmcnt(0) lgkmcnt(0)` is a full fence — and
usually a bug in a hot loop.

## The relaxed-count pipelining pattern

The canonical trick: overlap the *tail* of the LDS reads with the MFMA on the *previous* fragment.

```asm
    ds_read_b128  v[8:11],  v20          ; A fragment for the NEXT mfma
    ds_read_b128  v[12:15], v24          ; B fragment for the NEXT mfma
    s_waitcnt     lgkmcnt(1)             ; proceed once all-but-ONE ds_read has returned
    v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[4:7], a[0:3]   ; compute on the PREVIOUS fragment
    s_waitcnt     lgkmcnt(0)             ; last ds_read in; rotate buffers
```

This is what the LLVM scheduler emits for a well-formed MFMA loop, and what CK/ck_tile v3/v4 pipelines
generate for you. If you are writing it by hand, you are reproducing the compiler — make sure you
verified it was not already doing this.

## Pinning the schedule

- **`s_setprio`** raises wave priority during the compute burst so the MFMA issuer is not starved.
- **`__builtin_amdgcn_sched_barrier(mask)`** and
  **`__builtin_amdgcn_sched_group_barrier(mask, size, sync_id)`** pin the
  `buffer_load` / `ds_read` / `v_mfma` interleave so the compiler cannot reorder it.

The `SchedGroupMask::MFMA` bit is how the pipeliner identifies matrix ops — which is exactly why
hand-written MFMA disappears from its view.

## Inline asm hygiene — three rules

```cpp
asm volatile(
  "global_load_dwordx4 %0, %2, off\n"
  "global_load_dwordx4 %1, %3, off\n"
  "s_waitcnt vmcnt(0)\n"
  : "=&v"(v0), "=&v"(v1) : "v"(ptr0), "v"(ptr1) : "memory");
```

1. **One block for an ordered sequence.** Multiple `volatile` blocks can collapse into the same
   registers or get reordered relative to each other (HIP #3333).
2. **Early-clobber `"=&v"`** when an output register must not alias an input. Without it you get the
   classic "the first load clobbers `v[0:1]`, later loads break" bug.
3. **`"memory"` clobber + `volatile`** around timing or sync code. Without them `-O2` reorders or
   deletes it outright — an `s_memtime` latency probe silently returns nonsense.

## Memory dataflow on gfx950

```
buffer_load (global→VGPR) → ds_write (VGPR→LDS) → ds_read (LDS→VGPR) → v_mfma
```

**Direct-to-LDS** collapses the first two into one operation:

```asm
buffer_load_dwordx4 ... lds        ; descriptor form
global_load_lds_dwordx4 ...        ; 16 B/lane straight into LDS
```

This skips the VGPR stage, the `ds_write`, and the copy index math — a major occupancy and register
win. **gfx950 accepts 1/2/4/12/16 DWORD (up to 128 b/lane)**, 4× the width of the previous generation,
and adds **read-with-transpose `ds` loads** so the B operand needs no transpose pass.

HipKittens uses `buffer_load_dwordx4` for BF16/FP8 and `dwordx3` for FP6, synchronized with
`vmcnt` / `s_waitcnt` / `q_waitcnt`.

## SMFMAC — structured sparsity

`v_smfmac_*` implements **structured sparsity: 2 non-zeros per 4 elements** in A (B stays dense), for
roughly 2× throughput. Only use it with genuinely pruned weights.

The C/Src2 operand is replaced by a **matrix of compression indices**: for 8-bit inputs only 16 bits per
lane are needed.

- `CBSZ == 0` → `ABID` selects the top or bottom 16-bit half of the index register.
- `CBSZ != 0` → `ABID` ignored, low 16 bits used.

**This re-purposes `cbsz`/`abid`**, so they cannot also broadcast A — do not carry plain-MFMA habits
over. gfx950 adds `v_smfmac_f32_16x16x128_*` fp8 variants.

```bash
./matrix_calculator.py --architecture cdna4 --instruction v_smfmac_f32_16x16x64_f16 \
                       --compression --register-layout
# e.g. K[2][31] = v0{50}.[7:4]  -> compression bits in lane 50, bits 4..5 of the Src2 VGPR
```

## What to change

- **Hand-schedule the K-loop** only when disassembly shows the compiler's interleave is provably bad.
- **`s_waitcnt lgkmcnt(1)` before the MFMA** — the canonical prefetch-overlap pattern.
- **Direct-to-LDS at 12/16 DWORD** to skip the VGPR staging stage entirely.
- **128-bit everywhere**: `buffer_load_dwordx4`, `ds_read_b128` / `ds_write_b128`. Up to 4 adjacent
  `dwordx4` coalesce into one fabric transaction.
- **`s_setprio` + `sched_group_barrier`** to hold the interleave you built.

## Verify

```bash
hipcc --offload-arch=gfx950 -O3 --save-temps kern.cpp -o kern
grep -E 'v_mfma|v_smfmac|s_waitcnt|s_setprio|accvgpr|ds_read|buffer_load|scratch_' kern-*.s
```

| Check | Pass |
|---|---|
| `s_waitcnt lgkmcnt(1)` sits before `v_mfma` | the overlap actually landed |
| Load widths | `dwordx4` / `b128`, not `dword` / `b32` |
| Direct-to-LDS | 12/16-DWORD form where you staged tiles |
| `scratch_` | **none** in the hot loop |
| MFMA | still the intrinsic, not inside an asm block |

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Pipelining vanished after hand-tuning | MFMA moved into inline asm | keep MFMA as the intrinsic |
| Data race, or a stall you cannot explain | `s_waitcnt N` off-by-one — it is "≤ N remaining" | recount outstanding ops |
| Loads return the wrong registers | missing early-clobber `"=&v"` | add it; use one asm block |
| Timing probe returns nonsense | missing `"memory"` clobber / `volatile` | add both |
| SMFMAC gives garbage | treated `cbsz`/`abid` as broadcast flags | they are index selectors here |
| Direct-to-LDS emitting 4-DWORD | inherited from a 32 b/lane part | request 12/16 DWORD |

## Sources
- AMD CDNA4 ISA (waitcnt semantics, SMFMAC, `buffer_load ... lds`): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- LLVM gfx9 `s_waitcnt` semantics ("wait for remaining"): https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
- LLVM AMDGPU backend guide (buffer descriptors, sched builtins): https://llvm.org/docs/AMDGPUUsage.html
- amd_matrix_instruction_calculator (SMFMAC compression-index layout, `--compression`): https://github.com/ROCm/amd_matrix_instruction_calculator
- HipKittens (arXiv 2511.08083 — direct-to-LDS `dwordx4`/`dwordx3`, `vmcnt`/`q_waitcnt`, aiter raw-asm baselines): https://arxiv.org/abs/2511.08083
- HIP #3333 (inline GCN asm multi-load register clobber): https://github.com/ROCm/HIP/issues/3333
- ROCm/aiter (raw-asm fastest paths): https://github.com/ROCm/aiter
