---
title: asm — the ten traps, by symptom
kind: language
lever: asm_traps
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://github.com/ROCm/HIP/issues/3333
  - https://github.com/llvm/llvm-project/issues/131954
  - https://github.com/iree-org/iree/issues/23765
  - https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
  - https://arxiv.org/abs/2511.08083
---

# ASM traps

Indexed **by symptom** so you can jump straight from what you observed. Almost all of these are
diagnosable from the disassembly — compile, grep, and read before theorizing.

## Symptom → trap

| What you observe | Trap | § |
|---|---|---|
| Silently wrong numbers, no NaN | wrong fragment placement · FP8 encoding mismatch | §1 §2 |
| Loads return the wrong registers | inline-asm clobber | §3 |
| Timing probe returns nonsense | missing `"memory"` / `volatile` | §3 |
| Data race, or an unexplained stall | `s_waitcnt` off-by-one | §4 |
| Loads never overlap compute | MFMA hidden in inline asm | §5 |
| TFLOP/s plateaus or regresses as the tile grows | spurious accumulator spills | §6 |
| `ds_read` stalls starve the matrix core | LDS bank conflicts on **64** banks | §7 |
| Slower than expected at 32×32 | power/clock, not software efficiency | §8 |
| Direct-to-LDS not helping | assuming widths that are not there | §9 |
| Stuck at ~80% of peak, otherwise well-tuned | wave specialization | §10 |

---

### §1 Wrong MFMA fragment placement
Lanes pack A/B/C in a fixed pattern with **no guaranteed element order**. Guessing gives a silent wrong
answer — no error, no NaN.
**Fix:** `matrix_calculator.py --architecture cdna4 --instruction <op> --register-layout --A-matrix`.
Never guess lane order. → `asm_register_budget.md`

### §2 FP8 encoding mismatch
**gfx950 FP8 is OCP** (E4M3FN bias 7, max ±448; E5M2 with inf). Earlier CDNA parts used **FNUZ**
(bias 8, max ±240, no inf). A mismatched dequant scale is silent garbage.
**Fix:** convert the checkpoint, never bit-copy. Use `__amd_fp8_*` (`hip_ext_ocp.h`), not `__hip_fp8_*`.
Also: **TF32 was removed** — no path will be emitted. → `hardware/mi350_dtypes.md`

### §3 Inline-asm clobber bugs (HIP #3333)
Multiple `volatile` blocks collapse into the same registers or get reordered.
**Three rules:**
- **One asm block** for an ordered sequence.
- **Early-clobber `"=&v"`** when an output must not alias an input — otherwise "the first load clobbers
  `v[0:1]`, later loads break."
- **`"memory"` clobber + `volatile`** around timing/sync code, or `-O2` reorders or deletes it (an
  `s_memtime` probe silently returns nonsense).

```cpp
asm volatile(
  "global_load_dwordx4 %0, %2, off\n"
  "global_load_dwordx4 %1, %3, off\n"
  "s_waitcnt vmcnt(0)\n"
  : "=&v"(v0), "=&v"(v1) : "v"(ptr0), "v"(ptr1) : "memory");
```

### §4 `s_waitcnt` off-by-one
`vmcnt(N)` / `lgkmcnt(N)` mean **"wait until ≤ N remaining"**, not "wait N instructions". Too low and
you stall; too high and you have a data race.
**Fix:** count outstanding ops explicitly. → `asm_inline_and_raw.md`

### §5 MFMA inside inline asm kills pipelining
`SchedGroupMask` only recognizes **intrinsic** MFMA. Hand-writing `v_mfma` in `asm volatile` blinds the
software pipeliner — you lose the load/compute overlap you were trying to hand-build.
**Fix:** keep MFMA as the builtin; hand-schedule only the surrounding `buffer_load` / `ds_read`, and
pin the order with `sched_group_barrier`.

### §6 Spurious accumulator spills (LLVM #131954)
At large tiles the compiler inserts unnecessary `v_accvgpr_read/write` and/or `scratch_` spills.
**Signature: TFLOP/s plateaus or *regresses* as you grow the tile** — the one symptom that reliably
identifies this.
**Fix:** `grep -cE 'v_accvgpr|scratch_' kern.s`; shrink the tile, or let the accumulator stay in VGPR.
→ `asm_register_budget.md`

### §7 LDS bank conflicts — **64 banks on gfx950**
The `ds_read` feeding MFMA must avoid lane→bank collisions. Bank index is **`(byte_addr / 4) mod 64`**.
**Any swizzle or padding inherited from a 32-bank part is unverified here — re-derive it.**
**Fix:** prefer **XOR swizzle** of the LDS *write* address (no extra LDS) over **padding** (costs LDS,
lowers occupancy; the AMD tuning guide warns about this). See iree #23765 for the
direct-to-LDS / XOR-swizzle / pad tradeoff. → `hardware/mi350_lds.md`

### §8 Defaulting to 32×32 MFMA
Two independent reasons it loses: it carries **16 C-registers/lane vs 16×16's 4**, and it **draws more
power → the part clocks lower → lower max-achievable FLOPs** (ROCm Max-Achievable-FLOPs Part 2).
**Fix:** default 16×16; test 32×32 only for a specific large square shape.

### §9 Direct-to-LDS assumptions
gfx950 accepts **1/2/4/12/16 DWORD** (up to 128 b/lane) and adds read-with-transpose `ds` loads. Two
opposite mistakes: assuming the wide path exists on older parts, or emitting the narrow 4-DWORD form
here because that is what the old code did.
**Fix:** check the disassembly for the 12/16-DWORD form. → `asm_inline_and_raw.md`

### §10 Wave specialization underperforms on CDNA
Producer/consumer wave splits — the NVIDIA model — **do not work here.** AMD's static register
allocation means producer waves hold registers without computing, and the kernel tops out around
**~80% of peak BF16 GEMM** (HipKittens, MI355X). There is no warp-group specialization escape hatch.
**Fix:** use a symmetric all-waves-compute schedule — **8-wave ping-pong** or **4-wave interleave**.
Reference: HIP 8-wave ping-pong 3204 TFLOPS, HK 4-wave interleave 3327 TFLOPS (MI355X, ROCm 7.1.0,
FP8, M=N=K=8192).

---

## The standard diagnostic pass

```bash
amdclang++ -x hip --offload-arch=gfx950 -O3 -S kern.cpp -o kern.s
grep -E 'v_mfma|v_smfmac|s_waitcnt|accvgpr|ds_read|buffer_load|scratch_' kern.s
```

| Also | For |
|---|---|
| `rocprof-compute` LDS panel — bank-conflict rate over **64** banks | §7 |
| `hipcc -Rpass-analysis=kernel-resource-usage` | §6 |
| `matrix_calculator.py --register-layout` | §1 |
| `../intellikit/instructions/nop_hazard_summary.md` | NOP/wait-state hazards (silent corruption) |

**IntelliKit is the deeper reference** for anything instruction-level: measured cycle counts and hazard
rules on real MI355X silicon that are not in the public ISA docs. Its methodology — *disassemble a
working `.co` → round-trip validate bit-identical → one targeted change → profile* — is the right loop
for every trap on this page.

## Sources
- HIP #3333 (inline GCN asm multi-load register clobber): https://github.com/ROCm/HIP/issues/3333
- LLVM #131954 (large MFMA tiles → spurious `v_accvgpr` / spills): https://github.com/llvm/llvm-project/issues/131954
- iree #23765 (direct-to-LDS + XOR-swizzle vs LDS-pad bank-conflict tradeoff): https://github.com/iree-org/iree/issues/23765
- ROCm Blog — Max-Achievable FLOPs Part 2 (16×16 vs 32×32 power/clock): https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
- MI300X workload optimization (LDS padding vs occupancy): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- HipKittens (arXiv 2511.08083 — wave specialization underperforms on CDNA; ping-pong / interleave numbers): https://arxiv.org/abs/2511.08083
