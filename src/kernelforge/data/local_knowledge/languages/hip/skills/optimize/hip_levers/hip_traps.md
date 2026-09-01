---
title: HIP — CUDA→HIP traps, by symptom, and the ISA checklist
kind: language
lever: hip_traps
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://github.com/ROCm/HIP/issues/3667
  - https://github.com/llvm/llvm-project/issues/131954
---

# HIP traps

Indexed **by symptom**. Most of these are a CUDA assumption that compiles fine and is wrong or slow.

## Symptom → trap

| What you observe | Trap | § |
|---|---|---|
| Wrong reduction results, or a static-assert on a mask | `warpSize` / mask assumed 32 | §1 |
| Half the machine idle, everything "works" | block size not a multiple of 64 | §2 |
| Truncated ballot results | `__ballot` stored in `unsigned` | §1 |
| Launch failure, or occupancy collapses to 1 | LDS budget | §3 |
| Serialized LDS, `ds_*` stalls | no padding/swizzle, or a 32-bank one | §4 |
| 3–5× slower than expected | scratch spill | §5 |
| Low bandwidth on a streaming kernel | scalar global loads | §6 |
| Reduction is the bottleneck | fp atomics not enabled | §7 |
| **TFLOP/s regresses as the tile grows** | `v_accvgpr_*` in the MFMA loop | §8 |
| FP8 results wrong, or no lowering | wrong fp8 dialect | §9 |
| Cross-lane slower than expected | mask has holes | §1 |
| `__shfl` on half fails | unsupported | §1 |

---

### §1 wave64 assumptions
`warpSize == 64`. Consequences that all trace to this one fact:
- Masks must be **`unsigned long long`** with **`__popcll`** — a 32-bit mask static-asserts in
  `amd_warp_sync_functions.h`, and storing `__ballot` in `unsigned` truncates.
- Reduction loops start at `warpSize/2 = 32`, not 16.
- **Contiguous, hole-free masks are faster** — `0xFF` beats `0xFB`; reduce over `0..N-1`.
- **Half-float `__shfl` is unsupported** — shuffle as int/float and repack.
- Cross-lane intrinsics carry **no memory barrier** — add `__syncthreads()`/fences for side effects.

> 32-lane CUDA code **runs correctly and uses half the machine.** There is no error to catch.

### §2 Block size not a multiple of 64
A 32-thread block is half a wave. Use 64/128/256; 256 (= 4 waves) is the common sweet spot.
Grid target: **≥1024 workgroups** across **256 CUs** — query the CU count, do not hardcode 304.

### §3 LDS budget
gfx950 has **160 KiB/CU**. Two opposite errors: porting from H100 (228 KB) and overflowing, or sizing
against MI300X's 64 KB and leaving 2.5× unused.
**Fix:** `occ_lds = floor(163840 / lds_bytes)`; remember the **320-DWORD** allocation granule.
→ `hardware/mi350_execution.md`

### §4 Bank conflicts — **64 banks on gfx950**
The `ds_read` feeding MFMA must avoid lane→bank collisions. Bank = `(byte_addr/4) mod 64`.
**Any padding or XOR swizzle inherited from a 32-bank part is unverified here.**
**Fix:** prefer XOR swizzle (no extra LDS) over padding (costs LDS, lowers occupancy); keep 16-byte
alignment so `ds_read_b128` survives. Direct-to-LDS **without** a swizzle measured 201 M conflicts and
−28% TFLOPS in one study. → `hip_lds_staging.md`

### §5 `__launch_bounds__` too tight → scratch spill
Spilling turns a register access into HBM traffic inside the inner loop: **3–5× slower**.
**Fix:** check `.private_segment_fixed_size == 0`. Back off the bound; `minWavesPerEU=2` → VGPR ≤ 256,
`=4` → ≤ 128. **2 waves/SIMD with no spills beats 3 that spill.**

### §6 Scalar global loads
**Fix:** `float4` / `int4` types plus `__restrict__` on pointers so the compiler can prove contiguity
and emit `global_load_dwordx4`. Verify in the ISA — source intent is not enough.

### §7 fp atomics not enabled
**Fix:** `-munsafe-fp-atomics` gives hardware `global_atomic_add_f32`. Material for split-K and
reductions.

### §8 `v_accvgpr_*` in the MFMA loop (LLVM #131954)
At large tiles the compiler inserts accumulator moves and spills, and performance falls back to
small-tile levels. **The signature is TFLOP/s that plateaus or regresses as you grow the tile** — the
one symptom that reliably identifies this.
**Fix:** keep the accumulator in a stable `__attribute__((vector_size))` variable across iterations so
it stays in AGPRs (the "tied accumulator" pattern CK relies on). Inline asm alone does not give you
this. Grep `accvgpr` in the `.s`.

### §9 fp8 dialect
**gfx950 is OCP** (E4M3FN, bias 7, max ±448). gfx942 was **FNUZ** (bias 8, max ±240). Feeding FNUZ
bits to an OCP MFMA is **silently wrong**; the reverse fails to lower.
**Fix:** convert, never bit-copy. Use `__amd_fp8_*` (`hip_ext_ocp.h`) on gfx950.
Also: **TF32 was removed** on CDNA4. → `hardware/mi350_dtypes.md`

---

## Predict occupancy before you measure

```
occ_vgpr    = floor(512 / round_up_16(vgpr_used))            # waves/SIMD
occ_lds     = floor(163840 / lds_bytes_per_workgroup)        # workgroups/CU
occ (wg/CU) = min(floor(occ_vgpr * 4 / num_waves), occ_lds)  # 4 SIMD/CU
```

`__launch_bounds__(maxTPB, minWavesPerEU)` is the lever. Going past what the kernel needs forces
spills — **verify, do not guess.** On gfx950 the LDS term rarely binds; registers usually do.

## The ISA checklist

```bash
amdclang++ -x hip --offload-arch=gfx950 -O3 --save-temps kern.cpp -o kern
grep -E 'global_load|ds_read|ds_write|v_mfma|accvgpr|scratch_|s_waitcnt' kern-*.s
hipcc --offload-arch=gfx950 -Rpass-analysis=kernel-resource-usage ...
```

| Look for | Good | Bad → retune |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| LDS access | `ds_read_b128` / `ds_write_b128` | `ds_read_b32` |
| MFMA | dense `v_mfma_f32_16x16x32` | sparse, with gaps = starved core |
| Accumulator | stays in `a[0:n]` (AGPR) | `v_accvgpr_read/write` **in the loop** |
| **Scratch** | **`.private_segment_fixed_size: 0`** | nonzero → spilling to HBM |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap |

`-Rpass-analysis=kernel-resource-usage` prints `.vgpr_count`, `.sgpr_count`,
`.group_segment_fixed_size` (LDS) and `.private_segment_fixed_size` (scratch) at a glance.

## When not to write raw HIP

If you are reaching for `__builtin_amdgcn_mfma_*` + `sched_group_barrier` + double-buffering by hand,
check first whether **rocWMMA**, **ck_tile / Composable Kernel**, **HipKittens** (`hipkittens.md`) or
**FlyDSL** already expresses it. They encode the tied-accumulator and pipeline patterns correctly and
sidestep §8 entirely. Raw HIP is for fusions those cannot express, or when you must own the exact ISA.

## Sources
- `warpSize` / masks / `__shfl` / half-float: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- `__ballot` 64-bit return and mask requirements: https://github.com/ROCm/HIP/issues/3667
- MFMA + pipelining AGPR spill / tied accumulator: https://github.com/llvm/llvm-project/issues/131954
- VGPR/LDS limits, grid sizing, occupancy: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Direct-to-LDS swizzle requirement (201 M conflicts, −28%): https://github.com/iree-org/iree/issues/23765
