---
title: Triton on AMD — the traps, by symptom
kind: language
lever: triton_traps
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/sgl-project/sglang/pull/2601
  - https://arxiv.org/abs/2511.08083
---

# Triton traps

Indexed **by symptom**. Several of these fail silently — the kernel compiles, runs, and is simply slow
or wrong.

## Symptom → trap

| What you observe | Trap | § |
|---|---|---|
| 3–5× slower than expected after a port | `num_warps=8` carried from NVIDIA → spill | §1 |
| Knob set, nothing changed | AMD knobs outside `triton.Config` | §2 |
| **`num_stages` sweep is flat** | the loop is never pipelined | §3 |
| `Unsupported conversion 'f8E4M3FN'` | wrong fp8 dialect for the target | §4 |
| Results silently ~2× off | wrong fp8 dialect, other direction | §4 |
| ISA shows `global_load_dword` + `v_cmp` | buffer ops not enabled | §5 |
| Occupancy 1, or compile failure on a big tile | LDS budget | §6 |
| Slow GEMM at specific N/K only | 512 B leading-dimension stride | §7 |
| `ds_read_b32` in the ISA | LDS layout / `BLOCK_K` too small | §8 |
| Backend warns about `kpack` | gfx942 config on gfx950 | §8 |
| Faster in isolation, no e2e change | not on the dispatch path | §9 |
| Beat by hipBLASLt on plain GEMM | that is expected | §10 |
| First-call latency in serving | autotune in the hot path | §11 |

---

### §1 `num_warps=8` from NVIDIA
Eight warps → two waves share a SIMD → ~256 VGPR each → **spill to scratch (HBM)** → 3–5× slower.
**Fix:** start at `num_warps=4`; go to 8 only if the kernel is VGPR-light and occupancy-bound. Confirm
`.private_segment_fixed_size: 0`.

### §2 AMD knobs set as Python variables
`matrix_instr_nonkdim`, `kpack`, `waves_per_eu` only take effect **inside `triton.Config({...})`**
(they map to `HIPOptions` fields). Set anywhere else, they are silently ignored.
**Fix:** put them in the Config kwargs dict. Verify in the ISA that the shape changed.

### §3 A data-dependent loop forfeits pipelining **and** async copy
The stream pipeliner schedules `for`/`tl.range` loops whose bound is loop-invariant and whose staged
loads are affine in the induction variable. `while blk < tl.load(counts + pid)`, or
`page = tl.load(bt + blk)` feeding `tl.load(kv + page*stride + offs)` in the same iteration, gives you
**neither — with no diagnostic.**

**The tell is a `num_stages` sweep that is flat inside the noise band.** That is a diagnostic, not a
result: it says pipelining never ran.

**Fix:** bound the walk by a shape-static range and move the selection into a mask. This can be a net
win *even though it visits more blocks* — a measured case ran **~2× faster while visiting 2–4× more
blocks**. → `common_methodology/optimization/lever_loop_form.md`

### §4 fp8 dialect mismatch
**gfx950 = OCP. gfx942 = FNUZ.** The exponent bias differs by 1, so:
- OCP `float8_e4m3fn` into `tl.dot` **on gfx942** → `Unsupported conversion 'f8E4M3FN'` (loud).
- The reverse — FNUZ values read as OCP — is a **silent ~2× error**, not a crash.

SGLang/vLLM call `normalize_e4m3fn_to_e4m3fnuz` before the matmul **when targeting gfx942**
(sglang PR #2601). Doing that on gfx950 corrupts your data.
**Fix:** check which dialect the checkpoint stored, and which the target wants.

Also: **`tf32` is CDNA3-only and removed on CDNA4.** Valid AMD `input_precision` is `"ieee"`;
NVIDIA's `"tf32x3"` is not an AMD path.

### §5 Buffer loads are not the default
Masked GEMM/attention tails want `buffer_load_dwordx4` — a 128-bit descriptor with **hardware bounds
checking**, so OOB lanes return 0 and there is no predication branch. **Many builds do not emit it by
default.**
**Fix:** set `knobs.amd.use_buffer_ops`. If the ISA shows `global_load_dword` with a `v_cmp` around
masked loads, you are on the slow path.

### §6 LDS budget
gfx950 has **160 KiB/CU** (gfx942 had 64 KiB). Two opposite errors: sizing tiles against 64 KiB and
leaving performance unused, or porting an H100 kernel (228 KB) and overflowing.
**Fix:** prune the config space against **160 KiB**; if occupancy drops to 1 wg/CU, shrink the tile or
`num_stages`, and set `OPTIMIZE_EPILOGUE=1`.

### §7 512 B leading-dimension stride (TN GEMM)
A leading dimension that is an exact multiple of 512 B can collide in the L2 tag RAM. Symptom: slow at
specific N/K while neighbours are fine.
**Fix:** pad `lda`/`ldb` by 128 when `K % 256 == 0`.

### §8 Narrow LDS reads / stale `kpack`
On gfx950 you should get `ds_read_b128` **without** `kpack` — it is deprecated and forced to 1 there
(the backend warns if you set 2). `ds_read_b32` in the ISA means `BLOCK_K` is too small (< 64) or the
swizzle did not apply.
**Fix:** bump `BLOCK_K`; drop `kpack` from gfx950 config spaces.

### §9 Not on the dispatch path
On sglang the dense GEMM path is **aiter**, not raw torch dispatch. An authored Triton GEMM must be
wired through the aiter seam (`aiter.tuned_gemm` `triton` libtype) or a call-site rebind, **then
e2e-gated**: only keep it if `pct_gpu_time × speedup` moves e2e beyond the noise band.

Validation note: an authored Triton GEMM measured **0.99–1.47× isolated** and did **not** beat the
aiter environment at e2e (2026-06).

Related: **the experimental Triton GEMM stub in aiter is not a real implementation** —
`aiter.ops.flydsl`/`tuned_gemm` treat `triton` as a libtype but the entry is a thin shim. Read "Triton
GEMM in aiter" as *author needed*, not *available*.

### §10 Expecting to beat tuned hipBLASLt/aiter on plain dense GEMM
You will not, as a rule. **The honest win is fusion** (epilogue/attention) **or skinny split-K decode.**
HipKittens (arXiv 2511.08083) shows compiler backends including Triton under-perform hand-tuned
asm/CK on CDNA GEMM and attention; a hand-written HIP/CK/asm kernel can be 1.2–2.4× faster in some
regimes.
**Fix:** pick the tool for the job — `triton_amd_delta.md` has the fit table.

### §11 Autotune in the serving hot path
Adds first-call latency and is non-deterministic.
**Fix:** bake a per-shape table (`triton_knob_space.md`, "Baking the winner").

---

## Other things that bite

- **`warpSize == 32` hardcoded** in grid/occupancy math — it is **64**.
- **A reduced dim < 64 wastes lanes** in `tl.sum` / `tl.max` wave reduces. Round the reduced dimension
  to a power of 2 ≥ 64.
- **`num_stages=3/4` for a single GEMM** — pipelines worse than 1–2 on the AMD stream pipeliner.
- **Grid sized for 304 CUs** — gfx950 has **256**.

## The one diagnostic pass

```bash
AMDGCN_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python k.py 2> dump.txt
```
Want: `global_load_dwordx4` / `buffer_load_dwordx4`, `ds_*_b128`, dense `v_mfma_*`, **no `v_accvgpr_*`
in the loop**, `.private_segment_fixed_size: 0`. → `triton_isa_check.md`

## Sources
- Optimizing Triton kernels (tuning pitfalls, `OPTIMIZE_EPILOGUE`, ISA): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- FNUZ fp8 normalization (sglang): https://github.com/sgl-project/sglang/pull/2601
- Honest compiler-vs-asm limits: HipKittens, https://arxiv.org/abs/2511.08083
- aiter `tuned_gemm` libtypes (the triton stub): ROCm/aiter:aiter/tuned_gemm.py
