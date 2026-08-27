---
title: Triton on AMD — pitfalls & porting checklist
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [both]
status: competitive
updated: 2026-08-23
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/sgl-project/sglang/pull/2601
  - https://arxiv.org/abs/2511.08083
---

# Triton on AMD — pitfalls

## NVIDIA → AMD porting checklist
| Pitfall | Symptom | Fix |
|---|---|---|
| Hardcoded `warpSize==32` in grid/occupancy math | wrong block sizing, half-utilized waves | use **64**; recompute |
| `num_warps=8` carried from NVIDIA | VGPR spill, 3–5× slower | try `num_warps=4`, retune |
| `num_stages=3/4` for GEMM | worse pipeline than 2 | `num_stages=2` (single GEMM), `1` (FA) |
| `while` loop bounded by a `tl.load` (sparse/gather walk) | `num_stages` + async copy silently off; the sweep reads flat | shape-static `tl.range` + mask — see `loop_form_and_pipelining.md` |
| OCP fp8 `e4m3fn` into `tl.dot` on gfx942 | `Unsupported conversion 'f8E4M3FN'` | normalize to `e4m3fnuz` / `tl.float8e4b8` |
| Big tiles ignoring 64 KB LDS | occupancy→1 or compile fail | shrink tile / `num_stages` / `OPTIMIZE_EPILOGUE=1` |
| AMD knobs as Python vars not Config kwargs | silently ignored | put `matrix_instr_nonkdim`/`kpack`/`waves_per_eu` in `triton.Config({...})` |
| Leading dim multiple of 512 B (TN) | slow GEMM (Tagram hotspot) | pad `lda/ldb` by 128 when `K%256==0` |
| Narrow `ds_read_b32` in ISA | poor LDS layout | bump `kpack` (gfx942), change tile, check swizzle |
| `mfma_32x32` everywhere | perf left on table | prefer `matrix_instr_nonkdim=16` |
| `kpack=2` on gfx950 | warning, forced to 1 | only set `kpack=2` on gfx942 |

## AMD-specific anti-patterns
- **Buffer loads not default.** Masked GEMM/attention tails want `buffer_load_dwordx4` (HW bounds
  check, no predication branch), but many builds don't emit it by default — set
  `knobs.amd.use_buffer_ops`. If the ISA shows `global_load_dword` with a `v_cmp` predication around
  masked loads, you're on the slow path.
- **A data-dependent loop silently forfeits pipelining and async copy.** The stream pipeliner
  schedules `for`/`tl.range` loops whose bound is loop-invariant and whose staged loads are affine in
  the induction variable. `while blk < tl.load(counts + pid)`, or `page = tl.load(bt + blk)` feeding
  `tl.load(kv + page * stride + offs)` in the same iteration, gives you neither — with **no
  diagnostic**. The tell is a `num_stages` sweep that is flat inside the noise band. The fix is to
  bound the walk by a shape-static range and move the selection into a mask, which can be a net win
  even though it visits more blocks:
  `local_knowledge/common_methodology/optimization/loop_form_and_pipelining.md`.
- **FNUZ vs OCP fp8 is a 2× silent error**, not a crash, when you read the wrong dialect (exponent
  bias differs by 1). gfx942 = FNUZ; gfx950 = OCP. SGLang/vLLM use `normalize_e4m3fn_to_e4m3fnuz`
  before the matmul (sglang PR #2601). Always check which the checkpoint stored.
- **Don't expect to beat tuned hipBLASLt/aiter on plain dense GEMM.** The honest win is **fusion**
  (epilogue/attention) or **skinny split-K decode**. HipKittens (arXiv 2511.08083) shows compiler
  backends including Triton under-perform hand-tuned asm/CK on CDNA3/CDNA4 GEMM and attention; a
  hand-written HIP/CK/asm/HipKittens kernel can be 1.2–2.4× faster than baselines in some regimes.
- **The experimental Triton GEMM stub in aiter is NOT a real impl** — `aiter.ops.flydsl`/`tuned_gemm`
  treat `triton` as a libtype but the entry is a thin shim. Treat "Triton GEMM in aiter" as "author
  needed", not "available."
- **`tf32` input_precision is CDNA3-only** and removed on CDNA4; valid AMD `input_precision` values
  are `"ieee"` and (CDNA3) `"tf32"`. NVIDIA's `"tf32x3"` is not an AMD path.
- **Reduced dim < 64 wastes lanes** in `tl.sum`/`tl.max` wave reduces. Round the reduced dimension to
  a power of 2 ≥ 64.
- **Autotune in the serving hot path** adds first-call latency and is non-deterministic — bake a
  per-shape table (knobs.md §10).

## Integration (rebind seam) — sglang/vLLM
On sglang the dense GEMM path is **aiter**, not raw torch dispatch. An authored Triton GEMM must be
wired via the aiter seam (`aiter.tuned_gemm` `triton` libtype) or a call-site rebind, then **e2e-gated**
(Amdahl): only keep it if `pct_gpu_time × speedup` moves e2e beyond the noise band. Validation note:
an authored Triton GEMM measured 0.99–1.47× *isolated* but did **not** beat the aiter env at e2e
(2026-06).

## Verify before you trust (see isa_verify.md)
`AMDGCN_ENABLE_DUMP=1` → want `global_load_dwordx4`/`buffer_load_dwordx4`, `ds_*_b128`, dense
`v_mfma_*`, no `v_accvgpr_*` in loop, `.private_segment_fixed_size: 0`.

## Sources
- Optimizing Triton kernels (tuning pitfalls, OPTIMIZE_EPILOGUE, ISA): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- FNUZ fp8 normalization (sglang): https://github.com/sgl-project/sglang/pull/2601
- Honest compiler-vs-asm limits: HipKittens, https://arxiv.org/abs/2511.08083
- aiter tuned_gemm libtypes (triton stub): ROCm/aiter:aiter/tuned_gemm.py
