---
title: aiter kernel authoring — delegate to the per-language folders
kind: language
gens: [gfx942, gfx950, gfx1250]
dtypes: [both]
regimes: [both]
status: sota
updated: 2026-07-14
---

# Authoring aiter kernels — where the real work lives

## TL;DR
aiter has **no language of its own** — every aiter kernel is written in CK, ASM, HIP/C++, Triton, or
FlyDSL. So when a task needs *editing kernel source* (not tuning the dispatch DB), you are working in one
of those languages, and the authoring knowledge lives in that language's `local_knowledge` folder. This
card is the router; it deliberately does **not** duplicate MFMA/LDS/knob docs.

## Which folder for which aiter kernel
| aiter kernel family | written in | source location (aiter repo) | authoring knowledge |
|---|---|---|---|
| CK GEMM/attention/norm (`gemm_a8w8_ck`, `ck_moe_*`, `*_cktile`) | Composable Kernel (C++ templates) | `csrc/ck_*`, `3rdparty/composable_kernel` | `local_knowledge/languages/ck/` (ck_tile, ck_classic, gemm/fmha templates, knobs) |
| ASM kernels (`*_asm`, `pa_fwd_asm`, `mla_*_asm`, HSACO) | raw AMDGCN assembly | `hsa/{gfx}/…` | CDNA ISA facts in `languages/hip/skills/optimize/hip_levers/`; kernelforge ships no assembly authoring layer |
| HIP/C++ ops (incl. HipKittens) | HIP C++ | `csrc/*` | `local_knowledge/languages/hip/` (intrinsics, lds_async, patterns, hipkittens) |
| Triton ops (`aiter.ops.triton.*`) | Triton | `aiter/ops/triton/*` | `local_knowledge/languages/triton/` (knobs, patterns, isa_verify) |
| FlyDSL ops (`aiter.ops.flydsl.*`) | FlyDSL | `aiter/ops/flydsl/*` | `local_knowledge/languages/flydsl/` |
| opus split-K GEMM/MoE (`opus_gemm`, `moe_stage2_a8w4`) | HIP/C++ split-K kernels | `aiter/ops/opus/*`, `csrc/opus_gemm/*` | aiter-internal; tune via `gemm_a16w16_tune.py --libtype opus` (see [tuning_db.md](tuning_db.md)) |
| hipBLASLt-dispatched GEMM | (closed library) | n/a | not authored — tune via the DB ([tuning_db.md](tuning_db.md)) |

## Decide: tune the DB, or author a kernel?
1. **First choice — tune the dispatch DB** ([tuning_db.md](tuning_db.md)). It's reversible, parity-safe,
   engages the live serving path, and needs no source edit. This is the default aiter optimization.
2. **Author/replace a kernel** only when (a) no library kernel exists for the shape/fusion, or (b) DB
   tuning has plateaued and the profile shows a real ceiling to beat. Then:
   - pick the language by the table above and open that folder's `*_levers` for the authoring rules;
   - build/JIT via [jit_and_build.md](jit_and_build.md) (mind the stale-cache `AITER_REBUILD` trap);
   - engage + e2e-gate via [dispatch_and_rebind.md](dispatch_and_rebind.md) — an isolated win that never
     hits the live seam is a reject.

## Why this card exists (no duplication)
Copying MFMA intrinsics, LDS swizzle, or Triton knobs into an aiter folder would fork the same facts
across backends and rot. aiter's unique knowledge is the **library control plane** (catalog, build,
DB tuning, dispatch); the language facts stay single-sourced in `languages/hip/`, `languages/triton/`,
`languages/flydsl/`, `languages/ck/`. Follow the links; don't re-document.
