---
title: Common optimization methodology — index, file roles & problem-routing
kind: index
scope: common_methodology
updated: 2026-08-28
---

# Common optimization methodology — knowledge map

This file is the entry index for everything under `common_methodology/`. It gives
(1) what this knowledge base covers and how it is organized, (2) for a given task/symptom, **which files
to read and in what order**, and (3) the role of every file and folder.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What this knowledge base is
The **backend-agnostic reasoning layer** between hardware facts and a concrete kernel implementation. It
answers "given a slow kernel, *how do I find the real bottleneck* and *which general lever fixes it*?"
independent of the kernel language. It sits **above** `hardware/` (the metal + concrete numbers) and
**below** the language/framework folders (`languages/{hip,triton,gluon,flydsl,ck,asm}/`, `framework/`)
that turn a chosen lever into code. Everything here is grounded in **CDNA4 (gfx950 / MI350X·MI355X)**
facts — `hardware/` covers gfx950 only.

It is organized on **two axes** — *diagnosis* then *treatment*:
- **`profiling/`** — **how to measure and classify**: benchmark honestly, build a roofline, and read the
  four bottleneck classes off the counters. This is where an optimization loop *starts*.
- **`optimization/`** — **the levers**: one file per technique, each tied to the hardware fact it exploits
  and the bottleneck class it addresses. Has its own hand-written section index at
  `optimization/LEVERS.md`.

**The load-bearing rule:** *classify before you optimize.* Never pull an `optimization/` lever until
`profiling/` has told you which roof the kernel sits under — tuning MFMA on a bandwidth-bound norm (or
coalescing on a compute-bound GEMM) is wasted effort.

## Portable golden rules (internalize before optimizing)
- **Classify first.** Four failure modes: compute-bound, bandwidth-bound, occupancy-limited,
  latency-bound. Different roofs, different levers — place the kernel on the roofline before touching it.
- **Most inference kernels are HBM-bandwidth-bound** — optimize **bytes moved**, not FLOPs.
- **Peak ≠ achievable.** Tuned GEMM sustains only **~45–55% of theoretical matrix peak**
  (software-maturity ceiling). The bar is the **best tuned library kernel**, never the datasheet number.
- **Measure honestly.** Warm, **REPEATS=7**, inside the **~0.5% e2e noise band**, clocks locked/monitored,
  **same-session non-overlapping A/B**. A sub-band delta is not a result. Quote achieved, never peak.
- **Apply levers top-down** (algorithm/fusion → feed matrix cores → parallelism → memory → autotune the
  residual → keep it correct).
- **Loop *form* is an optimization axis, not only work volume.** `num_stages` pipelining and
  direct-to-LDS async copies require a **shape-static** trip count and addresses; a `while` bounded by
  a tensor load, or a `tl.load` addressed from another `tl.load`, disables both **silently**. A gather
  rewritten as "static range + mask" can do **2–4× more nominal work and still run ~2× faster**
  (measured, gfx950/Triton 3.6). Never reject a loop restructuring on block count alone
  (`optimization/lever_loop_form.md`).
- **Accumulate in fp32; verify every fast path** against an fp32 reference (`err_ratio < 0.05`), and watch
  the **fp8 FNUZ (CDNA3) vs OCP (CDNA4)** scale trap.
- **The editable list is a floor, not a ceiling, and it never bounds what you change.** A permitted file that runs first can
  rebind an installed package, carry device source through the framework's own hook, or hold the constant
  another module's dispatch reads — and a constant defaulted from `os.environ` is still an ordinary
  editable constant (`optimization/lever_edit_surface.md`).
- **Only aiter's per-shape DB engages the live sglang/vllm path** — triton `@autotune` and
  `hipblaslt-bench` tune authoring, not the deployed dispatch, unless rebound through aiter.

## Start here — problem → files → order
The canonical optimization loop reads across both folders: **benchmark a baseline → roofline → classify →
pick a lever → apply → re-benchmark A/B.** Paths are relative to this folder.

| Task / symptom | Read in this order |
|---|---|
| "Where do I even start on this kernel?" | `profiling/measure_protocol.md` (baseline) → `profiling/measure_roofline.md` → `profiling/measure_triage.md` → then a lever below |
| "Actually profile this kernel now / get its SoL+roofline / `rocprof-compute` won't run" | `profiling/measure_rocpc_workflow.md`, then run `python3 profiling/rocpc_profile.py --driver <driver> [--roofline]` → classify with `profiling/measure_triage.md` |
| "Classify it: compute / BW / occupancy / latency" | `profiling/measure_triage.md` → `optimization/lever_bottleneck_class.md` → `profiling/measure_roofline.md` |
| "Build / read a roofline on MI350X·MI355X" | `profiling/measure_roofline.md` → `optimization/lever_bottleneck_class.md` |
| "Compute-bound GEMM — feed the matrix cores" | `optimization/lever_mfma_sched.md` → `optimization/lever_prefetch.md` → `optimization/lever_lds_banks.md` → `optimization/lever_occupancy.md` |
| "Bandwidth-bound (norm / elementwise / decode GEMV / KV read)" | `optimization/lever_coalescing.md` → `optimization/lever_fusion.md` → `optimization/lever_xcd_locality.md` |
| "Low occupancy / register pressure / spilling / few waves/CU" | `optimization/lever_occupancy.md` → `optimization/lever_grid_sizing.md` |
| "Latency-bound — both roofs far, occupancy OK, high stall %" | `optimization/lever_prefetch.md` → `optimization/lever_grid_sizing.md` (more in-flight work) |
| "Sparse / top-k / paged-gather kernel — which loop should I even write?" | `optimization/lever_loop_form.md` → `optimization/lever_prefetch.md` → `profiling/measure_protocol.md` (A/B both loop forms) |
| "`num_stages` changes nothing / the pipelining knobs are ignored / `while` loop over a data-dependent index list" | `optimization/lever_loop_form.md` (a data-dependent trip count disables pipelining + async copy silently) → `optimization/lever_prefetch.md` |
| "LDS bank conflicts / `ds_read`·`ds_write` stalls / tile won't fit" | `optimization/lever_lds_banks.md` → `optimization/lever_prefetch.md` |
| "Chiplet locality / L2 reuse / tile swizzle / <1024 workgroups" | `optimization/lever_xcd_locality.md` → `optimization/lever_grid_sizing.md` |
| "Should I fuse these two ops?" | `optimization/lever_bottleneck_class.md` (re-classify first) → `optimization/lever_fusion.md` |
| "Wave/workgroup/grid sizing, `__launch_bounds__`, persistent kernels" | `optimization/lever_grid_sizing.md` → `optimization/lever_xcd_locality.md` |
| "Tune a GEMM for the live serving path" | `optimization/lever_autotune.md` → `profiling/measure_protocol.md` (validate via A/B) |
| "Time one constant fast / sweep a dispatch literal / should I keep the env knobs?" | `optimization/lever_cheap_sweeps.md` → `profiling/measure_protocol.md` (noise band) |
| "This lever needs a file I was not given / 'that means patching the framework, not this file'" | `optimization/lever_edit_surface.md` → `optimization/lever_cheap_sweeps.md` |
| "Is this constant editable? its default comes from `os.environ`" | `optimization/lever_edit_surface.md` (yes — it is a module constant in an editable file) |
| "Accuracy regressed / fp8 mismatch / softmax overflow / norm drift" | `optimization/lever_numerics.md` |
| "Did my change actually help? benchmark hygiene / A/B / noise band" | `profiling/measure_protocol.md` |

## Folder structure & file roles
```
common_methodology/
├── INDEX.md                              ← this map (load first)
├── profiling/                        ← DIAGNOSIS: measure and classify (start the loop here)
│   ├── measure_protocol.md           # warmup, REPEATS=7, ~0.5% noise band, locked clocks, same-session A/B, HIP graphs
│   ├── measure_roofline.md           # build & read an empirical roofline, per-dtype roofs
│   ├── measure_triage.md             # decision flow: compute / BW / occupancy / latency + counter signatures
│   ├── measure_rocpc_workflow.md     # HOW to run rocprof-compute here: the rocpc_profile.py script, the dependency-gate problem + fix, reading the tables
│   └── rocpc_profile.py              # the profiling SCRIPT the agent runs — stdlib-only; auto-detects a python with rocprof-compute's deps and SKIPS cleanly (exit 3) if none; prints Top-Stats + Speed-of-Light (+ roofline); isolate a kernel with --kernel <index>
└── optimization/                     ← TREATMENT: cross-operator performance levers
    ├── LEVERS.md                     # section index: the top-down lever hierarchy + per-file table (load for this folder)
    ├── lever_bottleneck_class.md     # arithmetic intensity, machine balance, bottleneck→lever map, ~45–55% reality
    ├── lever_occupancy.md            # 512 VGPR/EU, 16-granule alloc, AGPR pool, waves/EU, spilling cliff, waves_per_eu
    ├── lever_lds_banks.md            # 160 KiB LDS over 64 banks, padding vs XOR swizzle, double-buffer
    ├── lever_prefetch.md             # global_load_lds / async copy, software pipelining, num_stages, 128-bit direct-to-LDS
    ├── lever_loop_form.md            # shape-static trip count/addresses gate num_stages + async copy; data-dependent while-gather → static range + mask; the "more work, faster" trade
    ├── lever_mfma_sched.md           # 16×16 vs 32×32 MFMA, AGPR accumulators, issue cadence, OPTIMIZE_EPILOGUE, 512B Tagram
    ├── lever_coalescing.md           # 128-bit dwordx4 loads, alignment, coalesced/grid-stride access
    ├── lever_grid_sizing.md          # wave64, workgroup size, __launch_bounds__, persistent kernels, 256 CU, 8 XCDs
    ├── lever_xcd_locality.md         # 8-XCD per-die L2, ≥1024 workgroups, 8-multiple tiles, swizzled CTA order
    ├── lever_fusion.md               # when to fuse (epilogue/prologue, norm+quant, rope+cache, comm+norm), donors, when NOT to
    ├── lever_autotune.md             # AITER_TUNE_GEMM → err_ratio<0.05 → AITER_CONFIG_GEMM_BF16, per-shape key, engagement
    ├── lever_cheap_sweeps.md         # FORGE_SWEEP_<NAME> + sweep_const echo, one command per data point, KEEP the knobs through the search
    ├── lever_edit_surface.md         # what an editable file reaches: package rebind, injected device source, module constants (incl. os.environ defaults), data/config rows
    └── lever_numerics.md             # fp32 accumulate, online softmax, Welford, fp8 OCP scale trap, err_ratio gate
```

> **Naming convention.** Every card here is prefixed `lever_` (a technique you apply) or `measure_`
> (a way to observe). This is deliberate: it keeps the filenames distinct from any upstream knowledge
> base so a card can never be confused with, or silently overwritten by, an external copy.

## Reading-depth guide (how much to load)
- **Just diagnosing** (which roof am I under?): `profiling/measure_triage.md` +
  `profiling/measure_roofline.md` — don't pull any lever yet.
- **Applying one lever**: load the single `optimization/` file for that technique; it names the
  `hardware/` cards it depends on. Consult `optimization/LEVERS.md` if unsure which lever fits.
- **A full optimization pass**: walk the whole loop — benchmark baseline → roofline → classify → lever →
  re-benchmark A/B — following the routing table top-to-bottom.
- **Correctness gate**: `optimization/lever_numerics.md` before shipping any fast path.

## Cross-links out of this folder
Methodology is the reasoning layer, not the source of numbers or code. For **concrete hardware numbers**
(peaks, cache sizes, opcode tables) see `hardware/` — **gfx950 only**; each lever cites the specific card.
For **kernel-language mechanics** see `languages/{hip,triton,gluon,flydsl,ck,asm}/`.

**Applying a lever to a specific operator:** the language folders no longer carry per-operator cards, and
this base does not maintain general operator knowledge — read the kernel source. `framework/aiter/` is the
library control plane: `framework/aiter/overall/tuning_db.md` is the canonical worked example of turning
a lever into a real change (capture shapes → tune the per-shape DB → prove engagement), and
`framework/aiter/overall/operator_catalog.md` gives the entry point and signature for each operator
family. There are no per-operator cards anywhere in this base — that knowledge goes stale faster than
it can be maintained.
