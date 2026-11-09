---
title: sgl-amd-mono knowledge map — monokernels for SGLang on AMD Instinct
kind: index
scope: framework/sgl_amd_mono
updated: 2026-11-09
pinned_source: sgl-project/sgl-amd-mono@219f40be
---

# sgl-amd-mono — knowledge map

`sgl-amd-mono` (`sgl-project/sgl-amd-mono`) is SGLang's monokernel library for AMD Instinct GPUs. Per
model, one persistent launch runs most of a decoder layer — its attention or FFN span, the MoE and both
tensor-parallel all-reduces — on a resident grid of one workgroup per CU, so a small-batch decode step
pays neither the launch gaps between kernels nor a separate collective. SGLang owns the interface
(`sglang.kernels.plugin_kernels.amd`); the repository owns the device runtime (`sgl_amd_mono/common/`)
and each model's spec, forward and launches. Every launch it ships today is built for gfx950
(MI350X / MI355X).

A monokernel campaign is a forge-loop task on the `mono` kernel backend, which forge also infers for
sources under the `sgl_amd_mono/` package. Its KEEP rests on the repository's own tests, declared in
the task's `config.yaml`; the driver's verdict is only the pre-filter.

## Reading order
1. `overall/repo_layout.md` — packages, how a served step reaches a launch, the three test tiers.
2. `overall/monokernel_interface.md` — the `MonoSpec` contract a candidate must not move.
3. `overall/device_runtime.md` — `common/`: the grid, mailboxes, guards, epochs, peer memory.
4. The model card for the launch under optimization (`models/`), then a lever.

## Start here — problem → files → order
| Task / symptom | Read in this order |
|---|---|
| "What does this launch do, stage by stage?" | `models/<model>.md` → the launch module's docstring |
| "Where does the step's time go?" | `skills/profile/stage_timeline.md` → `skills/optimize/mono_levers/handoff.md` |
| "Will this change fit the resident grid? Does it spill?" | `skills/profile/compile_check.md` |
| "Port a launch to another ISA (gfx950 → gfx942)" | `overall/isa_portability.md` → `skills/profile/compile_check.md` |
| "The launch hangs, stops at the null page, or fails after N steps" | `skills/bottleneck/debug_monokernel.md` |
| "CTAs wait on each other; the critical chain is hand-offs" | `skills/optimize/mono_levers/handoff.md` → `skills/optimize/mono_levers/stage_placement.md` |
| "A stage's work sits on a few CTAs while the rest wait" | `skills/optimize/mono_levers/stage_placement.md` |
| "The weights stream slower than HBM allows" | `skills/optimize/mono_levers/weight_streaming.md` |
| "Is a monokernel worth it for this span at all?" | `common_methodology/optimization/lever_monokernel.md` |
| "Turn a KEEP into an upstream PR" | `overall/contributing.md` |

## Folder structure & file roles
```
framework/sgl_amd_mono/
├── INDEX.md                          ← this map
├── overall/
│   ├── repo_layout.md                # packages, spec registry, step → launch path, test tiers
│   ├── monokernel_interface.md       # MonoSpec, DeviceLimits, Requirements, Numerics, routing, refusal
│   ├── device_runtime.md             # common/: plan, sync, guard, epoch, peer, stamps, build keys
│   ├── isa_portability.md            # gfx950-only primitives in common/, LDS and CU budgets, FP8 formats
│   └── contributing.md               # upstream PR shape and the evidence a perf PR carries
├── models/
│   └── minimax_m31_ffn.md            # MiniMax-M3.1-Flash FFN launch: stages, budgets, measured times
└── skills/
    ├── profile/
    │   ├── stage_timeline.md         # per-CTA stage marks and how to read the critical chain
    │   └── compile_check.md          # cross-ISA compile, VGPR / LDS / spill metadata, residency
    ├── bottleneck/
    │   └── debug_monokernel.md       # deadlines, error words, null-page stops, fault injection, check mode
    └── optimize/mono_levers/
        ├── handoff.md                # tagged pairs, uncached scratch, poll placement, wait-loop shape
        ├── stage_placement.md        # fixed blocks on few CTAs, task rotation, stage overlap
        └── weight_streaming.md       # next-unit weight issue, nontemporal loads, operands from LDS
```

## What to sync when sgl-amd-mono moves
1. Re-verify each card against the new commit: the `common/` APIs and constants, the `MonoSpec` fields of
   the installed SGLang, and every cited test path.
2. Bump `pinned_source` here and `sources:` / `updated:` on each touched card.
