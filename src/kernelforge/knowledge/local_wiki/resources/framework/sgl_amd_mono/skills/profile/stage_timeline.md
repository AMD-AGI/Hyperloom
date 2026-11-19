---
title: stage timelines — per-CTA marks inside a monokernel and how to read the critical chain
kind: technique
gens: [gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/stamps.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/deepseek_v41/gfx950/layer.py
  - sgl-project/sgl-amd-mono@219f40be:test/kernels/deepseek_v41/check_k2.py
  - sgl-project/sgl-amd-mono@219f40be:test/kernels/kimi_k3/time_k2_stages.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
---

# Stage timelines

A monokernel is one launch, so a kernel profiler sees one bar. Where the time goes inside it — which
stage, which wait, which CTA, which rank — needs marks the launch records itself.

## The mechanism
- `common/stamps.py::stamp(c, point)`: thread 0 of each CTA stores the 100 MHz clock at mark `point`
  into its CTA's row of a `[CTA][TL_POINTS]` int64 scratch region. A build without the region
  (`c["tl"]` absent) compiles the marks out, so the served launch is a different, unmarked code object.
- `deepseek_v41`: the build key carries `timeline: bool`; `test/kernels/deepseek_v41/check_{k1,k2,ffn}.py
  --timeline` builds the runner's launches with marks.
- `kimi_k3`: each stage keeps its marks in LDS (`stamp_begin` … `stamp_flush`, a stage-local
  `TL_POINTS`) and writes them once; `test/kernels/kimi_k3/time_k2_stages.py` reads a
  `[BLOCKS, TL_POINTS]` int64 tensor.
- A launch without marks gets them the same way: a build-key field that is off in the served build.
  Do not add an environment toggle for it.

## Reading one
1. Ticks are 10 ns. Subtract each row's first mark (the CTA's start) or, across CTAs, the earliest
   start — CTAs do not start together.
2. For each stage take the latest end over CTAs (and over ranks, for a multi-rank launch): that is when
   the next stage's consumers can start. The sum along the dependent stages is the critical chain;
   the launch time is the chain, not the sum of stage times.
3. Classify each gap on the chain:

| What the marks show | Where to look |
|---|---|
| a stage ends late on a few CTAs while the rest idle | a fixed block of work on those CTAs — `skills/optimize/mono_levers/stage_placement.md` |
| consumers start well after the producer's last mark | the hand-off — `skills/optimize/mono_levers/handoff.md` |
| a GEMV unit's time is mostly waiting on its own weights | `skills/optimize/mono_levers/weight_streaming.md` |
| one rank's chain is longer | the slowest rank sets the step: look at its routing and its peers' pushes |

#59 is the worked example: timing MiniMax-M3.1's stages per CTA showed the router waiting for the norm
to write its rows, per-lane B-operand reloads dominating vector memory traffic, and an up unit
waiting on its own weights for about 3 µs of 5.6 — each became one change of that PR.

## Caveats
- Each mark in `common/stamps.py` is a global store: a small skew on the waits after it. Compare marked
  builds with marked builds; take the launch's time from an unmarked build.
- The rows of one launch are one step. Look at several steps (and at every width you serve): routing
  changes which CTAs carry the heaviest units from step to step.
