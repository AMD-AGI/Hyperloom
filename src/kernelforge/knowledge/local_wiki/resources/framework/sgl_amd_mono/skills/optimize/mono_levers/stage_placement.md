---
title: stage placement — which CTAs run which work, and when
kind: lever
lever: monokernel_stage_placement
gens: [gfx950]
bottleneck: latency-bound (a fixed block on few CTAs)
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/plan.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
  - https://github.com/sgl-project/sgl-amd-mono/pull/68
---

# Stage placement

## Route here when
The stage timeline shows a stage ending late on a few CTAs while the rest of the grid waits for it, or
one CTA carrying the tails of two stages. Inside one launch, a CTA's place in the order of stages
decides the step as much as the work itself.

## What to change
1. **Run work that does not depend on the critical wait before it.** glm53's shared expert does not
   depend on the routing, but its up/gate tiles ran after it, a fixed block on 32 CTAs once the top-8
   was done; they now run first, with their weights issued ahead of the activations' wait (#68).
2. **Move a task off the CTA that already ends last.** An index-query tile that landed on the cache
   task's CTA now runs on the CTA of the first extra tile (#68).
3. **Choose where a stage starts.** `plan.first_task(bid, base)` gives CTA `bid` its first task of a
   stage whose task 0 sits on CTA `base`; pick `base` so a stage's first tasks land on CTAs that finish
   the stage before it early.
4. **Pull an adjacent kernel in when it sits on the chain.** MiniMax-M3.1's o_proj ran as two kernels
   in front of the launch; it now runs on the push CTAs, from the attention rows, to SGLang's bits (#59).

## Verify
- The timeline: the stage that ended late ends with the rest, and the chain shortened by about the
  block that moved.
- Bitwise equal to `main` at every width: placement moves work, not values (#68 compared every output
  and every write, bf16 and E4M3 MLA caches, TP4 and TP8).

## Expected magnitude
glm53 (#68), two placement changes with three hand-off changes: kernel-only 64.19 → 59.25 µs a launch
(−7.7%) at TP8, 2 rows; 84.96 → 76.90 (−9.5%) at 4 rows. #68 lists each change's own gain.

## Failure modes
| Symptom | Cause |
|---|---|
| no gain | the moved block was not on the critical chain: re-read the timeline |
| gain at one width, loss at another | the placement assumed a task count that changes with rows; check every served width |
| a TP size stops compiling | a placement that only exists at some TP sizes needs its guard (#68 kept TP4's indexer builds compiling) |
