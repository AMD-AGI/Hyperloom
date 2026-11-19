---
title: monokernel — one persistent launch for a whole decode span, collectives included
kind: lever
lever: monokernel
gens: [gfx950]
bottleneck: launch- and latency-bound small-batch decode
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:README.md
  - https://github.com/sgl-project/sgl-amd-mono/pull/56
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
---

# Monokernel

## Route here when
- Small-batch decode or speculative verify, where one step's rows are a few to a few dozen.
- A layer's span is a chain of short kernels **plus tensor-parallel all-reduces**, and the chain's time
  is well above the sum of its kernels' useful work even with CUDA graphs on.
- The model, its widths, TP sizes and ISA are fixed enough to build for: a monokernel is per model, per
  ISA and per TP size, and refuses everything else.

## Why it works
A graph replay removes launch overhead from the host, not the cost of a kernel boundary on the device:
every kernel still drains the GPU and ramps the next one up, and at decode widths each kernel is short,
so boundaries and collective latency become most of the chain. A monokernel keeps a grid of one
workgroup per CU resident for the whole span, hands data between stages and ranks through tagged
mailboxes in device and peer memory, runs the all-reduce inside the launch, and lets one stage's tail
overlap the next stage's weight loads.

| | `lever_fusion.md` (launch fusion) | Monokernel |
|---|---|---|
| Scope | a few neighbouring kernels | a whole span: projections, norms, routing, experts, both all-reduces |
| Synchronization | kernel boundaries remain around it | its own protocol: tags, epochs, guarded waits |
| Collectives | outside | inside, over peer memory |
| Cost | local | one register and LDS budget for every stage; per-model, per-ISA, per-TP code |

## Measured
MiniMax-M3.1-Flash at c1, TP4, MI350X: SGLang's standard path ran about seven launches a sparse layer
after attention, about 76 µs a layer, roughly half of an 8.66 ms step. One launch instead raised c1
decode from 390.8 to 422.3 tok/s (+8.1%, #56); folding o_proj in and cheapening its hand-offs added
another +7.4% (#59). `framework/sgl_amd_mono/` carries the per-launch numbers.

## What it costs
- **The largest stage sets the budget for all of them**: one 512-thread CTA a CU leaves at most 256
  registers a wave, and every stage's LDS pool shares the CU's LDS.
- **A protocol to get right**: a missed publication or a stale tag does not produce a wrong number — it
  stops the launch at its deadline. The waits must be fail-stop on every rank.
- **Coverage is explicit**: steps past the largest width, other TP sizes and other ISAs run the standard
  path; the spec refuses configurations it cannot serve.

## Verify
| Check | Pass |
|---|---|
| kernel-only time of the launch vs the standard path's chain for the same span | lower at every served width |
| e2e decode with monokernels on and off, including past the largest width | faster inside the widths; equal past them |
| numerics vs the standard path on the same GPU | within the spec's bounds (bitwise where it claims it) |
| fail-stop | a stalled rank stops every GPU and returns no token |

## Failure modes
| Symptom | Cause |
|---|---|
| times out at the first cross-CTA wait | the grid is not resident (LDS or registers) |
| slower than the kernels it replaced | one stage's budget starved the rest, or the critical chain is hand-offs — profile per stage |
| gain vanishes in serving | the served steps are past the largest width |

## Deeper
`framework/sgl_amd_mono/` (interface, runtime, stage timelines, hand-off / placement / weight-streaming
levers) · `lever_fusion.md` · `lever_grid_sizing.md` (persistent kernels) · `lever_occupancy.md`
