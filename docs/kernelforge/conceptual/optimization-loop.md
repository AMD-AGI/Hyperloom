---
myst:
  html_meta:
    "description": "The KernelForge enforced development loop: build, SNR correctness gate, benchmark, hardware-counter (PMC) analysis, and a measured keep-or-revert decision."
    "keywords": "KernelForge, optimization loop, PMC, wait/MFMA ratio, SNR gate, occupancy, VGPR, roofline, measurement-driven"
---

# Optimization loop

Every kernel backend follows the same enforced development loop. Skipping a step is
blocked by the system, so no change is ever accepted without measured evidence.

## The enforced loop

1. **Build** — compile the kernel; stale artifacts are auto-cleaned so an edit
   cannot be silently benchmarked against old code.
2. **SNR pre-filter** — run the correctness driver; a change must clear the
   signal-to-noise threshold (default 30 dB) before it can be benchmarked.
3. **Benchmark** — measure wall-clock and kernel time (median over iterations).
4. **PMC analysis** — read hardware counters and classify the bottleneck.
5. **Accept** — reproduce the arena's verdict on a candidate that would
   otherwise be taken: the task's own `compile_command`, then its
   `correctness_command`, stopping at the first failure. Keep the change only if
   both pass, otherwise revert. The compile step matters on its own — the task
   often builds a smaller shape than the one the loop measures. A knowledge-base
   warm start is accepted by the same step before it can become the starting
   point.

## PMC-guided optimization

Decisions are grounded in hardware counters rather than guesswork. The
wait/MFMA ratio is a primary signal:

| wait/MFMA ratio | Diagnosis | Action |
|:---------------:|:----------|:-------|
| < 5 | Compute-bound | Reduce MFMA count (tile shape, warp config) |
| 5–10 | Balanced | Profile deeper (LDS vs VMEM stalls) |
| > 10 | Memory-bound | Reduce HBM traffic (occupancy, prefetch) |

The `registers` tool predicts VGPR/SGPR usage and occupancy **before** building,
so an occupancy cliff (for example VGPR crossing 256) is caught before it costs
a build-bench cycle.

## Correctness and pitfall gates

Each validation gate exists because of a real incident. Examples enforced by the
knowledge base and tooling:

- Stale `.so` artifacts — the build tool auto-cleans and verifies deployment.
- A `BLOCK_M=64` sparse-attention configuration that silently corrupts data —
  a hard constraint in the knowledge base.
- An AGPR inline-asm register drop that produces silently wrong output — blocked
  by the SNR pre-filter below 30 dB.

## Autonomous loop

The autonomous loop wraps this cycle for overnight, unattended optimization:
Analysis and Orchestration produce one plan, the Implementer edits the working
tree, the driver-owned complete correctness suite validates it, and three
independent benchmarks decide whether to commit or restore the candidate.
See {doc}`Autonomous overnight loop </kernelforge/how-to/autonomous-loop>`.
