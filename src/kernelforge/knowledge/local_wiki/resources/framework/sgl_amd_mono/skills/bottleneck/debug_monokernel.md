---
title: debugging a monokernel — deadlines, error words, null-page stops, fault injection, check mode
kind: technique
gens: [gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/guard.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/sync.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/spec.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/55
  - https://github.com/sgl-project/sgl-amd-mono/pull/56
  - https://github.com/sgl-project/sgl-amd-mono/pull/62
---

# Debugging a monokernel

A broken monokernel rarely returns a wrong number: its waits are guarded, so a broken protocol stops
the launch at a deadline. Start from how it stopped.

## Read the stop
- **A GPU fault at address 0x0** is the guard's deliberate stop (`guard.stop_launch`, a volatile store
  to the null page) after a failed wait. A fault at any other address is a real out-of-bounds access.
- **The error words** at the head of the epoch buffer say which wait failed: word 3 is
  1 + the site of the first wait on this GPU that timed out; word 4 is set when a wait saw a failure on
  another rank (look there first); word 5 is set by fault injection.
- A wait times out at its deadline: half of `--watchdog-timeout`, at most 60 s. Never raise it to make
  a candidate pass — a wait that needs longer is waiting on something that will not come.
- One GPU, no stop: `Guard(stop=False)` keeps a failed launch running so a test reads the words
  afterwards.

## Symptom → first suspect
| Symptom | First suspect |
|---|---|
| times out at the first cross-CTA wait, every time | the grid is not resident (`skills/profile/compile_check.md`) |
| times out only at one width | that width's own path: an LDS pool or branch the others do not take |
| the first launch works, a later one stops | state left by the launch before: a sentinel not re-armed, an epoch not moved on, a tag reused |
| stops after tens of thousands of launches | a tag wrapped: MiniMax-M3.1's 16-bit seam tags cycle 1 … 65535 per parity, and the last launch of a cycle zeroes that parity's receive half (#56) |
| only one rank stops first | its peers' word 4 points at it: read that rank's word 3 |
| eager passes, the replayed graph differs | host state or a host sync in the step; compare a graph's replays with the eager step |
| serving accuracy drifts, kernels pass | run check mode |

## Tools
- **Check mode**: `SGLANG_AMD_MONO_CHECK=1` runs SGLang's layer on a copy of each covered layer's input
  in eager steps and logs both outputs' differences — the source of every `NUMERICS` bound.
- **Fault injection**: `SGLANG_AMD_MONO_FAULT_INJECT=rank:site[:after]` stalls the wait at `site` on
  `rank` once `after` launches ran; the e2e fault cases expect every GPU to stop at the null page with no
  token sent. Sites are numbered per build: a change that adds or removes guarded waits renumbers them,
  so keep the count (#62 did) or update the cases in the same change.
- **Crash dumps** for fault tests: `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` (ROCr's per-rank VRAM dump was
  about 82 GB at TP4), `SGLANG_PYSPY_DUMP_BEFORE_CRASH=0`, `SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0` (#55).

## In a forge campaign
A candidate that breaks the protocol ends its driver run at the deadline with a fault, which forge
records as a failed validation. Read the error words before reading the diff; then shrink the case —
one width, one rank's view, the stage before the failed site.
