---
title: sgl-amd-mono device runtime — resident grid, mailboxes, guards, epochs, peer memory
kind: technique
gens: [gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/__init__.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/plan.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/sync.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/guard.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/epoch.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/peer.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/runner.py
---

# The device runtime (`sgl_amd_mono/common/`)

Every launch builds on the same few modules, adapted from ROCm/ATOM (`atom/mono/`) by way of vLLM's
DeepSeek-V4.1 mono decode. A change here reaches every model's launches.

## Execution model (`plan.py`)
- `BLOCKS = 256` CTAs × `THREADS = 512` (`WAVES = 8`): one CTA a CU, every CTA resident at once.
  `DeviceLimits.resident_blocks` states the grid per ISA.
- A persistent grid deadlocks if one CTA waits on another that was never scheduled, so the runner
  refuses a width whose grid cannot be resident: `minimax_m31/runner.py::warm` asks the HIP driver for
  the compiled kernel's resident CTAs per CU and refuses `per_cu × CUs < BLOCKS`
  (`skills/profile/compile_check.md`).
- `first_task(bid, base)` = CTA `bid`'s first task of a stage whose task 0 is placed on CTA `base`
  (its next tasks follow every `BLOCKS`): the base is how a launch chooses which CTAs start a stage
  (`skills/optimize/mono_levers/stage_placement.md`).
- `kernel_sources(module)`: a digest of the module and every plugin and aiter FlyDSL module it imports,
  transitively, without comments or docstrings; taken at import (`SOURCES = kernel_sources(__name__)`)
  and part of every launch's cache key with the build parameters.

## Hand-offs (`sync.py`)
- A **mailbox** is a tagged-pair hand-off: each value is stored next to the launch tag (`PAIR_BYTES = 8`),
  so a consumer polls until the tag is this launch's and never needs a separate ready flag.
  Spaces: `SCRATCH` (between CTAs, device scope, cache policy `CM_DEV`) and `PEER` (between GPUs,
  system scope, `CM_SYS`).
- `publish` drains the CTA's plain stores, then stores the pair from one thread. A poll batches at most
  `POLL_MAX = 12` mailbox specs, which bounds its live registers.
- Sentinel regions: a word holds `SENT = -1` until written; a read consumes it and re-arms it, one thread
  a word.

## Guards (`guard.py`) — fail-stop
- Each wait on another CTA or rank has a deadline below the server's watchdog: half of
  `--watchdog-timeout`, at most 60 s; SGLang refuses one under 5 s. A pending wait checks its deadline
  and the peers' failure flags every `CHECK_POLLS = 64` polls, on the 100 MHz `s_memrealtime` clock.
- On a timeout the wait sets a sticky error word, flags every peer, fences and stops the launch with a
  volatile store to the null page; the process stops with it. A wait that sees a failure elsewhere
  stops too. A launch never runs on after a failed wait, so a finished step's tokens are correct with
  nothing checked on the host.
- The guard block heads the epoch buffer (int32 words): 0 epoch · 1–2 deadline ticks (int64) ·
  3 `deadline` = 1 + the site of the first timed-out wait · 4 `peer` · 5 `fault`. Each rank's peer
  buffer holds `FLAG_WORDS = 8` failure flags (TP 8 at most).
- `deadline_ticks` refuses a deadline the 32-bit check cannot hold (time is compared in 2.56 µs units).
- Test fault injection: `SGLANG_AMD_MONO_FAULT_INJECT=rank:site[:after]` stalls the wait at `site` on
  `rank` once `after` launches ran (`skills/bottleneck/debug_monokernel.md`).

## Epochs (`epoch.py`)
Every CTA marks the launch epoch; CTA 0 moves it on once all have. `kernel_symbol` names a kernel
`aiter::<stem>_<params>` so profilers demangle it.

## Peer memory (`peer.py`)
`PeerBuffer(nbytes, group, rank, npes, device)`: one uncached buffer a rank
(`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`, 2 MiB granules), HIP IPC handles exchanged
over a CPU (gloo) group, and every rank's base address in `addresses` (int64, on the device).
Construction is collective and every rank refuses together; ranks must build their buffers in the same
order and sizes. A one-rank `PeerBuffer` is also how a launch gets uncached scratch for its in-launch
hand-offs (#62, #68).

## The rest
| Module | Role |
|---|---|
| `ops.py` | wave ops (DPP within a row, `v_permlane{16,32}_swap` across rows), hardware math, packing, buffer access; cache-policy bits `CM_DEV = 16`, `CM_SYS = 17`, `CM_NT = 2` |
| `mx.py` | FP8 / MX arithmetic: E4M3 clamp (`FP8_MAX = 448`), E8M0 scales, the scaled MFMA |
| `stamps.py` | timeline stage marks, debug builds only (`skills/profile/stage_timeline.md`) |
| `hip_ext.py` | HIP C++ built into a torch extension at first use, cached under `SGLANG_AMD_MONO_CACHE_DIR`; one rank builds |
| `rocm_codegen.py` | `SGLANG_AMD_MONO_ROCM_LLVM=<prefix>` compiles this package's kernels with an external ROCm LLVM (FlyDSL's bundled LLVM and ROCm's each miscompiled a different heavily spilling variant) |
