---
myst:
    html_meta:
        "description": "Learn about GEAK, Hyperloom's agent-driven GPU kernel optimization framework. Covers Triton, HIP, and FlyDSL kernel rewriting, parallel optimization, and patch validation."
        "keywords": "GEAK, Hyperloom, GPU kernel optimization, Triton, HIP, FlyDSL, AMD GPU, ROCm, kernel rewriting, benchmarking, parallel optimization, LLM inference, agent, Ray"
---
# GEAK

GEAK (Generating Efficient AI-Centric Kernels) is a multi-agent framework for end-to-end GPU kernel
optimization in real codebases. It runs a closed loop of profiling, optimization, and
validation, and produces reviewable patches backed by reproducible benchmarks.
GEAK supports Triton, HIP (and CUDA, Composable Kernel (CK), and HSA Code Object (HSACO)), and FlyDSL
kernels. It is driven by [Claude Code](https://www.anthropic.com/claude-code) and ships two
deterministic JS Workflows — `e2e_workflow` for whole-model serving throughput and `kernel_workflow`
for single kernels — with a deterministic control plane (budget loop, parallel fan-out, verification,
stop conditions) that invokes LLM agents only for judgment.

Within Hyperloom, GEAK is the whole-pipeline end-to-end optimization delegate: when a workload is
handed off, the orchestrator invokes GEAK once at the kernel-agent phase through the stable
`interface/run_e2e.py` contract (a `handoff.json` in, a `result.json` back). Parallel exploration of
candidate kernels then happens inside GEAK's Workflows on the on-box GPUs.

- **Source**: <https://github.com/AMD-AGI/GEAK>
- **License**: MIT

## Role in Hyperloom

Hyperloom uses GEAK as the **whole-pipeline e2e delegate** when
`KERNEL_OPT_BACKEND_ORDER=geak` (the bare-metal default). In this mode the
orchestrator hands the optimization workload to
`src/hyperloom/agents/kernel/tools/backends/geak_runner.py`, which resolves the
GEAK checkout and launches GEAK's e2e runner (`interface/run_e2e.py`) with the
generated session context.

When GEAK owns the phase it runs the whole optimization loop itself — both the
end-to-end serving optimization *and* the per-kernel work underneath it, since
GEAK's `e2e_workflow` recursively drives `kernel_workflow` to author and tune the
individual hot kernels worth fixing. See
[Hyperloom optimization loop](../conceptual/optimization-loop.md).

## GPU pinning in the handoff

GEAK launches full servers out-of-process (baseline, profile, config-tuning
validation) and writes a visible-devices mask for each one, so the handoff has
to say which cards the run owns. Two fields carry that, in two different
coordinate systems:

| Field | Coordinate system | Value |
|-------|-------------------|-------|
| `gpu_ids` | HIP-level device list — HIP indexes into the ROCr-visible set | logical positions inside an *inherited* `ROCR_VISIBLE_DEVICES` mask, capped at `tp` and at the mask width (`ROCR=6` → `"0"`); a `HIP`/`CUDA` mask uncapped (`HIP=4,5` → `"4,5"`); `0..tp-1` when the run is unpinned. Ids are re-serialized from the parsed mask, so whitespace and repeats are normalized |
| `gpu_pin` | absolute device ids | `{"var", "value", "ids", "count", "source"}` for the winning mask — omitted entirely when no mask is set anywhere, which means "whole machine visible", not "pinned to card 0". `value` is the mask verbatim (so a UUID mask can be re-exported); `ids` is empty for a non-numeric mask, hence `count` |

The mask is resolved variable-major — `ROCR_VISIBLE_DEVICES` before
`HIP_VISIBLE_DEVICES` before `CUDA_VISIBLE_DEVICES`, the same order as
`orchestrator/bus/gpu_pool.py` and `orchestrator/policy/gate.py` — and within
each variable the process environment before the materialized baseline
recipe's `benchmark.envs`.

The process env comes first because the recipe's ROCR key is not evidence of a
pin: `materialize_config_with_envs` autofills `ROCR_VISIBLE_DEVICES=0..tp-1`
into every materialized recipe when the mask is absent or narrower than `TP`.
A recipe ROCR value byte-identical to that default is therefore ignored, so a
`HIP`-pinned or genuinely unpinned run is not silently re-pinned to cards
`0..tp-1`. A recipe mask that differs from the default *is* honoured — but its
ids are forwarded absolute, not logical, because the GEAK child inherits the
process environment and never sees that mask.

`tp` in the handoff is read from the same resolved recipe as `gpu_ids`, so the
two cannot disagree when the materializer clamps `TP` to the visible GPU count.

A consumer that re-exports `HIP_VISIBLE_DEVICES` should use `gpu_ids`; one that
writes `ROCR_VISIBLE_DEVICES` itself must use `gpu_pin["value"]`, because
writing `gpu_ids` into `ROCR_VISIBLE_DEVICES` resets the child to physical card
0 regardless of the run's pin.

## GEAK documentation

For detailed documentation on GEAK, see [GEAK on ROCm Docs](https://rocm.docs.amd.com/projects/geak/en/latest/).
