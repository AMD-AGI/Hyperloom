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
| `gpu_ids` | HIP-level device list — HIP indexes into the ROCr-visible set | logical positions inside an *inherited* ROCr-level mask, capped at `tp` and at the mask width (`ROCR=6` → `"0"`); a HIP-level mask nested inside that slice is already logical and is forwarded instead (`ROCR=4,5,6,7` + `HIP=2,3` → `"2,3"`, i.e. cards 6 and 7); any other mask passes through uncapped (`HIP=4,5` → `"4,5"`); `0..tp-1` when the run is unpinned. Never empty — a falsy `gpu_ids` sends GEAK back to its own `0..tp-1` fallback |
| `gpu_ids_space` | — | `"logical"` when `gpu_ids` indexes into an inherited ROCr mask, `"absolute"` otherwise. The one field that makes the two coordinate systems distinguishable from the payload alone |
| `gpu_pin` | absolute device ids | `{"var", "value", "ids", "count", "source"}` for the winning mask, plus `"inner"` (the same shape) when a HIP-level mask is nested inside a ROCr-level one. Omitted entirely only when no mask is set anywhere, which means "whole machine visible", not "pinned to card 0"; a mask that is *set but empty* is reported with `count: 0` (zero devices visible). `value` is the mask as authored, only whitespace-trimmed and (for a YAML sequence) comma-joined, so a UUID mask can be re-exported as-is; `ids` are its numeric ids and `count` how many devices it exposes — both derived from the same effective token list, with duplicate and negative ordinals dropped, so `count >= len(ids)` always and they differ only for a (partly) non-numeric mask |

The mask is resolved variable-major, over the full ROCm precedence chain in
`common/visible_devices.py`: the ROCr-level masks (`ROCR_VISIBLE_DEVICES`, then
its legacy spelling `HSA_VISIBLE_DEVICES`) before the HIP-level ones
(`HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, `GPU_DEVICE_ORDINAL`). Within
each variable the process environment comes before the materialized baseline
recipe's `benchmark.envs`.

That module is also the single definition of the tuple and the mask parser that
`orchestrator/bus/gpu_pool.py`, `orchestrator/policy/gate.py`,
`actions/executors/_ray_serving.py` and `common/env_safety.py` use. Those layers
read the narrower `COUNTING_VISIBLE_DEVICE_VARS` on purpose — they answer "how
many GPUs does this process have", and widening that would change GPU accounting
repo-wide — while this resolver answers "where is this run pinned", and a run
pinned with a legacy spelling is really pinned.

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

A consumer that re-exports `HIP_VISIBLE_DEVICES` and leaves ROCr alone should
use `gpu_ids`; that is correct in both coordinate systems, because the child
inherits the same ROCr mask this process runs under.

A consumer that writes `ROCR_VISIBLE_DEVICES` itself must use
`gpu_pin["value"]` — writing `gpu_ids` there resets the child to physical card
0 regardless of the run's pin — and must then renumber: it has just made the
device set `0..count-1` from the child's point of view, so the inner HIP mask
is `0..count-1`, **not** `gpu_ids`. Re-applying an absolute `gpu_ids` on top of
a ROCr mask it also wrote yields out-of-range ordinals (`ROCR=4,5` plus
`HIP=4,5` indexes positions 4 and 5 of a two-element set). `gpu_ids_space` is
how a consumer tells the two cases apart without inferring it from `source`.

## GEAK documentation

For detailed documentation on GEAK, see [GEAK on ROCm Docs](https://rocm.docs.amd.com/projects/geak/en/latest/).
