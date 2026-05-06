---
name: inference-optimization-sweep
description: |
  Sweep-only LLM serving benchmark on AMD GPUs. Skips DFS / kernel-opt /
  baseline / profile and runs a single ISL/OSL/CONC sweep with user-provided
  server args via Magpie's sweep_matrix (single-server reuse).
globs:
  - "**/sweep*"
  - "**/benchmark*sweep*"
---

# Inference Optimization (Sweep-Only)

## Overview

This skill maps the Pareto frontier (CONC × ISL × OSL) on a fixed, user-provided
server configuration. It is a stripped-down sibling of `inference-optimization-magpie`:
no DFS search, no kernel optimization, no accuracy gating, no KB. Use it when
you already know the winning server args (from a prior optimization run, or a
production reference config) and only need throughput/latency curves.

The full DFS-guided autonomous skill is [`inference-optimization-magpie`](../inference-optimization/SKILL.md).
Pick this lightweight skill when:

- Server config is fixed and validated.
- You only need an ISL/OSL/CONC matrix on the existing setup.
- You want to reuse a single server across the matrix to avoid N × CUDA graph capture.

## Required Inputs

The agent MUST receive these from the user (or environment):

| Variable | Purpose | Example |
|----------|---------|---------|
| `MODEL` | Model path or HF id | `/shared_nfs/models/DeepSeek-R1-0528` |
| `TP` | Tensor parallel size | `8` |
| `FRAMEWORK` | `sglang` or `vllm` | `sglang` |
| `EXTRA_SGLANG_ARGS` or `EXTRA_VLLM_ARGS` | Server flags (winning config) | `"--attention-backend aiter --kv-cache-dtype fp8_e4m3 --disable-radix-cache"` |
| `RUNNER_TYPE` *(optional)* | Auto-detected from GPU arch when omitted | `mi355x` |
| `SWEEP_CASES_YAML` *(optional)* | Override default matrix; YAML list under `sweep_matrix.cases` | (see Defaults) |

If `EXTRA_*_ARGS` is not provided the skill stops with an error — sweep needs
a fixed server config to amortize startup.

## Default Sweep Matrix

```yaml
cases:
  - { CONC: 4,  ISL: 1024, OSL: 1024 }
  - { CONC: 16, ISL: 1024, OSL: 1024 }
  - { CONC: 64, ISL: 1024, OSL: 1024 }
  - { CONC: 16, ISL: 8192, OSL: 1024 }
  - { CONC: 16, ISL: 1024, OSL: 8192 }
on_failure: continue
inter_client_sleep_s: 5
```

## Iron Rules (subset of inference-optimization-magpie)

These rules are kept verbatim because they apply to any sweep run:

- **IR-4**: Always `kill_server` and verify GPU memory is free before launching.
- **IR-5**: Use targeted `kill $(pgrep -f 'python.*-m sglang.launch_server')` or
  `kill $(pgrep -f 'python.*-m vllm.entrypoints')`. NEVER `pkill -f sglang`.
- **IR-8**: Wrap long-running `magpie benchmark` in a background runner so the
  agent can stream progress without holding the foreground.
- **IR-10**: In claw mode, only `workload_create(kind="RayJob")`, `workload_get`,
  `workload_list`, `workload_stop`. Never `workload_delete`.
- **IR-12**: Claw sandbox already mounts shared NFS at `/shared_nfs`; do not
  attempt to remount it.

Rules from the full skill that are **dropped** in this sweep-only flow:

- IR-1 / IR-2 / IR-3 / IR-7 (kernel optimization & GEAK) — sweep does not
  touch kernels.
- IR-6 (`patch_inductor.py`) — same reason.
- IR-9 (RayJob wrapper for kernel-opt) — only the inference RayJob is needed.
- IR-11 (kernel-opt image pinning) — not applicable.

## Magpie Prerequisite

The skill requires Magpie with `sweep_matrix` support. `actions/setup.md`
imports `Magpie.modes.benchmark.config.SweepMatrix`; if the import fails the
run aborts with a clear message instructing the user to upgrade Magpie.

## Execution Mode

- **Local mode** (default): see [`modes/LOCAL.md`](modes/LOCAL.md)
- **Claw mode** (`MODE=claw`): see [`modes/CLAW.md`](modes/CLAW.md). Wrap the
  single `magpie benchmark` invocation with `exec_on_gpu`.

## Orchestrator Loop

```
PROCEDURE sweep_only():

  1. SETUP_LITE
     → Execute actions/setup.md
     → Detect environment, resolve $MODEL / $TP / $FRAMEWORK / $RUNNER_TYPE,
       install Magpie, verify SweepMatrix support.

  2. SWEEP
     → Execute actions/sweep.md
     → Generate sweep_matrix YAML using $EXTRA_*_ARGS + $SWEEP_CASES_YAML
       (or defaults), invoke `magpie benchmark` once, write
       sweep_report.json + results.tsv.

  3. REPORT
     → Print top-3 cases by tput_per_gpu and the path to results.tsv.
     → No KB ingest, no accuracy gate.
```

Steps 1 and 2 are mandatory; step 3 is a pretty-print convenience.

## Outputs

- `$SWEEP_DIR/sweep_report.json` — aggregated case-by-case results
- `$SWEEP_DIR/results.tsv` — flat TSV (CONC, ISL, OSL, output_tput, tput_per_gpu, TPOT_mean, TTFT_mean, success)
- `$SWEEP_DIR/case_*/inferencex_result.json` — raw InferenceX results per case
- `$SWEEP_DIR/server.log` — single shared server log (one launch for the whole sweep)

## Common Pitfalls

1. **Forgot `--disable-radix-cache`** in `EXTRA_SGLANG_ARGS` — KV cache leaks
   between cases and skews TPOT. Magpie's runner includes this flag by default
   only when `EXTRA_SGLANG_ARGS` does not already pin `mem-fraction-static`.
   If you supply a custom EXTRA flag set, ensure `--disable-radix-cache` is in it.
2. **Mixing server-side dimensions** (TP, mem-fraction, EXTRA_*_ARGS) in
   `cases` — `Magpie.modes.benchmark.config.SweepMatrix` will reject the
   benchmark config at parse time.
3. **`profiler.torch_profiler.enabled: true`** is not allowed in sweep mode —
   it is an explicit constraint enforced by Magpie. Use the full
   `inference-optimization-magpie` skill for profiling runs.

## See Also

- [`inference-optimization-magpie`](../inference-optimization/SKILL.md) — full DFS-guided autonomous optimization
- Magpie `sweep_matrix` reference: `Magpie/modes/benchmark/config.py::SweepMatrix`
