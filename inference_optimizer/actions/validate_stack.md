# `validate_stack` action playbook

## Purpose

Validate the **cumulative effect** of every KEEP'd entry on
`SharedState.optimization_stack` by running a single end-to-end Magpie
benchmark on a fresh server with all of those modifications applied at
once, then writing the measured gain into
`SharedState.cumulative_gain_validated`.

Per-round gains (recorded as `SharedState.cumulative_gain` after each
`backends` / `params` / `integrate` round) **do not compose linearly**:

* one round's "+3% over current_best" might collide with another
  round's "+2%" and end up as "+1.7%" together;
* kernel-opt patches that look great on a 10-token micro-benchmark
  often regress on the full Magpie workload;
* env-var combinations (e.g. AITER_USE_OOB + a custom NCCL config) can
  silently disable each other.

The final report quotes the validated number, not the per-round sum,
so `validate_stack` is the **only honest** measurement of where the
session has actually arrived.

## When the Coordinator demands it

After a successful `backends` / `params` / `integrate` round produces a
KEEP entry on `optimization_stack`, the Coordinator's
`_required_next_step()` adds a `TODO 5/5: validate_stack required`
entry to the per-tick checklist. Until Orchestration runs
`validate_stack`, every other action is denied with
`policy_denied{rule='validate_stack_required'}`. The TODO clears as
soon as the executor returns successfully.

## Inputs (task.params)

All optional; defaults below come from SharedState:

```yaml
config_path:        # absolute Magpie YAML path (default: baseline_config_path)
output_dir:         # workspace root (default: <SESSION_DIR>/runs/validate_stack/<task_id>/)
timeout_sec:        # subprocess hard cap (default: baseline cold/warm cap)
include_actions:    # list[str] — limit to specific stack entries by .action;
                    # default: all KEEP entries on optimization_stack
exclude_variants:   # list[str] — skip variants by .variant_name
```

The executor is **read-only** w.r.t. `optimization_stack` — it never
adds or removes entries; mutation is the Coordinator's job after
KEEP/REVERT propagates from explore/integrate actions.

## Outputs

Returns a dict shaped like a baseline result so the same parsing /
promotion path works:

```yaml
status:                "succeeded" | "failed"
output_throughput:     <float, tok/s/GPU on the validated stack>
ttft_mean_ms:          <float>
e2el_mean_ms:          <float>
accuracy:              <float, optional — only when RUN_EVAL=true>
validated_stack_len:   <int — len(optimization_stack) at run time>
applied_args:          <str — combined extra_sglang_args>
applied_envs:          <dict[str, str] — combined extra_envs>
workspace:             <str — Magpie workspace dir>
```

## Coordinator post-processing

When `delegated_result.kind=='validate_stack'` arrives with
`status=='succeeded'` and a positive `output_throughput`, the
Coordinator writes (in one atomic state.json save):

* `cumulative_gain_validated = (output_throughput - baseline_tput) / baseline_tput * 100`
* `cumulative_gain_validated_ts = <iso utc now>`
* `cumulative_gain_validated_stack_len = validated_stack_len`

These three fields live in `policy.CORE_STATE_FIELDS` so no LLM agent
can fake them via `UPDATE_STATE`.

## Failure handling

`status == "failed"` does **not** clear the validate_stack TODO — the
stack length didn't change so the trigger fires again next tick. To
bail out of an infinite validation loop:

1. operator manually sets `stop_reason` via the resume CLI, or
2. Robustness escalates after `crash_count >= 3` (existing rule).
