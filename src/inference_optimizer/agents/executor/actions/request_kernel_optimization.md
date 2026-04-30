# Request Kernel Optimization (guided / marathon)

**Trigger**: `state.execution_mode in {guided_kernel_opt, marathon_multi_agent}`
AND `baseline_tput > 0` AND no in-flight kernel work (check inbox for
recent `response{kind=...}` from `from_agent="kernel"`).

> Plan A: kernel-opt is **owned by the kernel agent**. You no longer
> emit `delegate(action_name="kernel_opt")` or
> `delegate(action_name="integrate")` — PolicyGate will reject those.
> The only path is `request{target_agent="kernel", kind=...}` and
> waiting for the matching `response{in_reply_to=..., kind=..._done}`.

## Pipeline at 30,000 ft

```
profile (you) → request(select_kernels) → response(select_kernels_done)
                                                 ↓
                          (you pick from candidates)
                                                 ↓
                          → request(run_optimization) → response(optimization_done)
                                                              ↓
                                                 (you pick patches to apply)
                                                              ↓
                          → request(apply_patch) → response(patch_applied)
                                                              ↓
                                       (you write update_state(current_tput=X))
                                                              ↓
                                              (loop back to profile)
```

Each arrow is one round-trip through the bus. Wait for each response
before issuing the next request.

## Step 1 — Capture (or reuse) a profile (still your job)

If your inbox has no recent `event{kind=profile_done}` event:

```json
{
  "intent_type": "delegate",
  "payload": {
    "action_name": "profile",
    "params": {},
    "predicted_gain_pct": 0.0,
    "reason": "need fresh trace to feed kernel agent's select_kernels"
  }
}
```

Profile soft-skip (`kind=profile_skipped`) → proceed to Step 2 anyway,
the kernel agent will reuse the existing trace under
`$SESSION_DIR/results/<task_id>/`.

## Step 2 — Ask the kernel agent which kernels to optimize

Once you have the trace path (from `event{kind=profile_done,
trace_path=...}`):

```json
{
  "intent_type": "request",
  "payload": {
    "target_agent": "kernel",
    "kind": "select_kernels",
    "params": {
      "trace_path": "<from profile_done event>",
      "top_n": 5
    },
    "reason": "executor needs candidate list before committing to optimization budget"
  }
}
```

Wait for `response{from_agent=kernel, in_reply_to=<your request msg_id>,
kind=select_kernels_done}`. The `result.candidates[]` array carries the
kernel agent's picks with `gpu_pct` + `rationale` per entry.

## Step 3 — Pick which candidates are worth optimizing

This is your decision (kernel agent only suggested). Look at:

- `gpu_pct` — kernels under 5% are usually not worth GEAK budget
- `framework` — `triton` / inductor cache is high-yield; `aiter::*` only
  works if user-provided source is on disk
- KB recall (ask sage if unsure) — has this kernel class regressed
  before?

Compose a subset (the kernel agent's full candidate list, or a
filtered version) and request the optimization round:

```json
{
  "intent_type": "request",
  "payload": {
    "target_agent": "kernel",
    "kind": "run_optimization",
    "params": {
      "selected_kernels": [
        {"name": "triton_red_fused_sum_42", "source_path": "/tmp/torchinductor_root/abc/123.py"},
        ...
      ],
      "backends": ["geak", "codex"],
      "prompt_file": "<path to your optimization prompt or default>"
    },
    "reason": "executor approved 3 of 5 candidates for kernel optimization"
  }
}
```

Wait for `response{kind=optimization_done}`. The `result.patches[]`
array carries one entry per (candidate, backend) pair that succeeded.

## Step 4 — Pick which patches to apply

This is again your decision. Evaluate:

- `predicted_gain_pct` — kernel agent's estimate; weight against KB
  Brier history for that kernel class
- `backend` — if both GEAK and OOB produced a patch for the same
  candidate, you can apply either or both (apply_patch will sequence them)
- `accuracy_risk` — IR-3 ensures apply_patch will run accuracy_gate; a
  candidate marked high-risk in `result.warnings` should be applied
  alone so any revert is unambiguous

Send the apply request:

```json
{
  "intent_type": "request",
  "payload": {
    "target_agent": "kernel",
    "kind": "apply_patch",
    "params": {
      "selected_patches": [
        {
          "candidate_id": "geak_triton_red_42",
          "patch_path": "/path/to/optimized_kernel.py",
          "best_config_path": "/path/to/best_config.json"
        }
      ]
    },
    "reason": "applying winning candidate; expect ~7% gain on this model class"
  }
}
```

Wait for `response{kind=patch_applied}`. If `result.reverted=True`, the
kernel agent already rolled back — no action needed beyond logging.

## Step 5 — Write the new measurement

`response{kind=patch_applied}` carries `result.current_tput`. The
kernel agent **cannot** emit `update_state` (PolicyGate denies). YOU
emit it:

```json
{
  "intent_type": "update_state",
  "payload": {
    "changes": {
      "current_tput": <result.current_tput>,
      "current_action": "kernel_opt_loop_idle"
    },
    "rationale": "kernel agent applied patch; new tput from re-baseline"
  }
}
```

The conductor's `_maybe_recompute_gain` auto-derives `cumulative_gain`.

## Looping

After Step 5, loop back to Step 1 (fresh profile) IF cumulative_gain
hasn't plateaued AND time_left ≥ 30 minutes. Each iteration the hot
kernel set typically shifts.

## Soft lane coordination

While the kernel agent has an `apply_patch` in flight (you can tell
because the latest response was `optimization_done` but no
`patch_applied` yet, OR you sent `apply_patch` request and haven't
received response), **DO NOT** issue `delegate(bench_runner)` — the
kernel agent's `apply_patch.sh` will restart the server and your bench
will hit the dead window. Wait for `patch_applied` first.

## Mode-specific notes

### `guided_kernel_opt`

Single kernel-opt loop typically; budget tight. Cap `top_n` at 3.

### `marathon_multi_agent`

Multiple loops; can use `propose_action` (not `delegate`) for
high-risk patches to invite parliament review (critic + sage vote)
before committing to apply_patch.

## Failure modes

| Symptom | Recovery |
|---|---|
| `response{status=failed, reason="trace_not_found"}` | Re-issue `delegate(profile)` then retry select_kernels |
| `response{kind=optimization_done, n_succeeded=0}` | All backends failed; pivot to `delegate(param_sweep_run)` for shallow gains |
| `response{kind=patch_applied, reverted=True}` | Accuracy gate rolled back; record kernel class in KB and skip in future loops |
| No response within ~5 minutes (request lost) | Check `bus.tail topic=alert` for kernel agent issues; if stuck emit `alert{severity=medium}` and pivot |

## DON'T do these

- DON'T emit `delegate(kernel_opt)` or `delegate(integrate)` — PolicyGate
  will return `policy_denied{rule=kernel_owned_by_kernel_agent}`.
- DON'T issue a fresh `request(select_kernels)` while a previous
  optimization round is still in flight — the kernel agent serializes
  its own work but your wasted request burns latency.
- DON'T modify any kernel source via the `Edit` tool between Step 2 and
  Step 4 — that's IR-2 (recommended). The kernel agent is the only
  writer in the kernel-opt window.
