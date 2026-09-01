# `integrate_patch` action playbook

## Purpose

Take a completed specialist's worktree patches, apply them to the
framework source roots, restart the inference server, run the throughput
+ accuracy gate, and either KEEP (advance `optimization_stack`) or
REVERT (roll the source tree back). This is the `integrate` step of the
Arbor optimization loop (arXiv:2606.12563) — Arbor being the research name
for this orchestration.

`integrate_patch` is a **deterministic Python executor**, not an LLM
sub-agent. It is the orchestrator's serving-lane-locked integration
point — only this action is allowed to mutate
`INFERENCEX_PATH` / `framework_source_roots`. Specialists produce
patch *files*; this action produces *outcomes*.

## When to delegate

* Phase is `FRAMEWORK_AGENT` (per `PHASE_ALLOWED_ACTIONS` in
  `src/hyperloom/orchestrator/phases/machine_state.py`).
* A specialist task has emitted `specialist_done.patches_written` with
  at least one patch path inside its worktree.
* **Required** — `params.specialist_task_id` is set and the Critic has
  recorded a verdict in {`approve`, `advise`} for that specialist
  (`SharedState.specialist_patch_verdicts`). Without it PolicyGate denies
  the intent with `rule="integrate_patch_requires_critic_verdict"`. The sole
  exemption is the Coordinator-internal `enablement_launch_only` build probe,
  which applies no patch and is never LLM-proposable.

## Who delegates this action

* **Orchestration** only. Robustness can trigger recovery via
  `delegate(recover)`; it does not directly integrate patches.

## Inputs (task.params)

| Key                  | Type     | Required | Description |
|----------------------|----------|----------|-------------|
| `specialist_task_id` | string   | yes      | Task id of the specialist whose worktree carries the patches. |
| `patches`            | list[str]| no       | Explicit patch path list (relative to specialist workspace or absolute under `SESSION_DIR`). Defaults to `specialist_done.patches_written`. |
| `config_changes`     | object   | no       | `env_var -> value` map layered onto the server-launch env before restart. Reverted with the patches on gate failure. |
| `keep_threshold_pct` | float    | no       | KEEP threshold over baseline_tput; defaults to the session's decaying per-cycle bar (read from `SharedState`). |
| `accuracy_baseline`  | float    | no       | Baseline accuracy score (0-1). Backfilled from `SharedState.baseline_accuracy` when omitted; `<= 0` skips the gate. |

## EMIT format

```
delegate{
  action_name = 'integrate_patch',
  params = {
    specialist_task_id  = '<specialist-task-id-from-tasks-table>',
    // patches / config_changes / keep_threshold_pct / accuracy_baseline optional
  },
  idempotency_key = 'integrate-patch-<specialist-task-id>',
}
```

## Sequence

1. Read `runs/specialist/<task_id>/specialist_done.json` and pull
   `patches_written` (when params.patches omitted).
2. Acquire `server_lifecycle + workspace_mutation + benchmark_lane`
   triple-lock.
3. `git apply -p1` each patch against the framework source roots
   (`INFERENCEX_PATH` or `framework_source_roots[0]`). On conflict,
   retry once with `git apply -3 -p1`. On second failure, mark the
   patch as REVERT-immediate (record `apply_failed`).
4. Layer `config_changes` onto the server-launch env.
5. Stop the existing server (`pkill -9 -f "VLLM::EngineCore|VLLM::Worker"`
   or sglang equivalents); wait for VRAM to drain.
6. Launch via `$ARBOR_LAUNCH_SCRIPT` (or the installed scripts);
   health-check until ready (≤ 20min for large MoE models).
7. Run the Magpie throughput benchmark + GSM8K accuracy eval, graded via
   the shared `_accuracy_gate.parse_eval_results(...)` +
   `_accuracy_gate.accuracy_passed(...)` helpers.
8. Decide:
   - KEEP — bench tput ≥ grading anchor * (1 + keep_threshold_pct/100) AND
     accuracy did not drop more than 0.05 absolute. Append the patch + config_changes to
     `SharedState.optimization_stack`, update `current_best`, restamp
     `cumulative_gain_validated`.
   - REVERT — any gate failure. `git checkout` the framework source
     roots, drop `config_changes`, restart with the previous config,
     record evidence in `last_action_failures`.

## Output (delegated_result.payload)

```
{
  "status":          "kept" | "reverted" | "apply_failed",
  "output_throughput": <float | null>,
  "delta_pct":         <float | null>,
  "accuracy_pass":   <bool | null>,
  "patches_applied": ["patches/001_cuda_graph_fix.patch", ...],
  "config_changes_applied": {"VLLM_USE_AITER": "1"},
  "reason":          "<one-line summary>",
  "specialist_task_id": "<id>",
  "evidence":        {"bench_log": "...", "gate_log": "..."}
}
```

## Idempotency & resume

The Coordinator dedups by `idempotency_key`. A REVERT'd patch can be
re-proposed only if the specialist emits a fresh patch (different
worktree task id). Resume is safe because `git apply` is detectable by
hash and `optimization_stack` is the authoritative ledger.

## Failure modes

| Symptom | Cause | Mitigation |
|---|---|---|
| `git apply` rejects | base HEAD drifted under specialist | reject patch, record `apply_failed`, REVERT path |
| server fails to start | broken patch | `git checkout` + restart prior config, mark REVERT |
| accuracy drops >0.05 absolute | semantic regression | REVERT, record into pitfalls KB |
| throughput unchanged within keep_threshold | no-op patch | REVERT, log low-confidence specialist for the gap |
