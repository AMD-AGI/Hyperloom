# Review Proposal

Use this action when Conductor asks Critic to review an Orchestration proposal,
Kernel response, generated patch, config change, dispatch fix, or
`integrate keep_proposed` decision.

## Expected Input

The packet may contain:

- `target_proposal_msg_id`.
- Proposal intent or Kernel response payload.
- Action metadata, including `family`, `owner`, `accuracy_risk`, and allowed
  side effects.
- Patch or diff content.
- Stated optimization goal and affected stack layer.
- Baseline benchmark result.
- After-change benchmark result.
- Accuracy gate result.
- Micro-benchmark data.
- Profile or dispatch evidence.
- Build, install, cache, and restart notes.
- Robustness findings or RCA summaries.
- KB evidence recalled by Critic or supplied by Conductor.
- Rollback plan.
- Prior attempts and session history.

Treat absent fields as missing evidence, not as passing evidence.

## Review Triggers

Critic review is required for:

- Orchestration `propose_action` where `accuracy_risk > 0`.
- Orchestration `propose_action` where `family in {"deep_kernel", "long"}`.
- Kernel `response` where `kind="integrate"` and `status="keep_proposed"`.

Critic does not block Robustness emergency actions such as `kill_task`,
`prune_branch`, or `force_dispatch`. Review those only after the fact as advice.

## Review Steps

1. Identify the claimed win.
   - What metric improved: `tok/s/GPU`, latency, memory, compile time, or kernel
     micro-benchmark?
   - Which layer changed: serving, framework/runtime, compiler, kernel/operator,
     communication, or GPU runtime?
   - Does the diff actually touch the claimed layer?

2. Check benchmark comparability.
   - Same model, GPU type/count, framework build, driver/ROCm version, launch
     script, TP/PP layout, batch/concurrency, ISL/OSL, dataset, warmup, sample
     count, and measurement window.
   - Same server lifecycle assumptions: clean restart when needed, no stale
     compiled cache, no leftover tuned config from another run.
   - Before/after results must include absolute values and gain percentage.

3. Check correctness evidence.
   - Accuracy gate passed, or Conductor provided an explicit waiver.
   - Numerical tolerance is appropriate for the modified layer.
   - Kernel or operator changes include correctness tests for the affected shapes.
   - Framework or dispatch changes prove the active path is the path measured.

4. Check patch risk.
   - Scope is minimal and relevant.
   - No unrelated refactors, hidden behavior changes, or debug-only shortcuts.
   - Build and install requirements are documented in the packet.
   - Rollback is straightforward.
   - Generated files, cache files, and `.best_config` changes are intentional.

5. Check cross-layer conflicts.
   - Serving/config changes do not invalidate compiler or kernel assumptions.
   - Framework dispatch changes do not bypass the optimized kernel.
   - Communication changes do not conflict with topology, TP/PP, or resource lane
     assumptions.
   - Robustness findings do not flag a known crash, regression, cache
     corruption, or accuracy failure for this patch family.

6. Decide the verdict.
   - `approve`: no blocker risk; include `predicted_gain_pct` when performance is
     relevant.
   - `reject`: do not dispatch; use when packet evidence or KB evidence proves
     the proposal is unsafe or invalid.
   - `redirect`: replace with `alternative_action`; use only when the action is
     registered and has the same owner.
   - `advise`: dispatch may proceed, but advice should be injected into the next
     Orchestration prompt.
   - `needs_review`: high-risk proposal has insufficient evidence or Critic is
     acting as mock/timeout/unavailable.

## Risk Types

Use one of these `risk.type` values when possible:

- `benchmark_missing`
- `benchmark_invalid`
- `accuracy_missing`
- `accuracy_failed`
- `patch_scope_mismatch`
- `active_path_unproven`
- `micro_only_evidence`
- `cache_or_rebuild_risk`
- `rollback_missing`
- `robustness_conflict`
- `cross_layer_conflict`
- `regression_risk`
- `insufficient_context`

## Severity

- `blocker`: must be fixed before dispatch.
- `major`: likely blocks dispatch unless Conductor explicitly accepts the risk.
- `minor`: does not block dispatch but should be tracked.

## Confidence

- `high`: evidence is complete and internally consistent.
- `medium`: enough evidence to decide, but some non-critical detail is missing.
- `low`: packet is sparse, contradictory, or depends on unverified assumptions.

## Output

Return a JSON object matching `review_verdict_schema` in
[references/verdict_schema.md](../references/verdict_schema.md).
