# Review Patch

Use this action when the Orchestration Core asks Critic to vote on an
optimization patch, config change, generated kernel, dispatch fix, rebuild
change, or keep/revert decision.

## Expected Input

The packet may contain:

- Patch or diff content.
- Stated optimization goal and affected stack layer.
- Baseline benchmark result.
- After-change benchmark result.
- Accuracy gate result.
- Micro-benchmark data.
- Profile or dispatch evidence.
- Build, install, cache, and restart notes.
- Triage findings or RCA summaries.
- Rollback plan.
- Prior attempts and session history.

Treat absent fields as missing evidence, not as passing evidence.

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
   - Accuracy gate passed, or the orchestrator provided an explicit waiver.
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
   - Triage findings do not flag a known crash, regression, cache corruption, or
     accuracy failure for this patch family.

6. Decide the vote.
   - `approval: true` only if there are no blocker objections.
   - `approval: false` if benchmark, correctness, deployability, or rollback
     evidence is missing or contradictory.
   - Use `warnings` for non-blocking follow-up items.

## Objection Types

Use one of these `type` values when possible:

- `benchmark_missing`
- `benchmark_invalid`
- `accuracy_missing`
- `accuracy_failed`
- `patch_scope_mismatch`
- `active_path_unproven`
- `micro_only_evidence`
- `cache_or_rebuild_risk`
- `rollback_missing`
- `triage_conflict`
- `cross_layer_conflict`
- `regression_risk`
- `insufficient_context`

## Severity

- `blocker`: must be fixed before approval.
- `major`: likely blocks approval unless the orchestrator explicitly accepts the
  risk.
- `minor`: does not block approval but should be tracked.

## Confidence

- `high`: evidence is complete and internally consistent.
- `medium`: enough evidence to decide, but some non-critical detail is missing.
- `low`: packet is sparse, contradictory, or depends on unverified assumptions.

## Output

Return a JSON object matching `patch_vote_schema` in
[references/verdict_schema.md](../references/verdict_schema.md).
