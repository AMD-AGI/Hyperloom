# Semantic Rules

Inspector verdicts are computed by `scripts/compute_verdict.py` from three
inputs:

1. The expectation **manifest** (built by `extraction-protocol.md`).
2. The agent's **observations** dump (built in `SKILL.md` step S5a).
3. This **semantic rules** pack (`scripts/semantic_rules.json`).

Manifest-derived violations cover the "did the action's MUST/SHOULD line up?"
question. Semantic rules cover the **structural**, **cross-file**,
**cross-phase** failure modes that no single action `.md` file alone can express
— and that prior runs have proven the inspector will otherwise miss.

This file documents what each rule does, when to add a new one, and the
guardrails for editing them.

## File location

- Rule pack: `.cursor/skills/inspector/scripts/semantic_rules.json`
- Rule schema: enforced by `scripts/compute_verdict.py::_check_passes`
- Documentation: this file (`semantic-rules.md`)

## Rule schema

```json
{
  "id":       "snake_case_stable_id",
  "description": "Why this rule exists, citing the failure mode it catches.",
  "when":     {"phase_in": ["BASELINE", "DFS_LOOP_*", "SWEEP", "REPORT"]},
  "check":    {"kind": "<one_of_check_kinds>", "...": "..."},
  "verdict":  "block | fatal",
  "channel":  "content | file | state",
  "expected": "Plain-English statement of what should be true.",
  "remediation": "Concrete one-paragraph instruction the agent must follow."
}
```

`when.phase_in` accepts shell-style glob patterns (`fnmatchcase` semantics),
e.g. `"DFS_LOOP_*"` matches `DFS_LOOP_3`, `DFS_LOOP_KERNEL_OPT`, etc.

### Supported `check.kind` values

| `kind` | Required fields | Fires when… |
|---|---|---|
| `transcript_lines_lt` | `value` (int) | The recovered transcript JSONL has fewer than `value` lines |
| `artifact_missing` | `artifact_id` | The artifact observation is absent or `exists=false` |
| `artifact_field_equals` | `artifact_id`, `field_path`, `value`, `trigger_if_missing` | A dotted JSON path inside the observed artifact equals `value` |
| `artifact_field_lt` | `artifact_id`, `field_path`, `value`, `trigger_if_missing` | The numeric value at `field_path` is less than `value` |
| `artifact_list_len_gt_field_len` | `artifact_id`, `a_path`, `b_path`, `trigger_if_missing` | `len(blob[a_path]) > len(blob[b_path])` (e.g. dfs_order vs completed_actions) |
| `iron_rule_must_unsatisfied` | `min_transcript_lines` (int, default 5) | At least one Iron-Rule-derived MUST tool_call has count==0 in a healthy transcript |

`trigger_if_missing` controls behavior when the artifact observation is absent
(usually because `must_be_nonempty=false` or the path isn't required for this
phase): `true` makes a missing artifact fire the rule, `false` is the safer
default and treats missing as "don't fire".

`artifact_id` must match an `id` produced by `extraction-protocol.md`
(typically `result_dir_<filename>_json`). Check
`/tmp/inspector_manifest_<phase>.json` to find available ids.

`field_path` is dotted notation: `"integration_status"`,
`"results.peak.tok_per_s"`, `"dfs_order.0.action_id"`. List indices use
integers (no `[0]` brackets).

## Current rules

### `transcript_too_short` (block)

When `find_transcript.py` returns a file but `wc -l` of it is below
`MIN_TRANSCRIPT_LINES` (default 5), Channel A (transcript-grep audit) cannot
verify *anything*. Without this rule, the inspector marks every Channel-A
expectation as `unverified` and the verdict slides to PASS. This is the exact
mechanism that hid the 2026-04-21 Qwen3-30B-A3B run's missing integration step.

### `not_integrated_after_dfs` (block)

After kernel-opt finishes, IR-3 of the inference-optimization SKILL says
integration is MANDATORY. The skill records its status in
`$RESULT_DIR/kernel_results.json::integration_status`. If that field equals
`"NOT_INTEGRATED"` while the orchestrator has already moved on to SWEEP or
REPORT, the integration step was skipped and end-to-end accuracy was never
re-validated.

### `unprocessed_action_stack` (block)

`$RESULT_DIR/action_stack.json` carries two parallel lists:

- `dfs_order` — what the BUILD_ACTION_STACK phase decided to try
- `completed_actions` — what actually ran

By the time the run reaches SWEEP / REPORT, every action must be in either
`completed_actions` or `deferred_actions`. A length mismatch
(`len(dfs_order) > len(completed_actions)`) means some actions silently fell
out of the stack, which is what happened to the four server-param actions in
the 2026-04-21 run.

### `missing_accuracy_reference` (block)

`actions/baseline.md` step 5 produces `$RESULT_DIR/accuracy_reference.json` —
the GSM8K reference score that every later DFS_LOOP and SWEEP must compare
against. If this file does not exist by the time BASELINE is "complete", every
downstream accuracy comparison is unfalsifiable, so the rule fires for every
phase from BASELINE onward.

### `iron_rule_must_unsatisfied` (fatal)

For each IR-N candidate that `parse_iron_rules.py` extracts as a `MUST`
tool_call, this rule checks if the corresponding observation reports
`count == 0` *and* the transcript is healthy (≥ MIN_TRANSCRIPT_LINES, so
absence is not an artifact of transcript collapse). If so, the iron rule
was actively skipped — escalate to FATAL.

## Adding a new rule

1. **Identify the failure mode in retrospect.** A rule should be born of a
   specific past run where the inspector returned PASS/WARN but a structural
   defect was visible on disk.
2. **Pick the smallest deterministic signal.** Prefer artifact field checks
   over transcript heuristics. Avoid anything that requires LLM judgment.
3. **Pick the right verdict level.**
   - `block` — orchestrator must not progress; defect is unambiguous and the
     fix is local.
   - `fatal` — Iron-Rule violation; the entire run is invalidated.
4. **Add the JSON entry**, then add a short paragraph here under "Current
   rules" with: link to the historical incident, the data field/path it
   inspects, and the remediation expected of the agent.
5. **Validate against the historical run** (`replay-validate` step in the
   inspector hardening plan): the rule should fire on the failed run and
   stay silent on a clean run.

## Anti-patterns when editing rules

- **Do NOT** soften a rule (raise threshold, narrow `phase_in`) just because
  the current run failed it. Fix the run, not the rule.
- **Do NOT** add free-text justifications inside rule entries; the JSON is
  consumed by `compute_verdict.py`, not by humans, and must stay declarative.
- **Do NOT** introduce LLM calls or shell-out from inside a check; the whole
  point of this layer is determinism.
