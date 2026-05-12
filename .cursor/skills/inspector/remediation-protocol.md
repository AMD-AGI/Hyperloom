# Remediation Protocol

Defines the four-level severity ladder, how extraction signals map to severity,
and the **binding obligations** the main agent must follow per verdict. This
protocol is referenced from [SKILL.md](SKILL.md) and from
[user-prompt-template.md](user-prompt-template.md). It exists because the
inspector cannot enforce anything by itself; the user prompt's contract on
top of this protocol is what makes the audit operational.

---

## 1. Severity Ladder

| Severity | Symbol | Aggregated verdict floor | Main-agent obligation |
|---|---|---|---|
| `info` | I | does not affect verdict | Log only. Continue. |
| `warn` | W | `WARN` | Log. May continue without remediation, but should mention the warning when narrating the next phase. |
| `block` | B | `BLOCK` | MUST execute the `remediation` field of every block-severity violation before reading the next phase's action `.md`. After remediation, MUST re-invoke inspector for the same phase. Do NOT advance until verdict is `PASS` or `WARN`. |
| `fatal` | F | `FATAL` | MUST rollback the phase (revert any patches, kill any servers started, remove the phase's RESULT_DIR contents if integrate-style). MUST mark the run invalid in the report. MUST jump to the target skill's REPORT phase regardless of remaining DFS budget. Do NOT attempt to remediate. |

Verdict aggregation rule: the verdict equals the **worst** severity present in
`violations[]` (FATAL > BLOCK > WARN > none -> PASS). See
[audit-report-schema.md](audit-report-schema.md) Section 3.

---

## 2. Mapping from Extraction Signals to Severity

The extraction protocol assigns a `modality` (MUST / SHOULD / MAY / UNVERIFIED)
to every expected item. Combined with the audit outcome, severity is computed
as follows:

| modality in manifest | passed | failed (not found / count too low) | unresolvable env var |
|---|---|---|---|
| `MUST` (default) | record in `passes[]` | `block` | record in `unverified[]` (NOT a violation) |
| `MUST` with Iron-Rule keyword | record in `passes[]` | `fatal` | record in `unverified[]` |
| `SHOULD` | record in `passes[]` | `warn` | record in `unverified[]` |
| `MAY` | record in `passes[]` | `info` (not a violation; logged only) | record in `unverified[]` |
| `UNVERIFIED` | record in `passes[]` | record in `unverified[]` | record in `unverified[]` |

### Iron-Rule trigger (for `fatal`)

A failed `MUST` is escalated to `fatal` only if **all** of these conditions
hold (verified mechanically against the action `.md` `source_quote`):

1. The `source_quote` text contains at least one of:
   `Violation = invalidation`, `Iron Rule`, `MUST NOT`, `mandatory.{0,30}invalid`,
   or matches `\bIR-\d+\b`.
2. The matching candidate is inside a section whose heading text contains
   `Iron Rule` or `Mandatory` (case-insensitive).
3. The expected item is in the `expected_tool_calls` or `expected_artifacts`
   bucket (state-assertion failures never escalate to fatal; they are at most
   `block`).

If any of the three conditions fails, the violation stays at `block`.

### Synthetic warnings

The inspector adds one or more synthetic violations in these cases.
Several of them encode hard lessons from the 2026-04-21 Qwen3-30B-A3B run
where BLOCK violations were structurally invisible to an inspector that
relied on transcript markers and free-form verdict prose.

| Synthetic violation id | Severity | Trigger |
|---|---|---|
| `extraction_low_confidence` | `warn` | `len(unverified) >= 0.5 * total_expected_items` AND there is at least one `MUST` candidate |
| `inspector_skipped_previous_phase` | `warn` | `next_checkpoint.should_invoke_inspector_after` from the previous audit does not match the current `phase` |
| `transcript_unreadable` | `block` | `find_transcript.py` returned no candidate file |
| `manifest_extraction_failed` | `info` | The action `.md` could not be parsed at all (missing or empty); cannot judge the phase, so do not block |
| `transcript_too_short` | `block` | Transcript file exists but has fewer than `MIN_TRANSCRIPT_LINES` lines (default 5). Without this rule, post-summary transcript collapse causes every Channel-A check to record `unverified` and the verdict silently slides to PASS. Source: [`scripts/semantic_rules.json`](scripts/semantic_rules.json) and SKILL.md §S3 step 2. |
| `not_integrated_after_dfs` | `block` | `phase ∈ {DFS_LOOP_*, SWEEP, REPORT}` AND `$RESULT_DIR/kernel_results.json::integration_status == "NOT_INTEGRATED"`. Encodes IR-3 of `inference-optimization`: integration is mandatory after every kernel-opt round. Source: [`scripts/semantic_rules.json`](scripts/semantic_rules.json). |
| `unprocessed_action_stack` | `block` | `phase ∈ {SWEEP, REPORT}` AND `len(action_stack.dfs_order) > len(action_stack.completed_actions)`. Catches actions that fell out of the DFS stack without being executed or explicitly deferred. |
| `missing_accuracy_reference` | `block` | `phase ≥ BASELINE` AND `$RESULT_DIR/accuracy_reference.json` is missing. Without it, every downstream accuracy comparison is unfalsifiable. |
| `iron_rule_must_unsatisfied` | `fatal` | At least one Iron-Rule-derived MUST tool_call has `count == 0` in a healthy transcript (≥ MIN_TRANSCRIPT_LINES). The transcript-too-short rule pre-empts this one to avoid double-counting transcript collapse as an Iron-Rule miss. |

---

## 3. Main Agent Obligations Per Verdict

The user prompt template binds the main agent to obey the following table.
Without that binding, this protocol is descriptive only.

The agent reads the on-disk `$RESULT_DIR/.audit/<PHASE>_<ts>.json` (path
comes from the inspector's one-line `[Inspection] ... -> <path>` ack)
instead of parsing a fenced JSON block from the chat. The inspector's
single ack line is the only chat-visible audit footprint per phase.

### `PASS`
- No additional chat output. The inspector's `[Inspection] ... verdict=PASS ...`
  line is sufficient evidence.
- Continue executing the target skill.

### `WARN`
- No additional chat output is required. If any `warn` violation has a
  `remediation` field that is cheap (estimated `<5min`), the agent SHOULD
  attempt remediation opportunistically before proceeding, but is not
  required to. Cheap remediations are framed as the agent's natural next
  step (e.g. "Re-running accuracy eval to refresh the reference."), not as
  "fixing inspector warning N".
- Continue executing the target skill.

### `BLOCK`
The contract is strict and ordered, but framed as natural work:

1. `Read` `$RESULT_DIR/.audit/<PHASE>_<latest-ts>.json` from the path the
   inspector printed. Do **not** narrate "inspector found N block violations
   in phase X"; just open the file.
2. For each `block` violation in `violations[]`, in array order:
   - Read the `remediation` field.
   - Execute it as the next natural step in the run (e.g. "Running GSM8K
     evaluation."). The remediation field is intentionally a self-contained
     shell or tool invocation; the agent MUST NOT invent its own remediation
     and MUST NOT narrate the violation id, source quote, or severity in
     the chat.
   - If a remediation requires environment context that is missing
     (`run_env_unresolved` mentions the needed var), STOP, ask the user
     using business-language ("I need the path to the calibration dataset
     to continue."), then resume.
3. After all block remediations have been attempted, **re-invoke inspector**
   for the same phase. Do NOT proceed to the next phase based on the
   agent's own judgment that the issue is resolved.
4. Loop until inspector returns `PASS` or `WARN` for this phase.
5. Maximum 3 remediation iterations per phase. If iteration 3 still returns
   `BLOCK`, escalate by:
   - Adding a synthetic `fatal` violation `id="remediation_loop_exhausted"`
     to a final inspector audit.
   - Treating the phase as `FATAL` (see below).

### `FATAL`
1. `Read` the on-disk audit_report.json once to extract the `fatal`
   violation's `source_quote` and observed value; use them to compose a
   single business-language stop sentence (see SKILL.md §S5c). Do not echo
   verdict id / severity / channel jargon.
2. Rollback obligations (in order):
   - Revert any kernel patches applied during this phase (per IR-3 / IR-6
     style rules in inference-optimization; analog in other skills).
   - Kill any inference / training servers started during this phase
     (`pgrep -f` patterns from the target skill's process-management section).
   - Remove or rename the phase's `$RESULT_DIR/<phase>` subdirectory so it
     cannot be mistaken for valid baseline data.
3. Mark the entire run as invalid in the orchestrator state
   (`state.run_invalid = true`).
4. Skip all remaining DFS / loop work. Jump directly to the target skill's
   REPORT phase. The REPORT must include:
   - A pointer to the fatal on-disk `audit_report.json` (path) and the
     full `_state.json::history` for the run.
   - A note that the run was inspector-invalidated.
   - Whatever partial results are still trustworthy (typically nothing past
     the previous PASS audit).
5. Do NOT re-invoke inspector after fatal. The audit is final.

---

## 4. Re-invocation Discipline

When the main agent re-invokes inspector after remediation, it MUST:

- Pass the **same** `PHASE_NAME` and `PHASE_ACTION_FILES` as the previous
  invocation. Adding a new action file mid-loop is allowed only if a
  remediation explicitly required it (e.g. realising an Iron-Rule
  cross-reference was missed).
- Pass an updated `RUN_ENV` only if env vars actually changed during
  remediation.
- NOT delete the previous attempt's on-disk
  `$RESULT_DIR/.audit/<PHASE>_<ts>.json`. Each re-audit produces a new
  timestamped file; the older files are the per-attempt history. The
  sentinel `_state.json` is overwritten in place but its `history` array
  preserves a one-line entry per attempt.
- NOT echo "re-invoking inspector, attempt N/3" or any similar status
  line into the chat (implicit-mode rule). The attempt count is
  reconstructable from
  `len([h for h in _state.json.history if h.phase == PHASE])`. Long-run
  analysis tooling reads the sentinel + per-phase report files instead of
  greping the transcript.

This discipline is what allows long-run analysis to count remediation cycles
per phase from on-disk artifacts alone.

---

## 5. Common Remediation Templates

These are reference templates the inspector embeds in `remediation` fields
when it has enough context. They are guidance for the inspector author, not
runtime code.

| Violation kind | Template `remediation` field |
|---|---|
| Missing `eval_summary_<task>.json` | `EVAL_TASK=<task> NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL RESULTS_DIR="$RESULT_DIR/eval_<task>_<ctx>" bash $SKILL_ROOT/scripts/eval_accuracy.sh` |
| Missing `baseline_*.json` | `bash $SKILL_ROOT/scripts/run_baseline.sh` (after ensuring `kill_server` and `check_gpu_memory`, per IR-4) |
| Missing TraceLens output | `bash $SKILL_ROOT/scripts/run_profile.sh; then run TraceLens pipeline per actions/profile.md` |
| Missing `kb_query.py` invocation | `python3 $SKILL_ROOT/kb/kb_query.py --model "$MODEL_NAME" --top-k 20` |
| Missing `kb_ingest.py` invocation (post-action) | `python3 $SKILL_ROOT/kb/kb_ingest.py --category <category> --model "$MODEL_NAME" --action "..." --lesson "..." --tags <tags> --gain <pct> --status <status>` |
| Server still running when not expected | `kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null; sleep $SERVER_KILL_WAIT_S` (per IR-5) |
| Missing `results.tsv` after sweep | `bash $SKILL_ROOT/scripts/run_sweep.sh` |
| Missing `optimization_report.md` | `Follow actions/report.md from start; do not skip the kb_ingest.py call.` |
| GEAK candidates not all submitted in parallel (IR-1) | `For each remaining candidate kernel, immediately call geak_create_task in parallel (do NOT serialize). Total parallel submissions must equal GEAK_TOP_CANDIDATES.` |
| Patch applied but no re-baseline (IR-3) | `Run bash $SKILL_ROOT/scripts/run_baseline.sh with RESULT_DIR="$RESULT_DIR/optimized_<kernel_name>"; then re-evaluate accuracy.` |

When the inspector cannot infer a precise template (because the action `.md`
expects something custom), it should still emit a remediation field whose text
quotes the relevant sentence from the action `.md`, prefixed with
`Re-perform the step described at <action_md>:<line_no>: "..."`. This lets the
main agent at least re-read the source.

---

## 6. Anti-Patterns (binding negatives)

The following are explicitly forbidden behaviors for both the inspector and
the main agent under this protocol:

1. **Inspector inflates severity to be "safe".** Severity must follow the
   modality table mechanically; no defensive escalation.
2. **Main agent self-approves a `BLOCK`.** The agent cannot decide that a
   block violation is "actually fine" and skip remediation. The only escape
   is the 3-iteration cap that escalates to fatal.
3. **Main agent silently skips re-invoking inspector after remediation.**
   Even if the agent is confident the fix worked, the next phase MUST NOT
   start until inspector confirms `PASS` / `WARN`.
4. **Inspector emits anything other than the one-line ack into the chat.**
   A reply that contains a fenced `audit_report.json` block, "## Audit
   verdict:" headers, "## Required remediations" lists, or any
   `=== INSPECTOR_... ===`-style markers is malformed and should be
   re-emitted as a single `[Inspection] ...` line. The full report
   belongs only on disk under `$RESULT_DIR/.audit/`. If you find yourself
   wanting to summarise the verdict in prose, the right action is to make
   the one-line ack carry the missing field, not to add a second line.
5. **Main agent argues with the audit.** No prose like "I disagree with the
   inspector's finding". The audit is authoritative within its scope; if the
   user disagrees, they update extraction-protocol.md or the action `.md`,
   not the live audit.
6. **Inspector tries to remediate itself.** Inspector is read-only. It must
   never call `Shell`, `Edit`, `Write`, or any tool that mutates state. Even
   for a one-line fix that would save tokens.
7. **Main agent post-edits `compute_verdict.py` output.** The JSON
   produced by `scripts/compute_verdict.py` is **frozen**.
   `scripts/emit_audit_report.py` writes the verdict block verbatim into
   the on-disk `audit_report.json`; editing the on-disk file after the
   emitter wrote it (or editing `_state.json` to mask a verdict) is the
   same anti-pattern. The forbidden mutations cover the `verdict`,
   `verdict_summary`, `violations`, `passes`, `unverified`, and
   `verdict_source` fields. Adding free-text justification keys like
   `because`, `deferred`, or `acceptable` to the observations file is also
   forbidden — the `_validate_observations` guard in `compute_verdict.py`
   rejects these keys upstream. The only legitimate response to "the
   verdict is wrong" is to update
   [`scripts/semantic_rules.json`](scripts/semantic_rules.json) or the
   extraction protocol BEFORE the next inspector invocation. This
   anti-pattern is what allowed the 2026-04-21 Qwen3 run's BLOCK
   findings to be re-narrated as "WARN per skill allowance" in the same
   audit reply.

---

## 7. Failure of the Protocol Itself

If the inspector cannot run at all (e.g. `find_transcript.py` errors,
`scripts/parse_action_outputs.py` raises, target skill path does not exist),
the inspector MUST emit a minimal `audit_report.json` with:

```json
{
  "verdict_source": "compute_verdict.py|self_failure",
  "phase": "<PHASE_NAME>",
  "verdict": "BLOCK",
  "violations": [
    {
      "id": "inspector_self_failure",
      "severity": "block",
      "channel": "content",
      "expected": "inspector audit completes successfully",
      "observed": "<short error message>",
      "modality_in_manifest": "MUST",
      "source_lines": [],
      "source_quote": "(synthetic)",
      "remediation": "Read inspector logs above. Fix the underlying error (most often: target skill path or RUN_ENV not provided correctly), then re-invoke inspector."
    }
  ],
  "passes": [],
  "unverified": [],
  "extraction_diagnostics": {"candidates_from_regex": 0, "candidates_kept_after_classification": 0, "modality_promotions": 0, "modality_demotions": 0, "regex_anchors_diff_summary": "extraction did not run"},
  "next_checkpoint": {"should_invoke_inspector_after": "<PHASE_NAME>", "reminder_text": "Inspector itself failed; fix the inspector invocation and retry the same phase."}
}
```

This rule guarantees a malformed inspector run cannot silently let the main
agent skip a phase: failure of inspector = `BLOCK`, with remediation pointing
at the inspector itself.
