# Inspector Skill — README

Generic, phase-level execution auditor for any other Cursor skill. After a
phase of a target skill ends, the inspector reads that skill's action
markdown files at runtime, derives an expectation manifest of mandatory tool
calls / output files / state assertions, audits the agent transcript JSONL
plus on-disk artifacts, and emits a structured `audit_report.json` with a
verdict in `{PASS, WARN, BLOCK, FATAL}`. The user prompt that started the
run binds the main agent to remediate every `block` / `fatal` violation
before advancing.

The inspector is read-only and same-agent. It does not modify the audited
skill, does not spawn subagents, and does not run as a hook.

---

## When to use this skill

- Running a long, multi-phase skill (e.g. `inference-optimization`,
  `training-optimization`, `mlperf-optimization`,
  `marathon-inference-optimization`) where the agent might silently skip a
  mandatory step.
- Building confidence that a "PASS" outcome actually executed every step
  the action `.md` claims it does.
- Detecting compaction-induced omissions in long-running optimization runs.

## When NOT to use this skill

- Single-shot tasks (one Q&A, one file edit). Overhead is not justified.
- Skills whose action `.md` files are extremely loose / English-only (no
  fenced commands, no env-var paths). Inspector will degrade to mostly
  `unverified` entries.
- When you want enforcement that survives the agent forgetting to invoke
  it. For that you need a hook, not a same-agent skill (see "Upgrading"
  below).

---

## Quick start

1. Pick a target skill, e.g. `.cursor/skills/inference-optimization`.
2. Copy the appropriate filled example from
   [user-prompt-template.md](user-prompt-template.md). Replace the slot
   values (`MODEL_NAME`, `MODEL`, `TP`, etc.) with your run inputs.
3. Send the resulting prompt to the agent. The "INSPECTOR BINDING CONTRACT"
   block in the prompt makes the inspector enforceable.

That's it. The agent will read the target skill's `SKILL.md`, execute its
phases, and after each phase will read this skill's [SKILL.md](SKILL.md) and
run the 5-step audit.

---

## How it works (1-minute version)

```mermaid
flowchart LR
  prompt[User prompt with INSPECTOR BINDING CONTRACT] --> agent
  agent[Main agent] -->|runs phase X| target[Target skill: SETUP, CLASSIFY, BASELINE, ...]
  target -->|phase X done| inspect[Read inspector SKILL.md]
  inspect --> S1[S1 bind context]
  S1 --> S2[S2 build expectation manifest from actions/X.md]
  S2 --> S3[S3 locate transcript JSONL]
  S3 --> S4A[S4A grep transcript for required tool calls]
  S3 --> S4B[S4B Read/Glob expected files]
  S3 --> S4C[S4C check state assertions]
  S4A --> verdict
  S4B --> verdict
  S4C --> verdict
  verdict{verdict?}
  verdict -->|PASS or WARN| advance[Phase X+1]
  verdict -->|BLOCK| remediate[Run violation.remediation, re-invoke inspector]
  verdict -->|FATAL| rollback[Rollback X, jump to REPORT]
  remediate --> S1
  advance --> target
```

---

## Files in this skill

| Path | Purpose |
|---|---|
| [SKILL.md](SKILL.md) | The 5-step audit procedure (S1-S5). What the inspector does on every invocation. |
| [extraction-protocol.md](extraction-protocol.md) | Deterministic 4-pass extraction of `{expected_tool_calls, expected_artifacts, expected_state_assertions}` from any action `.md`. |
| [audit-report-schema.md](audit-report-schema.md) | JSON schema for `audit_report.json` and the `INSPECTOR_BEGIN/END` marker format. |
| [remediation-protocol.md](remediation-protocol.md) | Severity ladder, modality -> severity mapping, main-agent obligations per verdict. |
| [user-prompt-template.md](user-prompt-template.md) | The binding-contract user prompt (blank template + filled examples). |
| [scripts/find_transcript.py](scripts/find_transcript.py) | Locate the current conversation's transcript JSONL and the last `INSPECTOR_END` line. |
| [scripts/grep_transcript.py](scripts/grep_transcript.py) | Stream-grep a transcript for tool_use blocks matching `(tool_name_pattern, arg_regex)`. |
| [scripts/parse_action_outputs.py](scripts/parse_action_outputs.py) | Mechanical regex extraction from an action `.md`; produces the candidate proposal that the LLM classifies in S2 pass 3. |
| [tests/](tests/) | Hand-written fixtures and run-by-hand checklists. See [tests/RUN_TESTS.md](tests/RUN_TESTS.md) and [tests/INTEGRATION.md](tests/INTEGRATION.md). |

---

## Reading an `audit_report.json`

The inspector emits exactly one JSON blob per invocation, wrapped in
markers:

```
=== INSPECTOR_BEGIN phase=BASELINE ts=2026-04-21T10:34:00Z ===

## Audit verdict: BLOCK

- 1 block, 0 fatal, 2 warn, 6 pass, 1 unverified
- BLOCK missing_eval_summary_baseline: $RESULT_DIR/eval_gsm8k_baseline/eval_summary_gsm8k.json not found

```json
{ ...full audit_report.json... }
```

## Required remediations

1. EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL RESULTS_DIR="$RESULT_DIR/eval_gsm8k_baseline" bash $SKILL_ROOT/scripts/eval_accuracy.sh

=== INSPECTOR_END phase=BASELINE ts=2026-04-21T10:34:00Z verdict=BLOCK ===
```

The fenced JSON block is the machine-readable artifact. Its top-level fields
are documented in [audit-report-schema.md](audit-report-schema.md). The
short bulleted summary above the JSON is for human readers.

### Interpreting `unverified` vs `miss`

- `unverified`: inspector could not evaluate the expectation. Most common
  cause is an unresolved env var in the path template (e.g. `$TARGET_DIR/...`
  when `TARGET_DIR` was not in `RUN_ENV`). `unverified` items by themselves
  do NOT trigger BLOCK; they are diagnostic.
- `violations[*].observed == "not_found"`: inspector evaluated the
  expectation and the artifact / tool call is genuinely missing. The
  `severity` field then determines whether this is `warn`, `block`, or
  `fatal`.

A high `unverified` count (>= 50% of expectations) auto-adds a synthetic
`extraction_low_confidence` `warn` violation. This makes silent extraction
failure visible without falsely blocking the run.

---

## Limitations & known failure modes

### 1. The agent might forget to invoke inspector (residual risk)

This is the **fundamental limit of the same-agent design**. After 50+ tool
calls in a long run, the agent can drift past the binding contract and
just keep going. Mitigations baked into the design:

- The `next_checkpoint.reminder_text` field in every
  `audit_report.json` echoes back what the next inspector invocation
  should look like; the agent re-reads its own recent reply.
- The user prompt includes Rule 5 ("Self-check before advancing") which
  asks the agent to grep for the most recent `INSPECTOR_END` before reading
  the next action `.md`.
- The README documents this risk explicitly so users do not over-trust the
  audit.

If your run is long enough that this risk is unacceptable, escalate to a
hook-based architecture (see "Upgrading" below). The contract files
([extraction-protocol.md](extraction-protocol.md),
[audit-report-schema.md](audit-report-schema.md),
[remediation-protocol.md](remediation-protocol.md)) are reusable as-is by
a hook; only the trigger mechanism changes.

### 2. Tool returns are not in the transcript JSONL

Cursor's transcript JSONL records tool *invocations* (`type: tool_use`,
with `name` and `input`) but does NOT reliably record tool *return values*.
The inspector therefore audits outcomes via direct file existence checks
(Channel B), not by reading recorded tool outputs. This is a hard
assumption baked into [SKILL.md](SKILL.md). If your target skill's success
criteria are not visible on disk, inspector cannot audit them; consider
adding small `touch $WORK_DIR/.phase_<X>_done` markers in your scripts.

### 3. Multiple parallel conversations in the same project

`find_transcript.py` defaults to mtime-based selection; if you have two
chats running concurrently it can pick the wrong one. The fix is to pass
`MARKER_SENTENCE` (the first sentence of your user prompt) to the
inspector via `RUN_ENV`; `find_transcript.py` will then prefer the JSONL
whose content contains that sentence. The user-prompt template's filled
example shows exactly how to do this.

### 4. Action `.md` files written in unusual styles

The extraction protocol depends on patterns like backticked commands,
env-var-prefixed paths, and "Outputs"/"Procedure" section headings. If a
target action `.md` is mostly free-form English, the manifest will be
heavy on `unverified` and the audit's signal will be weaker. Inspect the
`extraction_diagnostics` block in `audit_report.json` to see the
candidate-from-regex vs kept-after-classification ratio. Fix by adding
"Outputs:" or "Procedure:" sections to the target action.

### 5. False positives are possible if the action `.md` is wrong

If the target action `.md` says "MUST run X" but the actual correct
behavior is to skip X under some condition, inspector will block the run.
The fix is to update the target action `.md` (e.g. add an "unless"
clause), not to argue with the inspector. Inspector is mechanical by
design.

### 6. Extraction is a probabilistic step

The extraction protocol's pass 3 (LLM classification) is per-candidate and
narrowly scoped, but it is still LLM. Pass 2 (regex) provides a non-LLM
anchor; the diff between them is recorded in `regex_anchors_diff_summary`.
If the diff is large, take the audit verdict less seriously and consider
adding more structure to the action `.md`.

---

## Debugging tips

### "Inspector returned BLOCK and I think it's wrong"

1. Open the relevant `audit_report.json` (it's in your transcript inside
   the `INSPECTOR_END` markers).
2. Find the violation. Read the `source_lines` and `source_quote` fields:
   they cite the exact location in the target `actions/<X>.md` that drove
   the expectation.
3. If the source quote is genuinely strict but you believe it should be
   conditional, edit the target action `.md` to add the condition (e.g.
   "MUST do Y unless TARGET_DIR is unset"). Then re-run.
4. If the source quote is ambiguous, the inspector defaulted conservatively
   (extraction-protocol §3 non-goal: "if uncertain, classify as `unverified`,
   never `miss`" — a `MUST` from the inspector means the source had
   explicit "MUST"/"MANDATORY"/Iron-Rule keywords).

### "Inspector returned PASS but I know a step was skipped"

1. The action `.md` probably did not declare the step as MUST. Inspector
   only blocks on items the source explicitly marked mandatory.
2. Check `unverified[]`: the step might be there because of an
   unresolvable env var or unrecoverable state.
3. Promote the step to MUST in the target action `.md` by adding "MUST" /
   "MANDATORY" near it, or move it under a section heading containing
   "Iron Rule" / "Mandatory".

### "Inspector says transcript_unreadable"

1. Run `python3 .cursor/skills/inspector/scripts/find_transcript.py`
   manually from the same cwd. Inspect the output.
2. Check `~/.cursor/projects/` exists and contains a directory matching
   your slug (cwd with `/` -> `-`).
3. If the slug computation is wrong for your environment, pass
   `--slug <slug>` explicitly when invoking.

### "Inspector flagged extraction_low_confidence"

This is informational. It means more than half of your expectations are
`unverified`. Either the action `.md` is loose (add structure) or many env
vars in the action `.md` were not in `RUN_ENV` (add them).

---

## Upgrading: when same-agent invocation is not enough

If your runs are long enough that the agent drift risk dominates (see
limitation #1 above), upgrade to a hook-based enforcement. The migration
path is:

1. Keep all four protocol documents
   ([extraction-protocol.md](extraction-protocol.md),
   [audit-report-schema.md](audit-report-schema.md),
   [remediation-protocol.md](remediation-protocol.md), this README) and
   the three helper scripts unchanged. They are not coupled to the
   trigger mechanism.
2. Add a Cursor hook (e.g. `PostToolUse` or a custom event) at
   `.cursor/hooks/phase-audit-hook.py`. The hook detects phase boundaries
   (e.g. by greping for the most recent action `.md` `Read` call in the
   transcript), runs the same audit logic in Python (no LLM
   classification), and returns a `followup_message` that injects the
   verdict into the agent's context.
3. Replace the LLM classification in extraction-protocol pass 3 with a
   pre-built lookup table mapping anchor-types to default modalities.
   This loses some flexibility but gains determinism, which matters in a
   hook.

The user prompt template is unchanged in the hook architecture; the
difference is that the hook *also* enforces invocation, so even if the
agent forgets, the audit still happens.

---

## Versioning

- `inspector_version`: `1.0` (defined in [SKILL.md](SKILL.md)).
- `manifest_version`: `1.0` (defined in
  [extraction-protocol.md §6](extraction-protocol.md)).
- Schema bumps are documented in
  [audit-report-schema.md §6](audit-report-schema.md).

When auditing a transcript that contains an older `audit_report.json`,
read the embedded `inspector_version` field and apply backward
compatibility rules from the relevant schema doc.
