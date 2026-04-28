---
name: inspector
description: |
  Generic, phase-level execution auditor for any other skill. When invoked after a
  phase of a target skill (e.g. inference-optimization, training-optimization) ends,
  the inspector reads that target skill's action `.md` files at runtime, derives an
  expectation manifest of mandatory tool calls / output files / state assertions,
  audits the agent transcript JSONL plus on-disk artifacts, and emits a structured
  audit_report.json with verdict in {PASS, WARN, BLOCK, FATAL}. The user prompt
  binds the main agent to remediate every BLOCK / FATAL violation before advancing
  to the next phase. The inspector itself is read-only and same-agent; it does not
  modify the audited skill. Use this skill whenever a long, multi-phase skill is
  being executed and you want to detect skipped or incomplete steps.
disable-model-invocation: true
---

# Inspector — Generic Phase-Level Execution Audit Skill

This skill audits another skill's execution. You (the agent) invoke it after a
phase of a target skill ends, follow the 5 steps below verbatim, and emit an
`audit_report.json` per [audit-report-schema.md](audit-report-schema.md). The
user prompt that started the run defines what you do with the verdict; obey it.

The companion documents are normative:

- [extraction-protocol.md](extraction-protocol.md) — how to derive the
  expectation manifest from the target action `.md`.
- [audit-report-schema.md](audit-report-schema.md) — exact JSON schema and
  marker format you must emit.
- [remediation-protocol.md](remediation-protocol.md) — severity ladder and the
  main-agent obligations per verdict.
- [user-prompt-template.md](user-prompt-template.md) — the binding contract
  that turns this skill into an enforceable check.
- [README.md](README.md) — usage, debugging, limitations.

---

## Hard Assumptions (read before proceeding)

1. **Tool results are NOT reliably present in transcript JSONL.** Each line of
   the transcript records `role` and `message.content[]` blocks of type
   `text` or `tool_use`. Tool *invocations* are visible. Tool *return values*
   may be absent entirely. The inspector therefore audits outcomes via the
   filesystem (Channel B), not by trying to read recorded tool results.
2. **The transcript file path is not exposed via env var.** Use
   [scripts/find_transcript.py](scripts/find_transcript.py) to locate it
   deterministically from cwd + mtime + optional marker-sentence cross-check.
3. **You are read-only.** You MUST NOT call any tool that mutates state:
   no `Write`, `Edit`, `StrReplace`, `EditNotebook`, `Delete`, `Shell` with
   side-effecting commands, `Task` subagents that mutate, etc. The only tools
   you call are `Read`, `Grep`, `Glob`, and `Shell` for read-only commands
   (`python3 scripts/...`, `ls`, `stat`).
4. **You do not modify the audited skill.** Even if you notice a bug in the
   target action `.md`, do not fix it. Report it as an `info` if relevant; the
   user updates the source.
5. **Versions are pinned.** This SKILL.md targets `inspector_version=1.1` and
   `manifest_version=1.1`. Both are recorded in your `audit_report.json`.
6. **Verdict computation is mechanical, not LLM-judged.** As of v1.1, the
   `verdict` field MUST come from `scripts/compute_verdict.py` (S5b below).
   You may NOT hand-write the verdict from prose, MAY NOT post-edit the
   script's JSON output, and MAY NOT add justification fields like
   `because`, `deferred`, or `acceptable` next to it. If a verdict is wrong,
   the only legitimate fix is to update `scripts/semantic_rules.json` or the
   extraction protocol BEFORE the next inspector invocation. See
   [semantic-rules.md](semantic-rules.md) and
   [remediation-protocol.md §6 anti-pattern 7](remediation-protocol.md).

---

## Inputs (provided by the user prompt or by your prior invocation)

You expect the following inputs in the working context (typically the user's
prompt copied them in via the binding contract template):

| name | type | required | example |
|---|---|---|---|
| `TARGET_SKILL_DIR` | absolute or workspace-relative path | yes | `.cursor/skills/inference-optimization` |
| `PHASE_NAME` | uppercase symbol matching `[A-Z][A-Z0-9_]*` | yes | `BASELINE` |
| `PHASE_ACTION_FILES` | list of paths inside `TARGET_SKILL_DIR` (single string also accepted for backward compat) | optional | `[actions/kernel-opt.md, actions/integrate.md]` |
| `PHASE_INDEX` | integer | optional | `5` |
| `RUN_ENV` | mapping of env var -> string | yes | `{"RESULT_DIR": "...", "MODEL": "..."}` |
| `MARKER_SENTENCE` | a sentence from the original user prompt | optional | `Run inference-optimization for Qwen3-14B...` |
| `MIN_TRANSCRIPT_LINES` | integer (default 5) | optional | `5` |

If `PHASE_ACTION_FILES` is missing, derive it from the target's `SKILL.md`
"Action Dispatch" / "Orchestrator Loop" table. Fall back to
`actions/<phase_name_lowercase>.md` if no table is present. **For DFS_LOOP
phases that perform kernel-opt, you MUST include both `actions/kernel-opt.md`
and `actions/integrate.md`** — Iron Rule IR-3 of the inference-optimization
skill makes integration mandatory after every kernel-opt round, and auditing
only `kernel-opt.md` was the structural gap that hid the 2026-04-21 Qwen3
failure.

For backward compatibility, the legacy `PHASE_ACTION_FILE` (singular) is
still accepted; treat it as a single-element `PHASE_ACTION_FILES`.

---

## The 5-Step Audit Procedure

Execute steps S1 through S5 in order, every time you are invoked. Do not skip,
reorder, or merge steps. Each step ends with a brief recap of what you have
produced; the recap is for the human reader and is not parsed.

### S1 — Bind context

1. Echo, on a single line, the resolved inputs:
   `inspector S1: target=<TARGET_SKILL_DIR> phase=<PHASE_NAME> action_file=<PHASE_ACTION_FILE>`
2. Verify `TARGET_SKILL_DIR` exists with `Read` on its `SKILL.md`. If the file
   does not exist, emit the self-failure `audit_report.json` per
   [remediation-protocol.md §7](remediation-protocol.md) and stop.
3. Resolve `PHASE_ACTION_FILE`:
   - If provided, attempt `Read` on `<TARGET_SKILL_DIR>/<PHASE_ACTION_FILE>`.
     On error, fall through to derivation.
   - Else read the target `SKILL.md` and locate an "Action Dispatch" table
     or an "Orchestrator Loop" section that maps phase names to action files.
   - Final fallback: `actions/<phase_name_lower>.md`.
   - If still not found, emit a `manifest_extraction_failed` `info` violation
     (NOT block) and proceed to S5 with an empty manifest.
4. Cross-verify the keys of `RUN_ENV` against the most recent `export VAR=`
   shell commands in the audit window (you will compute the audit window in
   S3; for S1 it is sufficient to flag any obvious mismatches). Do not error
   on mismatch; record in `run_env_unresolved` for any var you cannot
   confirm.

Recap: you now have `target_skill_dir`, `phase_action_file`, `run_env`, and
have confirmed the action file is readable.

### S2 — Build expectation manifest

Follow [extraction-protocol.md](extraction-protocol.md) verbatim. The
mechanical regex pass is implemented in
[scripts/parse_action_outputs.py](scripts/parse_action_outputs.py); the LLM
classification pass is your job.

1. Run the mechanical regex pass with **all** action files for this phase
   plus the target SKILL.md (Pass 0 Iron Rules intake — see
   [extraction-protocol.md §Pass 0](extraction-protocol.md)):
   ```bash
   python3 .cursor/skills/inspector/scripts/parse_action_outputs.py \
       --action <TARGET_SKILL_DIR>/<PHASE_ACTION_FILE_1> \
       --action <TARGET_SKILL_DIR>/<PHASE_ACTION_FILE_2> \
       --skill-md <TARGET_SKILL_DIR>/SKILL.md \
       --phase <PHASE_NAME> \
       > /tmp/inspector_pass2_<PHASE_NAME>.json
   ```
   `--action` may be repeated for every file in `PHASE_ACTION_FILES`.
   `--skill-md` triggers `parse_iron_rules.py`, which inserts iron-rule
   candidates (e.g. IR-3's `run_baseline.sh`) into the merged candidate
   list with `iron_rule=true`. Without `--skill-md` the inspector cannot
   detect missing IR-N tool calls.
   The output is a JSON proposal of candidates with line numbers.
2. `Read` each action `.md` once for context (so you can look at surrounding
   sentences when classifying modality). Iron-rule candidates already carry
   their source SKILL.md sentence in `source_quote`.
3. For each candidate in `/tmp/inspector_pass2_<PHASE_NAME>.json`, apply the
   modality classification rules from
   [extraction-protocol.md §1 pass 3](extraction-protocol.md). Process
   candidates in source-line order. Do not invent new candidates. Do not
   promote SHOULD to MUST without a quoted source line.
4. Apply the normalisation pass (de-dup, env var resolution, path
   canonicalisation) from
   [extraction-protocol.md §1 pass 4](extraction-protocol.md). Use the values
   in `RUN_ENV` for substitution.
5. Compute `regex_anchors_diff` by counting how many regex candidates were
   dropped or down-modalitied by your classification. Record in
   `extraction_diagnostics`.
6. Emit the manifest as a fenced JSON block in your reply with title
   `## Expectation manifest for <PHASE_NAME>`. This makes the manifest
   round-trip into the transcript so future inspector runs can re-read it.

Recap: you now have `manifest = {expected_tool_calls, expected_artifacts,
expected_state_assertions}` plus `extraction_diagnostics`. Hold this in
context for S4.

### S3 — Locate the current conversation transcript and check its health

1. Run:
   ```bash
   python3 .cursor/skills/inspector/scripts/find_transcript.py \
       ${MARKER_SENTENCE:+--marker-sentence "$MARKER_SENTENCE"}
   ```
   The script returns JSON with `transcript_path`, `audit_from_line`, and
   `previous_inspector_end_line`.
2. **Transcript health gate (v1.1).** Two independent failure modes both
   block the audit:
   - If the script returns `error: no_transcripts_found`, emit a
     `transcript_unreadable` violation (`block`) per
     [remediation-protocol.md §2](remediation-protocol.md) and stop here.
   - Else run `wc -l <transcript_path>` once. If the line count is below
     `MIN_TRANSCRIPT_LINES` (default 5, env-overrideable), the transcript
     has structurally collapsed (typically post-summary): Channel A would
     return `unverified` for every probe and the verdict would silently
     slide to PASS — this is exactly the failure mode that hid the
     2026-04-21 Qwen3-30B-A3B violations. Record the observation:
     ```json
     {"transcript": {"path": "...", "lines": <wc-l output>}}
     ```
     in `/tmp/inspector_obs_<PHASE>.json` so the
     `transcript_too_short` semantic rule fires in S5b. Do **not** fall
     through to "Channel A all unverified". You may still run S4 to
     populate `artifact_observations` (Channel B is unaffected by
     transcript collapse) but Channel A probes should record `count=0`
     with `reason="transcript_too_short"`.
3. Compute `to_line = max(audit_from_line, lines_in_file)`. The audit window
   is `[audit_from_line, to_line]`. Inspector's own tool calls in this
   invocation may appear inside this window; that is fine because S5 wraps
   the report in `INSPECTOR_BEGIN/END` markers and future runs will skip
   past them.
4. Echo, on a single line:
   `inspector S3: transcript=<path> window=<from>..<to> lines=<wc-l>`.

Recap: you now have an absolute transcript path and a numeric audit window.

### S4 — Two-channel audit

#### Channel A: content audit (transcript grep)

For every entry in `manifest.expected_tool_calls`, build a probe:

```json
{"id": "<entry.id>",
 "tool_name_pattern": "<entry.tool_name_pattern>",
 "arg_regex": "<entry.arg_regex>",
 "min_count": <entry.min_count>}
```

Submit all probes in one batch to `grep_transcript.py`:

```bash
echo '<probes_json_array>' | \
  python3 .cursor/skills/inspector/scripts/grep_transcript.py \
    --transcript "<transcript_path>" \
    --from-line <audit_from_line> \
    --to-line <to_line> \
    --probes -
```

Map each result entry:
- `result.passes == true` -> add to `audit_report.passes` with
  `channel="content"`, `observed_count=result.count`,
  `sample_lines=result.sample_lines`.
- `result.passes == false` -> map to severity per
  [remediation-protocol.md §2](remediation-protocol.md) using the manifest
  entry's `modality`. Add to `violations` or `unverified` accordingly. Set
  `observed=result.count`.

#### Channel B: file audit (artifact existence)

For every entry in `manifest.expected_artifacts`:

1. If the entry's `path_template` was tagged `unresolvable_env_var` in S2,
   add an `unverified` entry with `reason="unresolvable_env_var"`. Do NOT
   try to resolve at audit time.
2. Otherwise:
   - If the resolved path contains `*` or `?`, use `Glob` with the path as
     the pattern. Pass condition: at least one match.
   - Else, use `Read` with `limit: 1` against the resolved path. Pass
     condition: tool returns content (file exists and is non-empty when
     `must_be_nonempty=true`; file exists at all otherwise).
   - On any error from `Read`/`Glob` other than "file not found", treat as
     `unverified` with the error string as `reason`. On "file not found",
     map to severity per the modality table.

#### Channel C (state, derived): state assertions

For every entry in `manifest.expected_state_assertions`:

1. Try to recover the field's most recent value from transcript content via
   `Grep` for `state.<field>\s*=` or `<field>=` (shell export) inside the
   audit window.
2. If a value can be recovered, evaluate the assertion (`is_set_and_numeric`,
   `is_not_none`, etc.). On pass, add to `passes`. On fail, map to severity.
3. If no value can be recovered, add to `unverified` with
   `reason="state_not_recoverable_from_transcript"`. State assertion failures
   never escalate to `fatal` per the remediation protocol.

Recap: you now have three populated arrays — `passes`, `violations`,
`unverified` — covering content, file, and state channels.

### S5 — Observe → Compute → Emit (3 sub-steps, must be in order)

S5 is split into three sub-steps as of v1.1. Earlier inspectors composed a
free-form verdict in a single step, which let prose like "WARN because
deferred per skill allowance" silently downgrade BLOCK findings. The split
keeps observation (LLM's strength) separate from verdict computation
(deterministic).

#### S5a — Dump pure observations

Write a JSON file `/tmp/inspector_obs_<PHASE_NAME>.json` containing **only
facts**. Schema:

```json
{
  "tool_call_observations": [
    {"id": "<manifest entry id>", "count": <int>,
     "sample_lines": [<line numbers>],
     "iron_rule": <true|false>}
  ],
  "artifact_observations": [
    {"id": "<manifest entry id>", "exists": <bool>, "bytes": <int>,
     "json_fields": {"<json key>": <value>, ...},   // optional, only
                                                    // populate fields a
                                                    // semantic rule needs
     "error": "<string or omit>"}
  ],
  "state_observations": [
    {"id": "<manifest entry id>", "value": <any>, "recovered": <bool>}
  ],
  "transcript": {"path": "<absolute>", "lines": <wc-l count>}
}
```

**Forbidden top-level keys** (the script aborts if any appear):
`verdict`, `severity`, `violations`, `passes`, `because`, `deferred`,
`acceptable`, `warn`, `block`, `fatal`. Observations describe what *is*,
not what it *means*.

For artifacts whose semantic rule depends on a JSON field
(e.g. `kernel_results.json::integration_status`), populate `json_fields`
with the precise dotted-path values the rule reads. `compute_verdict.py`
will not re-read the file.

#### S5b — Compute verdict via script

```bash
python3 .cursor/skills/inspector/scripts/compute_verdict.py \
    --manifest /tmp/inspector_manifest_<PHASE_NAME>.json \
    --observations /tmp/inspector_obs_<PHASE_NAME>.json \
    --semantic-rules .cursor/skills/inspector/scripts/semantic_rules.json \
    --phase <PHASE_NAME> \
    --target-skill-dir <TARGET_SKILL_DIR> \
    --phase-action-files "<comma,separated,relative,paths>" \
    > /tmp/inspector_verdict_<PHASE_NAME>.json
```

The script's stdout is the canonical `audit_report.json` payload (verdict,
violations, passes, unverified, observations, verdict_source). Its
`verdict_source` field is `compute_verdict.py@<sha1>` — proof that the
verdict came from the script and not from prose.

> **S5 Rule (binding):** Agent MUST NOT hand-edit the JSON produced by
> `compute_verdict.py`. If you believe a verdict is wrong, the only
> legitimate remedy is to update `scripts/semantic_rules.json` or the
> extraction protocol BEFORE the next invocation, never the current
> audit. Anti-pattern 7 in
> [remediation-protocol.md §6](remediation-protocol.md) makes this
> normative.

If `compute_verdict.py` exits non-zero, treat as a self-failure per
[remediation-protocol.md §7](remediation-protocol.md).

#### S5c — Emit the report

Determine `next_checkpoint.should_invoke_inspector_after` by looking up
the position of `<PHASE_NAME>` in the target `SKILL.md`'s ordered phase
list. If `<PHASE_NAME>` is the last phase (e.g. `REPORT`), set
`should_invoke_inspector_after` to `null` and `reminder_text` to
`Run complete. No further inspector audits required.`.

Augment `/tmp/inspector_verdict_<PHASE_NAME>.json` with the
`next_checkpoint` block and the manifest sha1 (from
`parse_action_outputs.py`'s `action_md_sha1`). Do NOT touch any field
already produced by the script.

Emit the reply in this exact format:

```
=== INSPECTOR_BEGIN phase=<PHASE_NAME> ts=<ISO-8601> ===

## Audit verdict: <VERDICT>

- <verdict_summary one-liner copied verbatim from compute_verdict.py>
- <up to 5 most severe violations as bullets>

```json
{<full audit_report.json — verdict block is the script's output, untouched>}
```

## Required remediations  (omit this section if verdict is PASS or WARN)

1. <remediation field of violation 1, copied verbatim>
2. <remediation field of violation 2, copied verbatim>
...

=== INSPECTOR_END phase=<PHASE_NAME> ts=<ISO-8601> verdict=<VERDICT> ===
```

Markers: see [audit-report-schema.md §4](audit-report-schema.md) for exact
format constraints. The two `ts=` values must be identical (the BEGIN
timestamp). Markers are on their own lines with no leading whitespace.

Allowed prose in the markdown above the JSON block: a one-line restate of
`verdict_summary` and a bulletized list of violations. **Forbidden prose:**
the words "acceptable", "per skill allowance", "can be deferred", "okay
to skip", "not a real violation". If you find yourself wanting to add such
a phrase, the right action is to file a new semantic rule in
`scripts/semantic_rules.json`, not to soften this audit.

After emitting: STOP. Do not perform any remediation yourself. The main
agent reads the report and follows
[remediation-protocol.md §3](remediation-protocol.md) per its verdict.

Recap: a single reply was emitted, wrapped in `INSPECTOR_BEGIN/END` markers,
containing the script-computed verdict block (untouched) plus optional
remediation list.

---

## Self-Failure Behaviour

If at any step you cannot complete the protocol (e.g. `find_transcript.py`
errors, target skill path missing, manifest extraction crashes), emit the
self-failure report defined in
[remediation-protocol.md §7](remediation-protocol.md) and stop. The
self-failure report is itself a valid `audit_report.json` with verdict
`BLOCK` and a `inspector_self_failure` violation pointing at whatever went
wrong. This guarantees the main agent cannot misinterpret a broken inspector
run as a "skip allowed" signal.

---

## What Inspector Does NOT Do

The following are out of scope; reject any user prompt asking for them by
emitting an `info` note and continuing only with the in-scope work:

- Suggest optimizations to the target skill. (Use a different skill or a
  separate review pass.)
- Re-run remediations on the agent's behalf. (Inspector is read-only.)
- Compare audit results across multiple runs. (Stateless; one audit per
  invocation.)
- Modify the target skill's action `.md` files, even to fix obvious bugs.
- Decide that a violation is "actually fine" because of context the user
  did not put in `RUN_ENV`. Mechanical rules only.

---

## Pinned Versions and Cross-References

- `inspector_version`: `1.1` — defined here.
- `manifest_version`: `1.1` — defined in
  [extraction-protocol.md §6](extraction-protocol.md).
- `audit_report.schema_version`: `1.1` — defined in
  [audit-report-schema.md §6](audit-report-schema.md).
- Severity ladder: [remediation-protocol.md §1](remediation-protocol.md).
- Marker format: [audit-report-schema.md §4](audit-report-schema.md).
- Semantic rule pack: [semantic-rules.md](semantic-rules.md) and
  [scripts/semantic_rules.json](scripts/semantic_rules.json).
- Helper scripts:
  - [scripts/find_transcript.py](scripts/find_transcript.py)
  - [scripts/grep_transcript.py](scripts/grep_transcript.py)
  - [scripts/parse_action_outputs.py](scripts/parse_action_outputs.py)
  - [scripts/parse_iron_rules.py](scripts/parse_iron_rules.py) — new in v1.1
  - [scripts/compute_verdict.py](scripts/compute_verdict.py) — new in v1.1
- Test fixtures: [tests/](tests/) — see
  [tests/RUN_TESTS.md](tests/RUN_TESTS.md) and
  [tests/INTEGRATION.md](tests/INTEGRATION.md).
