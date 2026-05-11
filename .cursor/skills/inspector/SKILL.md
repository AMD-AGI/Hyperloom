---
name: inspector
description: |
  Generic, phase-level execution auditor for any other skill. When invoked after a
  phase of a target skill (e.g. inference-optimization, training-optimization) ends,
  the inspector reads that target skill's action `.md` files at runtime, derives an
  expectation manifest of mandatory tool calls / output files / state assertions,
  audits the agent transcript JSONL plus on-disk artifacts, and writes a structured
  audit_report.json to `$RESULT_DIR/.audit/<PHASE>_<ts>.json` with verdict in
  {PASS, WARN, BLOCK, FATAL}. The chat itself shows only a single-line
  acknowledgement; the full report lives on disk. The user prompt binds the main
  agent to remediate every BLOCK / FATAL violation before advancing to the next
  phase by reading the on-disk report. The inspector itself is read-only and
  same-agent; it does not modify the audited skill. Use this skill whenever a
  long, multi-phase skill is being executed and you want to detect skipped or
  incomplete steps without flooding the chat with audit machinery.
disable-model-invocation: true
---

# Inspector — Generic Phase-Level Execution Audit Skill

This skill audits another skill's execution. You (the agent) invoke it after a
phase of a target skill ends, follow the 5 steps below verbatim, and write an
`audit_report.json` to disk per [audit-report-schema.md](audit-report-schema.md).
**The chat itself shows only a single-line acknowledgement.** The full report
is on disk at `$RESULT_DIR/.audit/<PHASE>_<ts>.json`; the user prompt that
started the run defines what you do with the verdict (typically: read the
on-disk report, run any required remediations, advance).

The companion documents are normative:

- [extraction-protocol.md](extraction-protocol.md) — how to derive the
  expectation manifest from the target action `.md`.
- [audit-report-schema.md](audit-report-schema.md) — exact JSON schema for
  the on-disk `audit_report.json` and the one-line chat ack format.
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
5. **Verdict computation is mechanical, not LLM-judged.** The `verdict`
   field MUST come from `scripts/compute_verdict.py` (S5b below). You may
   NOT hand-write the verdict from prose, MAY NOT post-edit the script's
   JSON output, and MAY NOT add justification fields like `because`,
   `deferred`, or `acceptable` next to it. If a verdict is wrong, the only
   legitimate fix is to update `scripts/semantic_rules.json` or the
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
| `PHASE_ACTION_FILES` | list of paths inside `TARGET_SKILL_DIR` | optional | `[actions/kernel-opt.md, actions/integrate.md]` |
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

---

## The 5-Step Audit Procedure

Execute steps S1 through S5 in order, every time you are invoked. Do not skip,
reorder, or merge steps. Each step ends with a brief recap of what you have
produced; the recap is for the human reader and is not parsed.

### S1 — Bind context

You do **not** echo per-step status lines into the chat. Run the steps below
silently; the only chat output for the entire audit is the single-line ack
at S5c.

1. Verify `TARGET_SKILL_DIR` exists with `Read` on its `SKILL.md`. If the file
   does not exist, emit the self-failure `audit_report.json` per
   [remediation-protocol.md §7](remediation-protocol.md) and stop.
2. Resolve `PHASE_ACTION_FILES`:
   - If provided, attempt `Read` on each entry under `<TARGET_SKILL_DIR>/`.
     On error for any entry, fall through to derivation.
   - Else read the target `SKILL.md` and locate an "Action Dispatch" table
     or an "Orchestrator Loop" section that maps phase names to action files.
   - Final fallback: `[actions/<phase_name_lower>.md]`.
   - If still not found, emit a `manifest_extraction_failed` `info` violation
     (NOT block) and proceed to S5 with an empty manifest.
3. Cross-verify the keys of `RUN_ENV` against the most recent `export VAR=`
   shell commands in the audit window (you will compute the audit window in
   S3; for S1 it is sufficient to flag any obvious mismatches). Do not error
   on mismatch; record in `run_env_unresolved` for any var you cannot
   confirm.

Recap: you now have `target_skill_dir`, `phase_action_files`, `run_env`, and
have confirmed the action files are readable.

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
6. Write the manifest to `/tmp/inspector_manifest_<PHASE_NAME>.json`
   (so `compute_verdict.py` and `emit_audit_report.py` can read it in S5).
   You do **not** print the manifest into the chat; the on-disk
   audit_report.json (S5c) embeds the manifest's diagnostics.

Recap: you now have `manifest = {expected_tool_calls, expected_artifacts,
expected_state_assertions}` plus `extraction_diagnostics` written to
`/tmp/inspector_manifest_<PHASE_NAME>.json`.

### S3 — Locate the current conversation transcript and check its health

1. Run:
   ```bash
   python3 .cursor/skills/inspector/scripts/find_transcript.py \
       --result-dir "$RESULT_DIR" \
       ${MARKER_SENTENCE:+--marker-sentence "$MARKER_SENTENCE"}
   ```
   The script returns JSON with `transcript_path`, `audit_from_line`, and
   `window_source` (`sentinel` if `$RESULT_DIR/.audit/_state.json` from a
   previous audit was readable and matched the chosen transcript;
   `start_of_file` otherwise — typically the first audit of a run).
2. **Transcript health gate.** Two independent failure modes both block
   the audit:
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
   invocation may appear inside this window; that is fine because S5c
   updates the on-disk sentinel with `last_audit_to_line = to_line`, so the
   next inspector run starts at `to_line + 1` and never re-audits this
   inspector's own calls.
4. Do **not** echo the resolved transcript / window into the chat. The
   window is recorded in the on-disk audit_report.json's `audit_window`
   block.

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

S5 is split into three sub-steps. The split keeps observation (LLM's
strength) separate from verdict computation (deterministic) and from
emission (file write + one-line ack). Composing them in a single free-form
step lets prose like "WARN because deferred per skill allowance" silently
downgrade BLOCK findings; the split is the structural defense.

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

The script's stdout is the canonical verdict payload (verdict,
violations, passes, unverified, verdict_source). Its `verdict_source`
field is `compute_verdict.py` — proof that the verdict came from the
script and not from prose.

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

Determine the next phase symbolic name by looking up the position of
`<PHASE_NAME>` in the target `SKILL.md`'s ordered phase list. If
`<PHASE_NAME>` is the last phase (e.g. `REPORT`), pass `--next-phase ""`
to `emit_audit_report.py` (it will write a terminal `next_checkpoint`).

Then run the emitter:

```bash
python3 .cursor/skills/inspector/scripts/emit_audit_report.py \
    --verdict-json /tmp/inspector_verdict_<PHASE_NAME>.json \
    --observations /tmp/inspector_obs_<PHASE_NAME>.json \
    --manifest /tmp/inspector_manifest_<PHASE_NAME>.json \
    --result-dir "$RESULT_DIR" \
    --transcript-path "<from S3 find_transcript.py>" \
    --audit-from-line <from S3> \
    --audit-to-line <from S3> \
    --target-skill-dir <TARGET_SKILL_DIR> \
    --phase-action-files "<comma,separated,relative,paths>" \
    --next-phase "<NEXT_PHASE_NAME or empty if terminal>"
```

The emitter does three things and **only** three things:

1. Writes the full `audit_report.json` to
   `$RESULT_DIR/.audit/<PHASE_NAME>_<utc-ts>.json`. This is the canonical
   audit artifact.
2. Updates the sentinel `$RESULT_DIR/.audit/_state.json` with this audit's
   `last_audit_to_line`, `last_phase`, `last_verdict`, and an appended
   `history` entry. The next inspector run reads this sentinel via
   `find_transcript.py --result-dir` to know where to start its window.
3. Prints exactly **one line** to stdout in the format:
   ```
   [Inspection] phase=<PHASE> verdict=<V> passes=<N> fatal=<n> block=<n> warn=<n> info=<n> unverified=<n> [top=<id>] -> <report_path>
   ```

Your reply for the entire audit must contain that one line and **nothing
else** about the audit. Specifically:

- Do **not** print the full `audit_report.json` into the chat as a fenced
  block. It lives on disk; the main agent reads it from
  `$RESULT_DIR/.audit/<PHASE>_*.json` only when the verdict requires
  action (BLOCK / FATAL).
- Do **not** print a "## Required remediations" section. The main agent
  reads the on-disk report's `violations[*].remediation` and executes them
  as natural next steps; the chat does not narrate them as "inspector said
  to fix X".
- Do **not** add prose like "## Audit verdict: BLOCK" before or after the
  one-liner. The one-liner is the entire chat footprint of the audit.

Allowed: at most one extra line of natural-language continuation **after**
the ack, only when verdict is FATAL and the run must stop. Example:
```
[Inspection] phase=BASELINE verdict=FATAL passes=2 fatal=1 ... -> /shared/.audit/BASELINE_2026-04-21T10-34-00Z.json
Stopping run: GSM8K accuracy regressed below the 0.65 floor (see report above).
```
For PASS / WARN / BLOCK, the one-liner stands alone.

> **S5c Rule (binding):** Do not hand-edit any field in the on-disk
> `audit_report.json` after `emit_audit_report.py` writes it. Do not
> manufacture a different ack line; copy the script's stdout verbatim.
> If you believe a verdict is wrong, the only legitimate remedy is to
> update `scripts/semantic_rules.json` or the extraction protocol BEFORE
> the next inspector invocation. Anti-pattern 7 in
> [remediation-protocol.md §6](remediation-protocol.md) makes this
> normative.

If `emit_audit_report.py` exits non-zero (e.g. `$RESULT_DIR` not writable),
treat as a self-failure per
[remediation-protocol.md §7](remediation-protocol.md): print a single line
`[Inspection] phase=<PHASE> verdict=BLOCK self_failure=<short error>` and stop.

After printing the ack: STOP. Do not perform any remediation yourself. The
main agent follows [remediation-protocol.md §3](remediation-protocol.md)
per the verdict (PASS/WARN: silently advance; BLOCK: read the on-disk
report, run remediations, re-invoke; FATAL: rollback + stop with one
business-language sentence).

Recap: a single one-line ack was printed; the full report and the sentinel
are on disk under `$RESULT_DIR/.audit/`.

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

## Cross-References

- Severity ladder: [remediation-protocol.md §1](remediation-protocol.md).
- On-disk report layout & ack format: [audit-report-schema.md §4](audit-report-schema.md).
- Semantic rule pack: [semantic-rules.md](semantic-rules.md) and
  [scripts/semantic_rules.json](scripts/semantic_rules.json).
- Helper scripts:
  - [scripts/find_transcript.py](scripts/find_transcript.py) — locate transcript and read sentinel for next audit window
  - [scripts/grep_transcript.py](scripts/grep_transcript.py) — Channel A probes
  - [scripts/parse_action_outputs.py](scripts/parse_action_outputs.py) — mechanical extraction of expected items
  - [scripts/parse_iron_rules.py](scripts/parse_iron_rules.py) — Iron Rule intake from target SKILL.md
  - [scripts/compute_verdict.py](scripts/compute_verdict.py) — deterministic verdict computation
  - [scripts/emit_audit_report.py](scripts/emit_audit_report.py) — write on-disk report + sentinel + print one-line ack
- Test fixtures: [tests/](tests/) — see
  [tests/RUN_TESTS.md](tests/RUN_TESTS.md) and
  [tests/INTEGRATION.md](tests/INTEGRATION.md).
