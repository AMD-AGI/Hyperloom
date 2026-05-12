# Inspector Self-Tests — Run by Hand

These tests use the two fixtures in this folder
([fixture_action.md](fixture_action.md) and
[fixture_transcript.jsonl](fixture_transcript.jsonl)) plus the helper
scripts to validate that the inspector's mechanical layers (extraction
regex pass and transcript grep) behave as expected.

These tests do not exercise the LLM-classification step (S2 pass 3) or the
file-existence channel (S4-B); for those see
[INTEGRATION.md](INTEGRATION.md). All tests below are deterministic and
runnable from the repo root with no setup.

---

## Test 1 — Mechanical extraction finds the expected anchors

**What it covers:** [extraction-protocol.md §1 pass 2](../extraction-protocol.md)
and the regex tables in
[scripts/parse_action_outputs.py](../scripts/parse_action_outputs.py).

**Command:**

```bash
python3 .cursor/skills/inspector/scripts/parse_action_outputs.py \
    --action .cursor/skills/inspector/tests/fixture_action.md \
  | python3 -m json.tool
```

**Expected (must all hold):**

- `expected_tool_calls_candidates` contains entries for
  *all four* of these `raw_text` values (a `.py`/`.sh` path may appear under
  both `bash_script_bare`/`python_script_bare` AND `env_path` in the artifact
  bucket; the extraction-protocol's pass 4 dedups by section/anchor purpose):
  - `$SKILL_ROOT/kb/kb_query.py` (anchor `python_script_bare`, line 14)
  - `$SCRIPTS_DIR/run_fake_baseline.sh` (anchor `bash_script_bare`, line 20)
  - `curl -s http://localhost:$PORT/v1/health` (anchor `curl_bare`, line 26)
  - `$SKILL_ROOT/scripts/eval_accuracy.sh` (anchor `bash_script_bare`, line 33)
    *and* a backtick-only reference `eval_accuracy.sh` on line 29 from the
    "Iron Rule" sentence — both line 29 and line 33 should appear, and S2
    pass 3 will use line 29's surrounding text to set Iron-Rule modality.
- `expected_artifacts_candidates` contains anchored matches for at least:
  - `$RESULT_DIR/fake_baseline.json` (anchor `env_path` on line 38)
  - `$RESULT_DIR/eval_fake/eval_summary_fake.json` (anchor `env_path` on line 39)
  - `$RESULT_DIR/server_fake.log` (anchor `env_path` on line 40)
- `expected_state_assertions_candidates` contains exactly two entries with
  `raw_text` `fake_baseline_tput` (line 44) and `fake_baseline_accuracy`
  (line 45), both with anchor `set_keyword`.
- `section_map` shows the line 33 candidate sits in `PROCEDURE_SECTION` (the
  fixture does not use a separate Mandatory heading); the Iron-Rule keyword
  detection therefore relies on the surrounding sentence on line 28
  (`Mandatory accuracy gate (Iron Rule): ...Violation = invalidation.`),
  not on a section heading. The S2 pass 3 modality rule is satisfied by
  inline keywords (per [extraction-protocol.md §1 pass 3 rule 2](../extraction-protocol.md)).

**Pass condition:** all bullets above are observable in the JSON output.

---

## Test 2 — Iron-Rule keyword detection on `eval_accuracy.sh`

**What it covers:** [extraction-protocol.md §2 modality table](../extraction-protocol.md)
and [remediation-protocol.md §2 Iron-Rule trigger](../remediation-protocol.md).

**How to verify (manual reasoning, no command):**

The sentence `Mandatory accuracy gate (Iron Rule): \`eval_accuracy.sh\` MUST be
invoked. Violation = invalidation.` (line 28 of fixture_action.md, with the
following bash fence at lines 31-33)
contains all three triggers required to escalate a `MUST` failure to
`fatal`:

1. `source_quote` contains `Iron Rule` and `Violation = invalidation` -> tick.
2. The candidate (`eval_accuracy.sh`) is in `expected_tool_calls` bucket
   -> tick.
3. The relevant section is "Procedure" but the surrounding sentence has
   `Iron Rule` keyword which still satisfies the modality rules.

A real inspector LLM-classification pass should therefore tag the
`eval_accuracy.sh` candidate as `MUST` with Iron-Rule keyword found, and an
audit that finds it missing should yield `severity=fatal`.

**Pass condition:** by inspection of fixture_action.md you can identify the
line that justifies a `fatal` escalation, and the inspector's
[remediation-protocol.md §2](../remediation-protocol.md) Iron-Rule rules
match it.

---

## Test 3 — Transcript grep counts the right tool calls

**What it covers:** [scripts/grep_transcript.py](../scripts/grep_transcript.py).

**Command:**

```bash
echo '[
  {"id":"kb_warmup","tool_name_pattern":"Shell","arg_regex":"kb_query\\.py","min_count":1},
  {"id":"baseline_run","tool_name_pattern":"Shell","arg_regex":"run_fake_baseline\\.sh","min_count":1},
  {"id":"eval_run","tool_name_pattern":"Shell","arg_regex":"eval_accuracy\\.sh","min_count":1},
  {"id":"smoke_curl","tool_name_pattern":"Shell","arg_regex":"v1/health","min_count":1}
]' | python3 .cursor/skills/inspector/scripts/grep_transcript.py \
       --transcript .cursor/skills/inspector/tests/fixture_transcript.jsonl \
       --probes -
```

**Expected:**

- `kb_warmup`: `count=1`, `passes=true`, `sample_lines=[3]`.
- `baseline_run`: `count=1`, `passes=true`, `sample_lines=[4]`.
- `eval_run`: `count=0`, `passes=false`. (The transcript deliberately
  skipped the eval — this is the BLOCK case.)
- `smoke_curl`: `count=0`, `passes=false`. (Not skipped; just not invoked.
  In real audit this is `info` since smoke probe is `MAY`.)

**Pass condition:** the JSON returned by `grep_transcript.py` matches the
above. Note that `lines_scanned` should equal the number of lines in
`fixture_transcript.jsonl` (which is 6).

---

## Test 4 — Sentinel-aware audit window

**What it covers:** [scripts/find_transcript.py](../scripts/find_transcript.py)
incremental window via the on-disk sentinel
`$RESULT_DIR/.audit/_state.json` written by
[scripts/emit_audit_report.py](../scripts/emit_audit_report.py).

**Negative case (no transcripts dir):**

```bash
python3 .cursor/skills/inspector/scripts/find_transcript.py \
    --projects-root /tmp \
    --slug inspector_test_fake_slug
# expected: {"error": "no_transcripts_found", ...}
```

**Positive case (sentinel match):**

```bash
RD=/tmp/inspector_test_run
PR=/tmp/fake_proj
mkdir -p "$PR/inspector_test/agent-transcripts/uuid-A" "$RD/.audit"
TS="$PR/inspector_test/agent-transcripts/uuid-A/uuid-A.jsonl"
cp .cursor/skills/inspector/tests/fixture_transcript.jsonl "$TS"
TS_ABS="$(realpath "$TS")"

# Pretend a prior audit ended at line 6 of the transcript.
cat > "$RD/.audit/_state.json" <<EOF
{
  "transcript_path": "$TS_ABS",
  "last_audit_to_line": 6,
  "last_phase": "PREV",
  "last_verdict": "PASS",
  "last_ts": "2026-04-21T10:00:00Z",
  "history": [{"phase":"PREV","ts":"2026-04-21T10:00:00Z","verdict":"PASS",
               "to_line":6,"report_file":"PREV_2026-04-21T10-00-00Z.json"}]
}
EOF

python3 .cursor/skills/inspector/scripts/find_transcript.py \
    --projects-root "$PR" \
    --slug inspector_test \
    --result-dir "$RD"
```

**Expected:**

- `selection_method`: `mtime` (only one candidate).
- `window_source`: `sentinel`.
- `audit_from_line`: `7` (= `last_audit_to_line + 1`).
- `transcript_path`: ends with `uuid-A.jsonl`.

**Cross-session safety:** if you change `transcript_path` inside
`_state.json` to point at a different file, `find_transcript.py` falls
through to `window_source: start_of_file` with `audit_from_line: 1`.

**Cleanup:** `rm -rf /tmp/inspector_test_run /tmp/fake_proj`.

---

## Test 5 — Full inspector dry-run (read-only, by hand)

**What it covers:** the SKILL.md S1-S5 protocol when run by you, the
human, against the fixture (no agent involved).

**How to do it:**

1. Open [fixture_action.md](fixture_action.md) and
   [fixture_transcript.jsonl](fixture_transcript.jsonl) in a viewer.
2. Walk through SKILL.md's S1-S5 steps as if you were the agent. Inputs:

   ```
   TARGET_SKILL_DIR=.cursor/skills/inspector/tests/        (treat fixture as the target)
   PHASE_NAME=FAKE_BASELINE
   PHASE_ACTION_FILES=[fixture_action.md]
   RUN_ENV={"RESULT_DIR":"/tmp/inspector_test/results",
            "SKILL_ROOT":"/tmp/inspector_test/skill",
            "SCRIPTS_DIR":"/tmp/inspector_test/skill/scripts",
            "MODEL_NAME":"fake-model","PORT":"9999"}
   ```
3. Run `parse_action_outputs.py` to get the regex proposal (Test 1 above).
4. Manually classify each candidate using
   [extraction-protocol.md §1 pass 3](../extraction-protocol.md):
   - `kb_query.py`: surrounded by "MUST run the warm-up script" -> MUST.
   - `run_fake_baseline.sh`: surrounded by "MUST run the baseline" -> MUST.
   - `curl ...health`: surrounded by "MAY optionally" -> MAY.
   - `eval_accuracy.sh`: surrounded by "Iron Rule" + "Violation =
     invalidation" + "MUST" -> MUST with Iron-Rule trigger.
   - All artifacts in OUTPUT_SECTION -> MUST.
   - State assertions in STATE_SECTION -> MUST.
5. Run `grep_transcript.py` against fixture_transcript.jsonl with all four
   tool-call probes (Test 3 above).
6. The file-channel audit cannot be exercised here because the artifacts
   live at `/tmp/inspector_test/results/...` which does not exist; expect
   the inspector to record `not_found` for all three artifact entries.
7. Compose the verdict by hand:
   - 1 violation `severity=fatal` (eval_accuracy.sh missing, Iron-Rule).
   - 3 violations `severity=block` (artifacts not found, MUST modality).
   - 0 violations `severity=warn`.
   - 1 unverified or `info` for the optional `curl` smoke probe.
   - State assertions: `unverified` (cannot recover from transcript since
     the transcript explicitly skipped the eval that produces them).
   - Worst severity wins -> verdict = `FATAL`.

**Pass condition:** Your hand-derived verdict is `FATAL`, with the
violation IDs you can reproduce by running the commands in Tests 1 and 3.

---

## When to run these tests

- After editing any of the helper scripts (especially regex patterns in
  `parse_action_outputs.py`).
- After editing [extraction-protocol.md](../extraction-protocol.md) or
  [audit-report-schema.md](../audit-report-schema.md).

If you change [fixture_action.md](fixture_action.md) or
[fixture_transcript.jsonl](fixture_transcript.jsonl), update the line
numbers and expected counts in this file in the same commit.
