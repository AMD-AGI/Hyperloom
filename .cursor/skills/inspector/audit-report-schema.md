# Audit Report Schema

Defines the JSON structure that the inspector emits at the end of every audit
(SKILL.md Step S5), and the surrounding markers that bracket the report in the
agent transcript.

The report is the **only** machine-readable artifact the inspector produces.
The user prompt parses it to decide whether to advance, remediate, or rollback.
Future inspector invocations grep the markers to set the next audit window.

---

## 1. Top-level Schema

```json
{
  "schema_version": "1.1",
  "inspector_version": "1.1",
  "manifest_version": "1.1",
  "verdict_source": "compute_verdict.py@a1b2c3d4e5f6",
  "phase": "BASELINE",
  "phase_index": 5,
  "target_skill": ".cursor/skills/inference-optimization",
  "phase_action_file": "actions/baseline.md",
  "phase_action_files": ["actions/baseline.md"],
  "audited_at_utc": "2026-04-21T10:34:00Z",

  "audit_window": {
    "transcript": "/root/.cursor/projects/root-Hyperloom/agent-transcripts/<uuid>/<uuid>.jsonl",
    "from_line": 412,
    "to_line": 1037,
    "previous_inspector_end_line": 411
  },

  "run_env_resolved": {
    "RESULT_DIR": "/shared_nfs/.../results/2026-04-21T10-12-00",
    "WORK_DIR": "/workspace/inference-optimization",
    "MODEL": "/shared_nfs/models/Qwen3-14B",
    "FRAMEWORK": "sglang",
    "TP": "8",
    "CONC": "16"
  },
  "run_env_unresolved": ["TARGET_DIR"],

  "verdict": "BLOCK",
  "verdict_summary": "1 block, 0 fatal, 0 warn, 2 unverified, 6 pass",

  "violations": [
    {
      "id": "missing_eval_summary_baseline",
      "severity": "block",
      "channel": "file",
      "expected": "$RESULT_DIR/eval_gsm8k_baseline/eval_summary_gsm8k.json",
      "expected_resolved": "/shared_nfs/.../results/2026-04-21T10-12-00/eval_gsm8k_baseline/eval_summary_gsm8k.json",
      "observed": "not_found",
      "modality_in_manifest": "MUST",
      "source_lines": [101, 102],
      "source_quote": "MANDATORY GSM8K eval via `eval_accuracy.sh` (see Accuracy Gate Protocol)",
      "remediation": "Run: EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL RESULTS_DIR=\"$RESULT_DIR/eval_gsm8k_baseline\" bash $SKILL_ROOT/scripts/eval_accuracy.sh"
    }
  ],

  "passes": [
    {
      "id": "run_baseline_sh",
      "channel": "content",
      "expected_arg_regex": "run_baseline\\.sh",
      "observed_count": 1,
      "modality_in_manifest": "MUST"
    }
  ],

  "unverified": [
    {
      "id": "target_dir_referenced_artifact",
      "channel": "file",
      "expected": "$TARGET_DIR/competitor_results.json",
      "reason": "unresolvable_env_var",
      "modality_in_manifest": "MAY"
    }
  ],

  "extraction_diagnostics": {
    "candidates_from_regex": 14,
    "candidates_kept_after_classification": 11,
    "modality_promotions": 0,
    "modality_demotions": 3,
    "regex_anchors_diff_summary": "3 candidates dropped by classification (all illustrative examples in fenced 'example' blocks)"
  },

  "observations": {
    "tool_call_observations": [
      {"id": "run_baseline_sh", "count": 1, "sample_lines": [421],
       "iron_rule": false}
    ],
    "artifact_observations": [
      {"id": "baseline_benchmark_json", "exists": true, "bytes": 14820,
       "json_fields": {}}
    ],
    "state_observations": [
      {"id": "baseline_accuracy_set", "value": 0.812, "recovered": true}
    ],
    "transcript": {
      "path": "/root/.cursor/projects/.../uuid.jsonl",
      "lines": 1037
    }
  },

  "next_checkpoint": {
    "should_invoke_inspector_after": "PROFILE",
    "reminder_text": "After completing PROFILE phase, invoke inspector again with PHASE_NAME=PROFILE."
  }
}
```

---

## 2. Field Reference

### Top level

| field | type | required | meaning |
|---|---|---|---|
| `schema_version` | string | yes (v1.1+) | Version of this report schema. Currently `"1.1"`. |
| `inspector_version` | string | yes | Version of the audit logic. Pin via SKILL.md S5. Bump on schema change. |
| `manifest_version` | string | yes | Version of the extraction protocol used to build the manifest. From [extraction-protocol.md](extraction-protocol.md). |
| `verdict_source` | string | yes (v1.1+) | Identifier of the program that produced the `verdict`/`violations`/`passes` block. Format: `compute_verdict.py@<sha1>` for normal audits, or `compute_verdict.py@<sha1>|self_failure` for inspector self-failure reports. The presence of this field is the agent's promise that the verdict block was not hand-edited; see [remediation-protocol.md §6 anti-pattern 7](remediation-protocol.md). |
| `phase` | string | yes | Symbolic phase name passed in S1. Uppercase by convention. |
| `phase_index` | integer | no | Optional integer index into the target skill's phase list (1-based). Helps with ordering when phases repeat (e.g. DFS LOOP iterations). |
| `target_skill` | string (path) | yes | Path to the audited skill, relative to repo root or absolute. |
| `phase_action_file` | string (path) | yes | Relative path inside `target_skill` of the **first** action `.md` audited. Kept for backward compatibility. |
| `phase_action_files` | string[] (paths) | yes (v1.1+) | All action `.md` files audited for this phase. For a single-action phase this is `[phase_action_file]`. For a DFS_LOOP that ran kernel-opt this MUST contain both `actions/kernel-opt.md` and `actions/integrate.md` per IR-3. |
| `audited_at_utc` | string (ISO-8601) | yes | UTC timestamp of S5 emission. |
| `audit_window` | object | yes | Transcript and line range. See below. |
| `run_env_resolved` | object | yes | Env vars actually used for path substitution. Cross-checked against transcript exports in S1. |
| `run_env_unresolved` | string[] | yes | Env vars referenced in the action `.md` that could not be resolved. |
| `verdict` | enum | yes | One of `PASS`, `WARN`, `BLOCK`, `FATAL`. Aggregated per S3 below. |
| `verdict_summary` | string | yes | Human-readable count, e.g. `"passes=8 fatal=0 block=1 warn=2 info=0 unverified=3"`. |
| `violations` | object[] | yes | Entries with `severity in {warn, block, fatal}`. Empty array if none. |
| `passes` | object[] | yes | Entries that passed audit. May be summarised when long. |
| `unverified` | object[] | yes | Entries the inspector could not evaluate. |
| `extraction_diagnostics` | object | yes | Metadata about the manifest, see below. |
| `observations` | object | yes (v1.1+) | The pure-fact observations dumped in S5a — `tool_call_observations`, `artifact_observations`, `state_observations`, `transcript`. See SKILL.md §S5a for schema and forbidden keys. Stored alongside the verdict so a re-audit (or a human review) can re-derive verdict by re-running `compute_verdict.py` against this block plus `semantic_rules.json`. |
| `next_checkpoint` | object | yes | Reminder to the main agent about when to invoke inspector next. |

### `audit_window`

| field | type | meaning |
|---|---|---|
| `transcript` | absolute path | The JSONL file inspector grepped. |
| `from_line` | int | First line considered (inclusive). Equals `previous_inspector_end_line + 1` if a prior INSPECTOR_END marker exists; otherwise the line of the original user prompt for this run. |
| `to_line` | int | Last line considered (inclusive). Typically the line just before the inspector's own first tool call this invocation. |
| `previous_inspector_end_line` | int or null | Line number of the most recent `=== INSPECTOR_END ... ===` before this audit; null if first invocation. |

### `violations[]` and `passes[]` and `unverified[]`

All three arrays use the same envelope (subset of fields per array):

| field | type | required | meaning |
|---|---|---|---|
| `id` | string | yes | Stable, snake_case identifier. Should be reproducible across runs (same manifest -> same id). |
| `severity` | enum | violations only | One of `warn`, `block`, `fatal`. Never `info` or `unverified` here; those go in their own arrays. |
| `channel` | enum | yes | `content` (transcript audit) or `file` (artifact audit) or `state` (state assertion). |
| `expected` | string | yes | Original templated string from the manifest (e.g. `$RESULT_DIR/...` or arg regex). |
| `expected_resolved` | string | file channel only | Same after `RUN_ENV` substitution. |
| `expected_arg_regex` | string | content channel only | The regex applied against transcript lines. |
| `observed` | string or int | yes | `not_found`, `0`, `3`, etc. |
| `observed_count` | int | content channel only | Count of matching transcript tool_use entries. |
| `modality_in_manifest` | enum | yes | `MUST`, `SHOULD`, `MAY`, `UNVERIFIED`. |
| `source_lines` | int[] | yes | Line numbers in the action `.md` where this expectation was extracted. |
| `source_quote` | string | yes | Verbatim text from the action `.md` justifying the expectation. |
| `reason` | string | unverified only | Free-text reason (e.g. `unresolvable_env_var`, `state_not_recoverable_from_transcript`). |
| `remediation` | string | violations only | Concrete shell command or tool invocation the main agent must perform to fix it. Should be self-contained (env vars assumed already exported). |

### `extraction_diagnostics`

| field | type | meaning |
|---|---|---|
| `candidates_from_regex` | int | Number of candidates produced by extraction-protocol pass 2. |
| `candidates_kept_after_classification` | int | Number of those that survived pass 3. |
| `modality_promotions` | int | Pass 3 upgrades over pass 2 default. Should normally be 0 (LLM downgrades, rarely promotes). |
| `modality_demotions` | int | Pass 3 downgrades over pass 2 default. |
| `regex_anchors_diff_summary` | string | Human-readable narrative of the diff. Used in `WARN`-tier diagnostics if extraction is uncertain. |

### `observations` (v1.1+)

| field | type | meaning |
|---|---|---|
| `tool_call_observations` | object[] | One entry per `expected_tool_calls` manifest item. Required keys: `id`, `count`, `sample_lines`, `iron_rule`. Pure facts from `grep_transcript.py`; no severity, no verdict. |
| `artifact_observations` | object[] | One entry per `expected_artifacts` manifest item. Required keys: `id`, `exists`, `bytes`. Optional: `json_fields` (only the dotted-path values needed by some semantic rule), `error`. |
| `state_observations` | object[] | One entry per `expected_state_assertions` manifest item. Required keys: `id`, `value`, `recovered`. |
| `transcript` | object | `{"path": "<absolute>", "lines": <wc-l>}`. Read by the `transcript_too_short` semantic rule. |

Forbidden keys at the top level of `observations`: `verdict`, `severity`,
`violations`, `passes`, `because`, `deferred`, `acceptable`, `warn`,
`block`, `fatal`. `compute_verdict.py` aborts with a non-zero exit code if
any of these is present, which guarantees the agent cannot smuggle a
verdict-y narrative into the observations dump.

### `next_checkpoint`

| field | type | meaning |
|---|---|---|
| `should_invoke_inspector_after` | string | Symbolic name of the next phase. Inspector derives this from the target skill's SKILL.md ordered phase list. |
| `reminder_text` | string | One-sentence reminder that the main agent should echo verbatim before continuing. Helps fight long-run forgetfulness. |

---

## 3. Verdict Aggregation Rule (worst severity wins)

Compute `verdict` from `violations` (PASSED entries do not affect verdict):

```
if any violation has severity == "fatal":
    verdict = "FATAL"
elif any violation has severity == "block":
    verdict = "BLOCK"
elif any violation has severity == "warn":
    verdict = "WARN"
else:
    verdict = "PASS"
```

`unverified[]` entries do **not** contribute to verdict by themselves. However,
if `len(unverified) >= 0.5 * total_expected_items`, the inspector adds a
synthetic `warn` violation `id="extraction_low_confidence"`, which can pull
the verdict from PASS to WARN. This makes silent extraction failure visible.

---

## 4. Markers (transcript-level wrapping)

Inspector's textual reply at S5 emission MUST start and end with these exact
single-line markers:

```
=== INSPECTOR_BEGIN phase=<PHASE_NAME> ts=<ISO-8601> ===
... markdown summary, then a fenced ```json block with the audit_report.json ...
=== INSPECTOR_END phase=<PHASE_NAME> ts=<ISO-8601> verdict=<VERDICT> ===
```

Format constraints:

- `<PHASE_NAME>` matches `[A-Z][A-Z0-9_]*` (e.g. `BASELINE`, `DFS_LOOP_3`).
- `<ISO-8601>` is the same value in both BEGIN and END markers (the BEGIN ts).
- `<VERDICT>` is one of `PASS`, `WARN`, `BLOCK`, `FATAL`.
- The markers are on their own lines, no leading whitespace, exactly three
  equals signs each side, single space inside.

These markers are how `find_transcript.py` locates the previous
`INSPECTOR_END` line for the next audit window, and how integration tests
verify the inspector ran. They MUST NOT appear inside the audit_report.json
itself (the JSON should not include `=== ... ===` lines).

---

## 5. Embedding the JSON in the reply

The reply structure between markers should be:

1. A short markdown header: `## Audit verdict: <VERDICT>`
2. A short bulleted summary mirroring `verdict_summary` plus the top
   violations (max 5 lines).
3. A fenced JSON code block with the full `audit_report.json`. Use language
   tag `json`.
4. (If verdict is BLOCK or FATAL) A second markdown section
   `## Required remediations` listing the `remediation` field of each
   violation as a numbered list.

The main agent parses item 3 (the fenced JSON) for machine action and reads
items 1-2 and 4 for human readability. Keeping the JSON inside a fenced code
block is necessary for it to round-trip through the transcript without
escaping issues.

---

## 6. Schema Versioning and Backward Compatibility

- `schema_version` (top-level) tracks the audit_report schema itself.
  Currently `1.1`.
- `inspector_version` and `manifest_version` are independent. Inspector logic
  may evolve without changing the manifest format and vice versa.
- Adding new optional fields is a non-breaking change.
- Removing or renaming a required field, or changing the type of a field, is
  a breaking change and must bump the major version.
- Verdict enum is closed; adding a new verdict value is a breaking change.

### Changes from 1.0 → 1.1

- New top-level required fields: `schema_version`, `verdict_source`,
  `phase_action_files`, `observations`.
- `phase_action_file` is retained for backward compatibility but should
  always equal `phase_action_files[0]`.
- The `verdict`, `verdict_summary`, `passes`, `violations`, `unverified`
  fields are now produced by `scripts/compute_verdict.py` and MUST NOT be
  hand-edited by the agent. The `verdict_source` field documents which
  build of the script ran.
- A v1.0 reader can ignore the new fields and still parse the rest of the
  report.

The README documents how to interpret an older `audit_report.json` blob found
in a transcript when reading historical conversations.
