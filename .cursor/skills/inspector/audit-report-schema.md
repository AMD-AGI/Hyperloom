# Audit Report Schema

Defines the JSON structure that the inspector writes at the end of every
audit (SKILL.md Step S5) to disk, the sentinel state file the inspector
maintains alongside it, and the one-line acknowledgement the inspector
prints into the chat.

The report is **on disk only**: the canonical artifact lives at
`$RESULT_DIR/.audit/<PHASE>_<utc-ts>.json`. The chat shows only a single
`[Inspection] ...` line so the user knows an audit ran but is not flooded
with audit machinery. The user prompt and the main agent read the on-disk
JSON to decide whether to advance, remediate, or rollback. Future
inspector invocations read the sentinel `$RESULT_DIR/.audit/_state.json`
to set the next audit window.

---

## 1. Top-level Schema

```json
{
  "verdict_source": "compute_verdict.py",
  "phase": "BASELINE",
  "phase_index": 5,
  "target_skill": ".cursor/skills/inference-optimization",
  "phase_action_files": ["actions/baseline.md"],
  "audited_at_utc": "2026-04-21T10:34:00Z",

  "audit_window": {
    "transcript": "/root/.cursor/projects/root-Hyperloom/agent-transcripts/<uuid>/<uuid>.jsonl",
    "from_line": 412,
    "to_line": 1037
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
| `verdict_source` | string | yes | Identifier of the program that produced the `verdict`/`violations`/`passes` block. Always the literal string `"compute_verdict.py"`. The presence of this field is the agent's promise that the verdict block was not hand-edited; see [remediation-protocol.md §6 anti-pattern 7](remediation-protocol.md). For inspector self-failure reports this becomes `"compute_verdict.py|self_failure"`. |
| `phase` | string | yes | Symbolic phase name passed in S1. Uppercase by convention. |
| `phase_index` | integer | no | Optional integer index into the target skill's phase list (1-based). Helps with ordering when phases repeat (e.g. DFS LOOP iterations). |
| `target_skill` | string (path) | yes | Path to the audited skill, relative to repo root or absolute. |
| `phase_action_files` | string[] (paths) | yes | All action `.md` files audited for this phase, relative to `target_skill`. For a DFS_LOOP that ran kernel-opt this MUST contain both `actions/kernel-opt.md` and `actions/integrate.md` per IR-3. |
| `audited_at_utc` | string (ISO-8601) | yes | UTC timestamp of S5 emission. |
| `audit_window` | object | yes | Transcript and line range. See below. |
| `run_env_resolved` | object | yes | Env vars actually used for path substitution. Cross-checked against transcript exports in S1. |
| `run_env_unresolved` | string[] | yes | Env vars referenced in the action `.md` that could not be resolved. |
| `verdict` | enum | yes | One of `PASS`, `WARN`, `BLOCK`, `FATAL`. Aggregated per §3 below. |
| `verdict_summary` | string | yes | Human-readable count, e.g. `"passes=8 fatal=0 block=1 warn=2 info=0 unverified=3"`. |
| `violations` | object[] | yes | Entries with `severity in {warn, block, fatal}`. Empty array if none. |
| `passes` | object[] | yes | Entries that passed audit. May be summarised when long. |
| `unverified` | object[] | yes | Entries the inspector could not evaluate. |
| `extraction_diagnostics` | object | yes | Metadata about the manifest, see below. |
| `observations` | object | yes | The pure-fact observations dumped in S5a — `tool_call_observations`, `artifact_observations`, `state_observations`, `transcript`. See SKILL.md §S5a for schema and forbidden keys. Stored alongside the verdict so a re-audit (or a human review) can re-derive verdict by re-running `compute_verdict.py` against this block plus `semantic_rules.json`. |
| `next_checkpoint` | object | yes | Reminder to the main agent about when to invoke inspector next. |

### `audit_window`

| field | type | meaning |
|---|---|---|
| `transcript` | absolute path | The JSONL file inspector grepped. |
| `from_line` | int | First line considered (inclusive). Equals `_state.json::last_audit_to_line + 1` if a sentinel from a previous audit was readable and matched the chosen transcript; otherwise `1` (first audit of the run). |
| `to_line` | int | Last line considered (inclusive). Typically the line just before the inspector's own first tool call this invocation; written into `_state.json::last_audit_to_line` so the next audit starts at `to_line + 1`. |

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

### `observations`

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

## 4. On-disk layout

After each audit, `scripts/emit_audit_report.py` writes:

```
$RESULT_DIR/.audit/
├── _state.json                     # sentinel; consumed by find_transcript.py
└── <PHASE>_<utc-ts>.json           # full audit_report.json (this schema)
```

Filename convention for the report file:

- `<PHASE>` matches `[A-Z][A-Z0-9_]*` (e.g. `BASELINE`, `DFS_LOOP_3`).
- `<utc-ts>` is ISO-8601 UTC with `:` replaced by `-` and the `Z` suffix
  preserved (e.g. `2026-04-21T10-34-00Z`). This makes the path
  filesystem-safe across all common OSes.
- One file per audit invocation. Re-audits after BLOCK remediation produce
  a new file with a later timestamp; older files are kept for the run
  history.

### `_state.json` (sentinel)

```json
{
  "transcript_path": "/root/.cursor/projects/.../uuid.jsonl",
  "last_audit_to_line": 1037,
  "last_phase": "BASELINE",
  "last_verdict": "PASS",
  "last_ts": "2026-04-21T10:34:00Z",
  "next_phase_hint": "PROFILE",
  "history": [
    {"phase": "SETUP",    "ts": "...", "verdict": "PASS", "to_line":  412,
     "report_file": "SETUP_2026-04-21T10-12-00Z.json"},
    {"phase": "BASELINE", "ts": "...", "verdict": "PASS", "to_line": 1037,
     "report_file": "BASELINE_2026-04-21T10-34-00Z.json"}
  ]
}
```

| field | meaning |
|---|---|
| `transcript_path` | The transcript JSONL the last audit was scoped to. `find_transcript.py` only trusts the sentinel if its own resolved transcript path equals this value (cross-session safety). |
| `last_audit_to_line` | The next inspector run uses `last_audit_to_line + 1` as `audit_from_line`. |
| `last_phase` / `last_verdict` / `last_ts` | Mirrors the head of `history` for cheap lookup. |
| `next_phase_hint` | Symbolic name passed via `--next-phase`. Used by `find_transcript.py` reporting only; not authoritative. |
| `history` | Append-only list of every audit, capped at the last 50. Removed entries cannot be reconstructed; the per-phase report files (which are not capped) are the long-term record. |

The sentinel is rewritten atomically on every audit (write-to-tmp,
`os.replace`). The full per-phase reports under `$RESULT_DIR/.audit/` are
the authoritative history.

### One-line chat acknowledgement

The inspector's entire chat output for an audit is a single line printed
by `emit_audit_report.py`:

```
[Inspection] phase=<PHASE> verdict=<V> passes=<N> fatal=<n> block=<n> warn=<n> info=<n> unverified=<n> [top=<id>] -> <report_path>
```

Format constraints:

- Stable prefix `[Inspection] ` (square brackets + literal `Inspection` +
  single space) so the line is greppable by tooling.
- Key-value pairs separated by single spaces. The `verdict_summary`
  produced by `compute_verdict.py` (e.g. `passes=8 fatal=0 block=0 warn=0
  info=0 unverified=2`) is appended verbatim.
- `top=<id>` is included only when `verdict ∈ {BLOCK, FATAL}`; it names
  the highest-severity violation so the user can recognise the issue
  without opening the report.
- The arrow `-> <report_path>` ends the line; `<report_path>` may be
  truncated to its trailing 80 chars (with `...` prefix) by the emitter.
- The agent prints this line **verbatim** from `emit_audit_report.py`'s
  stdout. No surrounding markdown headers, no extra prose, no JSON. Only
  for FATAL verdicts may the agent append exactly one extra natural-language
  line explaining the stop reason (e.g. `Stopping run: GSM8K accuracy
  regressed below the 0.65 floor (see report above).`).

---

## 5. Reading the report

The on-disk `audit_report.json` is the canonical artifact. The main agent
reads it via `Read`/`Grep`/`Glob` from
`$RESULT_DIR/.audit/<PHASE>_<utc-ts>.json` (latest file per phase) only when
the verdict requires action:

- `PASS` / `WARN`: agent does not need to read the report. The one-line ack
  carries enough information to continue.
- `BLOCK`: agent reads `violations[*].remediation` and executes them as
  natural next steps in the run, then re-invokes the inspector.
- `FATAL`: agent reads `violations[*]` once for the rollback narrative,
  then performs rollback per [remediation-protocol.md §3](remediation-protocol.md).

The chat **never** contains a fenced `audit_report.json` block. Reasons:

1. Long reports would dominate the chat for users only doing PASS runs.
2. The agent's parser doesn't need the JSON in the transcript — it can
   `Read` it from disk on demand.
3. Future inspector runs locate the audit window via the sentinel; the
   round-trip via chat is unnecessary.
