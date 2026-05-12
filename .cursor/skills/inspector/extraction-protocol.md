# Extraction Protocol

How the inspector derives an **expectation manifest** from an arbitrary action
markdown file (e.g. `actions/baseline.md` of any target skill). This is the
single most failure-prone part of the inspector, so the protocol is **deliberately
mechanical first, LLM-judgment second**, with explicit non-goals to suppress
false positives.

The output of this protocol is the manifest consumed by SKILL.md Step S4
(two-channel audit).

---

## 0. Inputs and Outputs

**Inputs:**
- `action_md_paths` (list of absolute paths to action markdown files; for
  phases that mandate multiple actions — most importantly DFS_LOOP_<N> rounds
  that ran kernel-opt and therefore must also be audited against
  integrate.md per IR-3 — pass every relevant file)
- `target_skill_md_path` (absolute path to the target skill's `SKILL.md`,
  used by Pass 0 to extract Iron Rules)
- `RUN_ENV` (mapping of env var names to current values, e.g.
  `{"RESULT_DIR": "/shared_nfs/.../results/2026-04-21T..."}`)
- Optional `phase_name` (used by Pass 0 to filter Iron-Rule candidates whose
  `applies_to_phases` does not match the current phase)

**Output:** an in-memory JSON manifest with three top-level buckets:

```json
{
  "source_action_md": "actions/baseline.md",
  "source_action_md_sha1": "<first 12 hex chars of sha1>",
  "source_action_mds": ["actions/baseline.md"],
  "source_skill_md": "SKILL.md",
  "extracted_at": "2026-04-21T10:34:00Z",

  "expected_tool_calls": [
    {
      "id": "run_baseline_sh",
      "tool_name_pattern": "Shell|run_terminal_cmd|exec_on_gpu",
      "arg_regex": "run_baseline\\.sh",
      "min_count": 1,
      "modality": "MUST",
      "source_lines": [42, 43],
      "source_quote": "Run `bash $SCRIPTS_DIR/run_baseline.sh`."
    }
  ],

  "expected_artifacts": [
    {
      "id": "baseline_benchmark_json",
      "path_template": "$RESULT_DIR/baseline_${FRAMEWORK}_tp${TP}_conc${CONC}_isl${ISL}_osl${OSL}.json",
      "resolved_path": "/shared_nfs/.../results/.../baseline_sglang_tp8_conc16_isl1024_osl256.json",
      "must_be_nonempty": true,
      "modality": "MUST",
      "source_lines": [88],
      "source_quote": "Outputs: `$RESULT_DIR/baseline_*.json`"
    }
  ],

  "expected_state_assertions": [
    {
      "id": "baseline_accuracy_set",
      "field": "baseline_accuracy",
      "assertion": "is_set_and_numeric",
      "modality": "MUST",
      "source_lines": [101],
      "source_quote": "MUST set `baseline_accuracy` from GSM8K eval_summary."
    }
  ],

  "regex_anchors_diff": {
    "tool_calls_found_by_regex_only": ["..."],
    "artifacts_found_by_regex_only": ["..."],
    "promoted_to_must_by_llm": ["..."],
    "demoted_to_unverified_by_llm": ["..."]
  }
}
```

`regex_anchors_diff` is the audit-of-the-extraction itself: it is what the
mechanical pass found that the LLM pass either kept, dropped, or re-classified.
A large divergence between the two layers signals that the action `.md` is
written in an unusual style and the manifest should be trusted less (the
inspector's S5 step uses this to bias `unverified` vs `miss`).

---

## 1. The 5-Pass Procedure

### Pass 0 - Iron Rules Intake (mechanical)

Iron Rules declared in the target's SKILL.md (e.g. IR-3 "Integration is
MANDATORY" in `inference-optimization/SKILL.md`) need to be surfaced as
expectation entries even when the corresponding action `.md` (e.g.
`kernel-opt.md`) does not anchor them locally. The 2026-04-21 Qwen3-30B-A3B
run failed exactly this way: kernel-opt finished, integration was skipped,
and an inspector that read only `kernel-opt.md` had no manifest entry for
IR-3. Pass 0 closes that gap by reading the target SKILL.md directly.

1. Run `python3 scripts/parse_iron_rules.py --skill-md
   <TARGET_SKILL_DIR>/SKILL.md` (helpfully invoked transparently by
   `parse_action_outputs.py --skill-md ...`). The script extracts each
   `^### IR-N:` block and surfaces:
   - concrete tool/script/MCP/path tokens found inside the block (treated
     as `tool_call` or `artifact` candidates with `iron_rule=true` and
     `modality=MUST`),
   - any standalone MUST/MUST NOT/MANDATORY/NEVER/DO NOT/DON'T sentence as
     a `policy` candidate (also `iron_rule=true`, `modality=MUST`).
2. Each IR-N candidate carries an `applies_to_phases` glob list (`["*"]` by
   default, narrowed by `IR_PHASE_HINTS` in the script — e.g. IR-3 is
   restricted to `["DFS_LOOP_*", "SWEEP", "REPORT"]`). When `phase_name` is
   provided, candidates whose `applies_to_phases` does not match the phase
   are filtered out.
3. Pass 0 candidates feed into the same buckets as the per-action
   candidates from passes 1-2 below; they are merged in Pass 4 with
   de-duplication that keeps `iron_rule=true` if any of the merged sources
   set it.
4. Pass 0 emits one entry per IR-N into the manifest's
   `iron_rules_intake` block (mirror of `parse_iron_rules.py` output) so a
   downstream reader can inspect which Iron Rules contributed which
   manifest entries.

### Pass 1 - Read and Section-Map (mechanical)

1. `Read` the entire action `.md` file in one call. Record total line count.
2. Identify section headings via regex `^(#{1,6})\s+(.+)$`. Build a list of
   `(line_no, depth, heading_text)`.
3. Tag sections by purpose using a fixed keyword map (case-insensitive,
   whole-word):
   - `outputs|produces|writes|emits|artifacts` -> `OUTPUT_SECTION`
   - `procedure|steps|how\s+to|execute|run` -> `PROCEDURE_SECTION`
   - `inputs|prereq|requires` -> `INPUT_SECTION`
   - `state|sets|populates|fields` -> `STATE_SECTION`
   - `iron\s*rule|mandatory|must` -> `MANDATORY_SECTION`
4. Compute, for every line in the file, which section it belongs to (nearest
   preceding heading). This per-line tag is consulted in passes 2 and 3 to
   decide whether a regex match should be treated as an *expected output* vs
   an *example input*.

### Pass 2 - Mechanical regex sweep for the three buckets

Run the regex anchors below against the file content. Every match becomes a
`candidate` with `(line_no, raw_text, section_tag)`. **No classification yet.**

#### 2a. `expected_tool_calls` anchors

| Anchor | Regex | Notes |
|---|---|---|
| Bash script | `` `[^`]*\b\w+\.sh\b[^`]*` `` | Matches backticked `bash run_baseline.sh ...` |
| Python script | `` `[^`]*python3?\s+[^`]*\.py\b[^`]*` `` | Includes `python3 kb_query.py --model X` |
| MCP geak tool | `\bgeak_[a-z_]+\b` | Matches `geak_create_task`, `geak_submit_task`, etc. |
| MCP agent tool | `\bagent_[a-z_]+\b` | OOB GPU Optimizer MCP |
| MCP browser tool | `\bbrowser_[a-z_]+\b` | (rarely relevant, but kept generic) |
| Cursor tool name | `\b(Read\|Write\|Edit\|Grep\|Glob\|Shell\|Task\|WebFetch)\b` (whole-word, in code-fence or backticks only) | Avoids matching prose use of "read" |
| Curl/HTTP | `` `[^`]*\bcurl\s+[^`]*` `` | For server health / accuracy probes |
| Inline shell command | fenced ```bash ... ``` block contents | Pass 2 records the **first command line** of each bash fence as a candidate |

#### 2b. `expected_artifacts` anchors

| Anchor | Regex | Notes |
|---|---|---|
| Env-var prefixed path | `\$(?:RESULT_DIR\|WORK_DIR\|TRACE_DIR\|SESSION_DIR\|SKILL_ROOT\|RESULTS_DIR\|BASE_DIR)/[^\s\`)]+` | The most reliable signal |
| Backticked path with output suffix | `` `[^`]+\.(json\|tsv\|log\|gz\|md\|bak\|csv\|xlsx\|env\|jsonl)` `` | Excludes `.py`, `.sh` (those are tools) |
| Glob in backticks | `` `[^`]*\*[^`]*\.(json\|tsv\|log\|gz)` `` | Captures `baseline_*.json` patterns |
| Touch / mkdir target | `` `(?:touch\|mkdir\s+-p)\s+([^\s`]+)` `` | Captures explicitly-created artifacts |

#### 2c. `expected_state_assertions` anchors

| Anchor | Regex | Notes |
|---|---|---|
| "Set X" | `\bSet\s+\` ([a-z_][a-z0-9_]*)\` ` | "Set `baseline_accuracy`" |
| "Populate X" | `\bPopulate(?:s)?\s+\`([a-z_][a-z0-9_]*)\`` | |
| "Updates X" | `\bUpdates?\s+\`([a-z_][a-z0-9_]*)\`` | |
| `state.X = ...` | `\bstate\.([a-z_][a-z0-9_]*)\s*=` | Inside code blocks |
| State schema entry | `^\s*"([a-z_][a-z0-9_]*)":\s*[^,]+,?\s*#` | Inside the SKILL.md state schema, but action files often reference it |

### Pass 3 - LLM classification (judgment, narrow scope)

For each candidate from pass 2, perform exactly one classification decision:
**modality in {MUST, SHOULD, MAY, UNVERIFIED}**.

The classification rules are fixed; they must be applied mechanically by the
LLM, not freely interpreted:

1. If the candidate sits inside a `MANDATORY_SECTION` -> `MUST`.
2. Else, if the candidate's surrounding sentence (the line plus +/- 1 line)
   contains any of: `MUST`, `MANDATORY`, `ALWAYS`, `Iron Rule`, `IR-\d+`,
   `Required`, `Required:` -> `MUST`.
3. Else, if it contains any of: `SHOULD`, `recommended`, `prefer`, `default` ->
   `SHOULD`.
4. Else, if it contains any of: `MAY`, `optional`, `if needed`, `as required`,
   `e.g.`, `for example` -> `MAY`.
5. Else, if the candidate sits inside an `OUTPUT_SECTION` -> `MUST` for
   artifacts, `SHOULD` for tool calls (a tool call described in an Outputs
   section is unusual; it likely is just an example).
6. Else, if the candidate sits inside an `INPUT_SECTION` and the bucket is
   `expected_artifacts` -> drop entirely (this is an input file, not an output).
7. Else -> `UNVERIFIED`.

**LLM step constraints (binding):**
- The LLM does **not** invent new candidates that did not appear in pass 2.
- The LLM does **not** promote a candidate to `MUST` without a textual
  justification (record the matching keyword and source line in the manifest's
  `source_quote` and `source_lines` fields).
- If a candidate has no surrounding signal at all, the answer is `UNVERIFIED`,
  never `MAY`.
- The LLM must process candidates **one at a time** in source-line order; it
  does not perform any global "is this action well-tested?" judgment.

### Pass 4 - Normalisation and emission

1. **De-duplicate**: candidates with identical `(bucket, normalized_text)`
   collapse, keeping the highest modality (`MUST > SHOULD > MAY > UNVERIFIED`)
   and the union of `source_lines`. When merging across files, the surviving
   entry's `source_files` is a list (e.g.
   `["actions/integrate.md", "SKILL.md"]`) and `source_quote` accepts
   cross-file references such as `SKILL.md::IR-3 | actions/integrate.md:42`.
   `iron_rule=true` survives if any input had it; `MUST` overrides any other
   modality if any input was `MUST`.
2. **Resolve env vars**: for every `expected_artifacts.path_template`, attempt
   substitution against `RUN_ENV`. If any required env var is missing, leave
   the template unresolved and tag the entry as `unresolvable_env_var`. The
   audit step (S4) will then emit `unverified` rather than `miss` for these.
3. **Canonicalise paths**: collapse `//`, resolve `..`, but do NOT resolve
   symlinks (the audit channel handles existence, not identity).
4. **Compute `regex_anchors_diff`**: count items found by pass 2 but dropped by
   pass 3, and items whose modality the LLM changed. This is metadata, not used
   for the audit verdict, but the README explains how to inspect it.
5. **Emit** the JSON manifest as a fenced code block in the agent's reply
   (so it is preserved verbatim in the transcript) and pass it to S4.

---

## 2. Severity Mapping

The manifest stores `modality`. The audit verdict in S4 maps modality + audit
outcome -> severity, per [`remediation-protocol.md`](remediation-protocol.md):

| modality | found / passes | not found / fails | unresolvable env var |
|---|---|---|---|
| MUST | `pass` | `block` (or `fatal` if Iron Rule keyword found) | `unverified` |
| SHOULD | `pass` | `warn` | `unverified` |
| MAY | `pass` | `info` | `unverified` |
| UNVERIFIED | `pass` | `unverified` | `unverified` |

The `fatal` upgrade applies only when the `source_quote` contains one of:
`Violation = invalidation`, `Iron Rule`, `MUST NOT`, or matches `IR-\d+` AND
the section heading mentions "Iron". Otherwise `MUST` -> `block`.

---

## 3. Non-Goals (binding negative constraints)

These exist to prevent inspector from generating false positives that would
spuriously block a healthy run:

1. **Do NOT invent expectations not present in the file.** If a sensible
   action would obviously produce some artifact but the file does not mention
   it, the inspector says nothing.
2. **Do NOT promote a SHOULD to a MUST without textual justification.** No
   modality upgrade is allowed without a quoted source line.
3. **Do NOT use cross-action inference between sibling action files.** Each
   action `.md` is audited only against itself plus any explicitly-passed
   sibling action `.md` files (multi-`--action` mode for phases like
   DFS_LOOP_<N> that mandate kernel-opt + integrate). The inspector still
   does not try to infer "baseline must run before profile, therefore profile
   expects baseline outputs"; that is the orchestrator loop's responsibility.

   *Exception (one-way only):* the target's `SKILL.md` `### IR-N` Iron
   Rules MAY inject expectations into action-level manifests via Pass 0
   (Iron Rules Intake). This injection is one-way (SKILL.md → action
   manifest, never the reverse) and is gated by the IR-N's
   `applies_to_phases` glob. This is what lets IR-3 "Integration is
   MANDATORY" attach a `run_baseline.sh` MUST expectation to every
   DFS_LOOP_<N> audit even when the agent only passed
   `--action actions/kernel-opt.md`.
4. **Do NOT classify uncertain candidates as `MAY`.** If unsure, classify as
   `UNVERIFIED`. `MAY` requires explicit weakening keywords ("optional",
   "if needed").
5. **Do NOT extract from inline examples that are obviously illustrative.**
   Specifically, anything inside a fenced block whose label is `example`,
   `example output`, `sample`, or that follows a sentence ending in "for
   example", "e.g." is downgraded one modality (MUST -> SHOULD, etc.).
6. **Do NOT bind to specific values of env vars.** The manifest stores
   `path_template`, never a concrete path baked in. Resolution happens once,
   in pass 4, against `RUN_ENV`; if env vars change between phases the
   inspector re-resolves.
7. **Do NOT modify any file** in the target skill, the transcript, or the
   `RUN_ENV`-pointed directories. Inspector is read-only.

---

## 4. Heuristic-anchor helper script

A non-LLM regex pass is implemented in
[`scripts/parse_action_outputs.py`](scripts/parse_action_outputs.py). The S2
flow is:

1. Run `python3 scripts/parse_action_outputs.py --action <path1>
   [--action <path2> ...] --skill-md <TARGET_SKILL_DIR>/SKILL.md
   --phase <PHASE_NAME> --env-json <run_env.json> > /tmp/inspector_pass2.json`
   to get the mechanical candidates with line numbers. `--skill-md`
   transparently invokes
   [`scripts/parse_iron_rules.py`](scripts/parse_iron_rules.py) (Pass 0)
   and merges its output. `--phase` filters Iron-Rule candidates whose
   `applies_to_phases` does not match.
2. The LLM then performs pass 3 (classification) on those candidates only,
   without re-grepping. Iron-rule candidates already arrive with
   `modality=MUST` and `iron_rule=true`; the LLM may not downgrade them.
3. The `regex_anchors_diff` block is computed by re-loading
   `/tmp/inspector_pass2.json` and diffing against the final manifest.

This split guarantees the LLM cannot silently miss a candidate the regex
caught (passes 1-2) or the Iron-Rule scanner caught (Pass 0): every drop has
to be explicit.

---

## 5. Failure modes inside the protocol

| Failure | What inspector should do |
|---|---|
| Action `.md` is missing | Emit a single `unverified` entry `extraction_failed_no_action_md`; verdict for the phase becomes `info` (cannot judge). Do not block. |
| Action `.md` has no Procedure/Outputs/State sections at all | Pass 1 yields no section tags; classification falls to defaults. Manifest will be heavy on `UNVERIFIED`. README documents this as expected for very loose action specs. |
| Multiple env vars unresolved | All affected artifact entries become `unverified`. The phase verdict cannot rise above `WARN` purely from file checks. Tool-call channel (which doesn't depend on env) still functions normally. |
| `RUN_ENV` not provided | Inspector still extracts the manifest with unresolved templates; all artifact checks become `unverified`. The user prompt should always provide `RUN_ENV`; absence is a configuration bug, not a target-skill bug. |
| Action `.md` is gigantic (>2k lines) | Pass 1 still works (single Read call). Pass 2 regex is O(n). Pass 3 LLM classification cost is proportional to number of regex matches, not file size, so cost stays bounded. |

