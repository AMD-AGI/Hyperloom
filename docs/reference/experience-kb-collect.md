---
myst:
    html_meta:
        "description": "Write Experiences from any producer log through a declarative mapping with Hyperloom's hyperloom_kb collect engine, including custom schemas."
        "keywords": "Hyperloom, Experience KB, collect, mapping, schema, declaration, session breakdown, hyperloom-kb-collect, KernelForge, kernelforge-v1"
---

# Experience collection

`collect` turns one producer log document into complete Experiences through a
**mapping**: a YAML file that says which parts of the document are units, how
each unit's fields map onto an Experience declaration, and which units are not
fit to publish. The producer keeps one call site; changing the log shape or the
declaration means editing the mapping, not the producer. Hyperloom itself
collects every session through the packaged `hyperloom-sbd-v6` mapping (see
[Framework Experience publication](../experience-kb-framework.md)).

```python
from hyperloom_kb.collect import collect

report = collect(
    "hyperloom-sbd-v6",  # packaged mapping name or a mapping file path
    session_dir / "session_breakdown.json",  # or the already-loaded document
    receipt=session_dir / "reports" / "experience_collect.json",
)
```

```bash
hyperloom-kb-collect --mapping hyperloom-sbd-v6 --document session_breakdown.json --dry-run
# from a pip --target workspace: python -m hyperloom_kb.collect.cli ...
```

Without `kb=`, the target is the Experience service named by `HYPERLOOM_KB_URL`
and `HYPERLOOM_KB_TOKEN`, writing the mapping's own declaration: a mapping with
a new declaration adds a new schema to the service rather than failing (see
[Schemas](experience-kb.md#schemas)). An unconfigured environment returns a
disabled report and writes nothing. `--dry-run` projects and validates without
any KB and prints every projected Experience in the report, which is how a new
mapping is reviewed against a recorded log.

## Mapping

```yaml
format: hyperloom-kb.collect.v2
declaration: inference-recipe-v1     # packaged declaration name, or a path relative to this file
producer: {name: my-producer, version: "1", snapshot_version: my-log.v1}

skip:                                # document-level: every unit is skipped with this reason
  - reason: unsupported benchmark mode
    when: {eq: [$doc.metadata.mode, agentx]}

units:                               # nested iteration; each step binds one name
  - {each: $doc.timeline, as: event, where: {eq: [$event.type, framework_agent]}}
  - {each: $event.ext.attempts, as: attempt}
unit_id: $attempt.attempt_id         # names the unit in the report

lookup:                              # first matching item, or null
  proposal:
    from: $event.ext.proposals
    as: candidate
    where: {eq: [$candidate.proposal_id, $attempt.proposal_ref]}

let:                                 # named values, evaluated in order
  decision: {map: $attempt.outcome, table: {KEEP: keep, REVERT: revert}}

require:                             # a failing check skips the unit with its reason
  - reason: attempt has no supported terminal outcome
    check: {present: $decision}

experience:
  run_id: $doc.metadata.session.session_id
  seq: {hash48: [$event.id, $attempt.attempt_id]}
  created_at: $event.start_time       # optional; defaults to completed_at
  completed_at: $attempt.ts
  objective: e2e_throughput@v1
  identity: {compact: {object: {model: $doc.metadata.model, gpu: $doc.metadata.gpu}}}   # one expression
  provenance: {source_ref: "log:{$attempt.attempt_id}", extra: {object: {}}}           # optional
  rendered_refs: []                                          # optional: [{id, purpose}]
  notes: {object: {interconnect: $attempt.link_report}}      # optional; a note without text is left out
  baseline:                                                  # each category: one expression per declared field
    baseline_fingerprint: {sha256: $attempt.baseline}
    value: $attempt.before
  rationale:
    preconditions: ["baseline={$attempt.before}"]
    reasoning: $attempt.reasoning
  change:
    change_family: config_variant
    change_fingerprint: {sha256: $attempt.delta}
    summary: $attempt.name
    content: {canonical_json: $attempt.delta}
  outcome:
    decision: $decision
    value: $attempt.after
  raw_logs: $attempt.log_files            # optional list of file paths; a directory is not a file
  reflection:
    text: "Recorded outcome: {$attempt.after}"
```

`identity` is one expression that evaluates to the whole identity map, since an
identity may carry keys no field declares. Each of `baseline`, `rationale`,
`change`, `outcome`, and `reflection` maps the fields its declaration names to
one expression each: a key the declaration does not name, or a `required` field
left unmapped, fails when the mapping loads. A field whose expression evaluates
to `null`, `""`, `[]`, or `{}` is left out of the record. `raw_logs` is optional,
a list of local file paths; a directory path is not a file.

A `file` field's expression evaluates to the path of a local file, or a list
of paths for a `many` field; a relative path is read from the source document's
directory, or from the working directory for a document passed in memory. The
record names the file by its `FileRef`, and publishing sends the file to the
service with the record (see [Files](experience-kb.md#files)). A unit whose
file does not exist is skipped with `mapping evaluation failed: ...`.

`code_fields` is an optional list of `category.field` entries. Each must name a
file field of this mapping's declaration. Collection screens those files with
the token formats only, the way it screens a raw log, so a kernel line such as
`offs_token = tl.load(...)` is kept and an `sk-` key is still skipped. A file
field the list does not name keeps the configuration rules. The list is not
part of the declaration, so it never reaches the knowledge base.

Names are bound in order -- `doc`, each unit step, each lookup, each `let` --
and an expression may read only names bound before it. Every problem with the
mapping itself (unknown keys, an unknown built-in, a wrong argument count, a
path whose root is not bound yet) fails when the mapping loads.

## Expressions

| Form | Meaning |
|---|---|
| `$name.key.0.key` | Read a path; a missing step, or a step into a non-container, is `null`. Digits index lists. |
| `"text {$path} text"` | Template. `null` renders empty, numbers as Python `str`, objects/lists as canonical JSON. |
| `$$literal` | The literal string `$literal`. |
| other scalars, lists | Themselves; list items are evaluated. |
| `{builtin: ...}` | A mapping is exactly one built-in call. |

Values:

| Built-in | Result |
|---|---|
| `text`, `lower`, `upper`, `squash` | Stripped string; lowercase/uppercase (`null` stays `null`); whitespace collapsed. |
| `rstrip: [text, chars]` | `text` without the trailing `chars`. |
| `token` | Experience-safe token (`[A-Za-z0-9_.:@/-]`, 200 chars), `null` when empty. |
| `number`, `integer` | Finite float / non-boolean int, else `null`. |
| `string_map`, `sorted_set` | `{str: str}` with stripped keys; sorted unique non-empty strings. |
| `canonical_json`, `sha256` | Canonical JSON text; SHA-256 of a string, or of canonical JSON otherwise. |
| `hash48: [a, b, ...]` | Deterministic 48-bit integer, suitable for `seq`. |
| `length`, `compact` | Length of a string/list/mapping; a mapping without `null`/`""` values. |
| `first: [a, b, ...]` | First value that is not `null`, `""`, `[]`, or `{}`. |
| `concat: [list, ...]` | Concatenated lists; `null` items are ignored. |
| `object: {k: expr}` | A mapping of evaluated fields. |
| `if: cond, then: a, else: b` | `else` defaults to `null`. |
| `map: key, table: {...}, default: expr` | Table lookup by `str(key)`. |
| `each: list, as: x, where: cond, value: expr` | Filter and transform a list. |

Conditions (used by `where`, `if`, `skip`, `require`; booleans are themselves, anything
else counts as true when present):

`present`, `absent`, `is_bool`, `not`, `all: [...]`, `any: [...]`, `eq: [a, b]`,
`ne: [a, b]`, `gt: [a, b]` (numbers only), `in: [a, list]`, `not_in: [a, list]`,
`min_length: [text, n]`, and `starts_with_any: [text, prefixes]`.

## Enforced by the engine

A mapping cannot turn these off:

- Every projected Experience is validated against the mapping's declaration. An
  explicit `kb=` target must write that same declaration; a mismatch raises
  `ConfigurationError` before anything is written.
- Credential-shaped content skips the unit: private keys, bearer/API/GitHub/AWS
  tokens, JWTs, presigned URLs, and -- outside prose fields -- credential
  assignments, credential CLI flags, and credential-shaped keys (a name containing
  `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, or `CREDENTIAL`, with `TOKENIZER`
  exempt). The prose fields, `rationale.plan`, `rationale.plan_file`,
  `rationale.reasoning`, `rationale.alternatives`, every `reflection` field,
  `notes`, `raw_logs`, and the change field with the `summary` role, get only
  the token formats, so prose, plan files, and raw logs are not mistaken for an
  assignment. A file field the mapping lists in `code_fields` is screened the
  same way, which is how a kernel and its diff keep a name such as `offs_token`
  and still reject an `sk-` key. The list only narrows a file to that screen.
  `rationale.preconditions` is checked in full, since it may restate the
  configuration measured against. Every text file a record names is screened
  line by line the same way; a binary file is not.
- The Experience id is derived from producer, `run_id`, and `seq`, so collecting
  the same document again is idempotent: an Experience that already exists
  unchanged reports `unchanged`, and a different one under the same id is an error.

An evaluation problem in one unit (for example a list where a mapping was
expected) skips that unit with `mapping evaluation failed: ...`; a document whose
unit steps do not match raises `SourceDocumentError`.

## Report

`collect` returns a `CollectReport`; `to_dict()` (also the CLI output and the
receipt) has `format: hyperloom-kb.collect-report.v1`, `counts` (with
`by_status`), and one row per unit under `collected` (`status` is `created`,
`unchanged`, `spooled`, or `dry_run`), `skipped` (with the reason), or `errors`
(a publish failure). The CLI exits 1 when any unit errored and 2 when the
mapping, document, or configuration is invalid.

## Packaged mapping: `hyperloom-sbd-v6`

Maps Hyperloom `session_breakdown.json` (SBD V6) to the packaged
`inference-recipe-v1` declaration: one Experience per
`timeline[type=framework_agent].ext.attempts[]` row, joined to its
`ext.proposals[]` row by `proposal_ref`. It reads:

- identity from `metadata.task_config` (including `ep` and `compute_partition`);
- the measured-against stack, measurement, config delta, gates, accuracy, and
  failure attribution from the attempt;
- `reasoning`/`reasoning_origin` from the attempt, falling back to the
  proposal's `reasoning`, and publishing only reasoning traceable to the
  action-time proposal;
- `kb_read_id`/`rendered_refs` from the proposal;
- source-arm patches from `attempts[].patch_material: [{path, sha256, content}]`,
  ordered as `patch_path`, `patches_applied`, `patches_reverted` without
  duplicates, into the text field `change.content`; an attempt whose change
  content is over 32 KiB is skipped with `schema validation failed`;
- `attempts[].log_files`, a list of local file paths, into the top-level `raw_logs`.
  A path that is not a file, including a directory, skips the attempt; an
  attempt that names no log file is recorded without raw logs.

AgentX sessions are skipped until their Experience identity is supported, and a
session whose `metadata.grading.objective` is not `output_throughput` is
skipped because every Experience records `e2e_throughput@v1`.

## Packaged mapping: `kernelforge-v1`

Maps the document `kernelforge export-runs` writes from KernelForge's
`forge-loop` runs to the packaged `kernelforge-run-v1` declaration: one
Experience per candidate kernel a run archived, the kernel one iteration
measured and KernelForge's decision to keep or revert it. Each lane of a
fan-out round is measured as its own candidate, built from its own plan, so a
run yields one or more Experiences. Collecting a run takes two steps:

```bash
kernelforge export-runs <LOG_ROOT>... -o runs.json
hyperloom-kb-collect --mapping kernelforge-v1 --document runs.json --dry-run
```

Every Experience records the `kernel_time_ms@v1` objective, lower is better:
the kernel's benchmark wall time in milliseconds, the time KernelForge records
(`wall_ms`). That time is the median of three run aggregates. KernelForge runs
the task's benchmark driver three independent times. One run's time is the
median of the `wall_ms` samples that run printed. If the driver prints one
`median_ms` or `mean_ms` line instead, that printed number is the run's time.
The grouped-GEMM driver prints `mean_ms`, the arithmetic mean of that run's
per-case times, so each of its runs contributes a mean and the recorded time is
the median of those three means. KernelForge keeps a candidate on its mean case
speedup, not on this time, so a kept candidate can still be slower on this time
than the best kernel before it.

It reads:

- identity from the run's `campaign_config.json`: `operator`, the
  `operator_name` it records; `kernel`, the name KernelForge files its own KB
  records under, which is that operator normalized (`_softmax_kernel` gives
  `softmax`); `framework`, unless recorded as `unknown`; `backend`; `gpu`; and
  `driver_sha256`, the SHA-256 of the benchmark driver file's bytes, so two
  runs of one operator on different drivers do not share a repeat group. The
  same driver bytes group together. A preparation rewrite that changes those
  bytes splits them. The declaration also has `backend_version` and `dtype`,
  which `forge-loop` does not record yet, so its Experiences leave them absent;
- `baseline.best_wall_ms_before` from the candidate's `meta.json`
  `best_wall_ms_before`, the recorded wall time of the best kernel KernelForge
  had kept when the iteration began, and `outcome.candidate_wall_ms` from its
  `wall_ms`, the candidate's own recorded wall time, both the median of three
  run aggregates the objective describes, beside KernelForge's decision label
  in `outcome.decision`;
- `baseline.source_ms` from the pristine baseline the run's `run_state.json`
  records, which for a run given `--baseline-json` is that file's time rather
  than that of the kernel its search started from. It is the same statistic,
  and it is not a measurement role;
- `rationale.plan` from the whole `Summary:` line the candidate's plan opens
  with, or from the plan's first three sentences when it has none;
- `rationale.reasoning` from the whole lesson KernelForge writes for the
  iteration, `lessons/iter_NNN.md` (see the
  [autonomous loop](../kernelforge/how-to/autonomous-loop.md)). That file is the
  only reasoning record: it includes `HELD-FIXED`, `NEGATIVES`, and `DISPROOF`,
  and the loop appends the scope and outcome lines after the verdict. A single
  session that wrote no lesson gives the first three sentences of the rationale
  its `meta.json` records. A lane or a merge with neither gives no reasoning.
  A lesson longer than the 32 KB a text field holds makes collection skip the
  candidate. Reflection does not also carry the lesson;
- `reflection.analysis_report` from the whole `report.md` of the Analysis
  generation the candidate's plan was built on: the
  `forge_experiments/analysis/<commit>/generation-NNN/` its planning round's
  `orchestration/iter_NNN/context.json` names as its `analysis_bundle`. That is
  the baseline's report until Analysis refreshes after a KEEP; on campaigns
  longer than two hours Analysis profiles the kernel with rocprof-compute. A
  merge has no report, and a report longer than the 32 KB a text field holds
  is left out, with `provenance.extra.analysis_report_omitted` saying so. None
  of the profiler's raw output is collected;
- the plan, the measured kernel, and the archived diff as files
  (`rationale.plan_file`, `change.source_code`, `change.change_diff_file`).
  `change.diff` is the unified diff of every file the commit touched, which is
  where a wrapper or config edit shows up. `source_code` is only the anchor
  kernel file. The diff is linked when the candidate archived one;
- `provenance.extra` from the logs, which never reaches a prompt: `meta`, the
  scalar measurement and decision fields of the candidate's `meta.json`
  (`commit_hash`, `decision`, `kept`, `validation_passed`, `wall_ms`,
  `mean_case_speedup`, `snr_db`, `vgpr`, `baseline_wall_ms`,
  `best_wall_ms_before`, `best_mean_case_speedup_before`, `delta_vs_best_pct`,
  `session_end_reason`, `turns`, `agent_profile_id`, `agent_provider`,
  `agent_model`, `blockage_epoch`), and not its plan, rationale, bench,
  profile, pmc diagnosis, or shape; `kb_record`, the record KernelForge's own
  experience KB filed a kept candidate under;
  `candidate_origin`, `session` or `lane`; and `analysis_generation`, the
  `analysis/<commit>/generation-NNN` the Analysis report comes from.

Every Experience has a `raw_logs` field, from the document's top-level
`raw_logs`, and it is intentionally empty for now: it links exactly one 0-byte
file, `raw-logs-not-collected.log`, and no run logs. The global KB refuses a
complete Experience without raw logs (HTTP 400
`a complete Experience requires raw_logs`), and KernelForge's run logs are not
collected yet. `kernelforge export-runs` writes that file beside its document,
which names it relative to itself, and collection reads a relative path from
the document's directory, so the file has to stay beside the document or every
candidate is skipped. A document without a top-level `raw_logs`, such as one an
earlier `export-runs` wrote, gives Experiences without raw logs, which the
global KB refuses: export its runs again before collecting them.

Of the run's own files, an Experience links only the plan, the measured
kernel, and the archived diff above, so they are the only ones that go with it
to a KB. No `forge-loop.log`, `events.jsonl`, or other archive file does;
the scalar fields of `meta.json` go as `provenance.extra.meta`, and its plan,
rationale, per-case times, and other benchmark measurements do not, and of the
Analysis directory only the report's text goes. A `forge-loop.log` is only
read, by the one run whose `kernel_path` the kernel line names, including a
run whose `forge_experiments` was moved out of that workspace. When several
runs record that kernel, it is read by the run whose workspace path ends most
like the path the line announces; one that fits two runs equally well stops
the export. A `forge_experiments` directory that never wrote
`campaign_config.json` or `run_state.json` is not a run the export reports.
A record that exists but is not readable JSON, and a candidate whose
`meta.json` is missing or malformed, is one entry saying why.

A unit is skipped when its logs could not be read as a run, or when it has no
reasoning, which is always the case for a merge of two archived candidates
because no agent made it, and for a lane whose iteration recorded no lesson.
KernelForge writes naive local times, which collection
reads as UTC; that is right for a run made in a container, which normally runs
in UTC.
