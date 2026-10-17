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
  exempt). The prose fields, `rationale.reasoning`, `rationale.alternatives`,
  every `reflection` field, `notes`, `raw_logs`, and the change field
  with the `summary` role, get only the token formats, so prose and raw logs are
  not mistaken for an assignment;
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

Maps the document `kernelforge export-runs` writes from KernelForge's run logs
to the packaged `kernelforge-run-v1` declaration: one Experience per forge-loop
iteration, the one candidate kernel it measured and KernelForge's decision on
it. Collecting a run takes two steps:

```bash
kernelforge export-runs <run directory>... -o runs.json
hyperloom-kb-collect --mapping kernelforge-v1 --document runs.json --dry-run
```

It reads:

- identity from the run's `campaign_config.json`;
- `baseline.source_ms` from the pristine baseline the run's `run_state.json`
  records, which for a run given `--baseline-json` is that file's time rather
  than that of the kernel its search started from;
- `rationale.plan` from the `Summary:` line the candidate's plan opens with,
  and `rationale.reasoning` from the `Summary:` line of the reasoning log the
  Implementer kept beside that plan (see the
  [autonomous loop](../kernelforge/how-to/autonomous-loop.md)), each cut to
  three sentences. Logs without one, such as every run made before KernelForge
  asked for it, give the plan's first three sentences, and those of the
  Implementer's Claude Code transcript or of the rationale KernelForge recorded;
- the plan, the reasoning log, the measured kernel, and the lesson as files
  (`rationale.plan_file`, `rationale.reasoning_log`, `change.source_code`,
  `reflection.lesson_file`), and the transcript by its path under the log
  directory (`rationale.reasoning_file`);
- `provenance.extra` from the logs: where the candidate came from, the commit
  KernelForge kept, the record its own experience KB filed a kept candidate
  under, and the Implementer's backend and model;
- the iteration's archive, the run's `forge-loop.log`, and the Implementer's
  transcript, linked by the reasoning log its first prompt names, into the
  top-level `raw_logs`. A log or transcript first goes to the run whose
  workspace path ends most like the path it names; one that fits two runs
  equally well stops the export.

A unit is skipped when its logs could not be read as a run, when its iteration
recorded no decision, or when it has no reasoning, which is always the case for
a merge of two archived candidates because no agent made it. KernelForge writes
naive local times, which collection reads as UTC; that is right for a run made
in a container, which normally runs in UTC.
