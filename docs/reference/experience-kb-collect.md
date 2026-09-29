---
myst:
    html_meta:
        "description": "Write Experiences from any producer log through a declarative mapping with Hyperloom's hyperloom_kb collect engine, including custom schemas."
        "keywords": "Hyperloom, Experience KB, collect, mapping, schema, declaration, session breakdown, hyperloom-kb-collect"
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
format: hyperloom-kb.collect.v1
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
  identity: {compact: {object: {model: $doc.metadata.model, gpu: $doc.metadata.gpu}}}
  objective: e2e_throughput@v1
  baseline_value: $attempt.before
  baseline_identity: {object: {baseline_fingerprint: {sha256: $attempt.baseline}}}
  preconditions: ["baseline={$attempt.before}"]              # optional
  provenance: {source_ref: "log:{$attempt.attempt_id}", extra: {object: {}}}   # optional
  reasoning: $attempt.reasoning
  change:
    identity: {object: {change_family: config_variant, change_fingerprint: {sha256: $attempt.delta}}}
    summary: $attempt.name
    kind: config_variant                                     # optional
    content: {canonical_json: $attempt.delta}                # optional
    resource_refs: []                                        # optional
  rendered_refs: []                                          # optional: [{id, purpose}]
  outcome:
    decision: $decision
    value: $attempt.after                                    # optional
    constraints: []                                          # optional: [{name, passed, value}]
    error_class: ""                                          # optional
  reflection: "Recorded outcome: {$attempt.after}"
```

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
  tokens, JWTs, presigned URLs, and -- outside free-text fields -- credential
  assignments, credential CLI flags, and credential-shaped keys (a name containing
  `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, or `CREDENTIAL`, with `TOKENIZER`
  exempt). `reasoning`, `reflection`, `change.summary`, and `alternatives` get
  only the token formats, so prose is not mistaken for an assignment.
- `change.content` is limited to 256 KiB and a whole Experience to 1 MiB.
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
  duplicates.

AgentX sessions are skipped until their Experience identity is supported.
