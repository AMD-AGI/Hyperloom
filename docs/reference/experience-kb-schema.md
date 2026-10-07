---
myst:
    html_meta:
        "description": "The Experience record, its declaration, field kinds and roles, identity, and immutability rules in Hyperloom's Experience KB."
        "keywords": "Hyperloom, Experience KB, experience schema, declaration, schema_ref, field kinds, field roles, files, identity, immutability"
---

# Experience schema

An Experience is the canonical record of one attempt and what came of it. The
KB defines the frame every record shares: its categories, the kinds a field
can take, and the roles a field can play for the KB's functions. Which fields
each category holds is a producer's choice, made in its declaration. The KB
does not prescribe how a producer uses the frame; Hyperloom's fields are one
declaration among any number.

## Categories

| Category | Holds | Fields |
|---|---|---|
| `identity` | The conditions the Experience applies under. | Declared by the producer; undeclared scalar keys are kept too. Scalars only. |
| `objective` | What the attempt pursues. | The record names one declared objective by its id; the declaration describes it in text, such as "maximize B while A holds". |
| `baseline` | What the change was measured against. | Declared by the producer. |
| `rationale` | Why the change was chosen. | The KB provides `preconditions` (text, many), `reasoning` (text), and `alternatives` (text, many); a producer declares more. |
| `change` | What changed. | Declared by the producer; the change itself can be text or a file. |
| `outcome` | What came of it. | Declared by the producer. |
| `reflection` | How the outcome reads. | Declared by the producer. |
| `notes` | Anything worth keeping that no declaration names. | Labelled texts; see [Notes](#notes). |

Every category except `objective` is a map from field name to value. A
category other than `identity` holds only the fields its declaration names.

## Field kinds

| Kind | Holds | In a prompt |
|---|---|---|
| `string`, `number`, `boolean`, `version` | One scalar. A string can declare the `values` it takes. | As is. |
| `text` | Inline text, at most 32 KiB (`TEXT_MAX_BYTES`). | Whole. |
| `file` | A `FileRef` `{name, sha256, bytes}`; the KB stores the file itself. | Its name, size, and the local path an agent reads it from. |

A `string`, `text`, or `file` field can declare `many`, holding a list. Text
over the limit fails validation and names the fix: declare the field as a file.
`identity` fields hold one scalar each, never a list or a file.

A file's `name` is a relative path for display; its content is named by its
SHA-256. A service stores each content once, under `<kb_id>/files/<sha256>`, and
refuses a record that names a file it does not hold. Push sends a record's
files with it and pull brings them back, so a read always renders a path on the
host that serves it. See [Files](experience-kb.md#files).

## Field attributes and roles

A field declaration has a `name` (lowercase), a `description`, a `kind`
(default `string`), and these optional attributes:

| Attribute | Meaning | Allowed on |
|---|---|---|
| `required` | The field must be present once the record is complete. | Any field. |
| `sensitive` | The field must be redacted or pseudonymized before it leaves the producer. | Any field. |
| `many` | The value is a list. | `string`, `text`, `file`; not a field with a role or `group`. |
| `values` | The values the field takes. | `string`. |
| `group` | The field joins the Repeat Group key. | A single scalar in `baseline` or `change`. |
| `search` | The field's fuzzy-matching weight; `0` leaves it unsearched. | Any kind but `file`. |
| `role` | A KB function reads the field; see below. | One field per role per category. |

| Role | Category | Kind | Read by |
|---|---|---|---|
| `decision` | `outcome` | `string` with `values` | The outcome filter of a read, and each Repeat Group's decision counts. |
| `measurement` | `baseline`, `outcome` | `number` | List summaries, and each Repeat Group's median and variance of the outcome. |
| `summary` | `change` | `string` or `text` | List summaries, and the summary similarity of fuzzy matching. |

A schema without a decision field reads `mixed` only; one without a
measurement reports none. Every boolean outcome field is counted `true` and
`false` across a Repeat Group.

What the KB's functions read, all derived from the declaration:

- **Repeat Group key**: the schema, the identity, the objective, and every
  `group` field of `baseline` and `change`.
- **Exact lookup**: `schema_ref`, `objective`, `status`, and every single
  scalar field as `category.field`, such as `identity.gpu` or
  `outcome.decision`; every identity key is indexed, declared or not.
- **Fuzzy matching**: each field by its `search` weight. Identity fields
  default to 3, the rationale's `reasoning` to 1.5; notes are weighted 1.5 and
  the objective's id and description 0.5.
- **Completeness**: the declared kinds always, and `required` once the record
  is complete. The KB requires no field of its own.

## Metadata

Every top-level key is either knowledge or metadata, fixed by
`KNOWLEDGE_FIELDS` and `METADATA_FIELDS`:

| Kind | Keys | In a prompt |
|---|---|---|
| Knowledge: what was learned | `objective`, `identity`, `baseline`, `rationale`, `change`, `outcome`, `reflection`, `notes` | yes |
| Metadata: what keeps the record | `kind`, `id`, `schema_ref`, `schema_version`, `status`, `created_at`, `completed_at`, `run_id`, `seq`, `parent_id`, `supersedes`, `rendered_refs`, `provenance` | no |

- `kind` is `experience`; `schema_version` is `2`.
- `id` is `exp-{uuid5}`, derived only from the producer, `run_id`, and `seq`.
- `schema_ref` is the content-addressed `schema:sha256:{digest}` of the exact
  declaration the record is written under.
- `status` is `in_progress` or `complete`; `created_at` and `completed_at` are
  RFC 3339 UTC timestamps, and only a complete record has `completed_at`.
- `parent_id` names a retry's or refinement's parent; `supersedes` names the
  prior Experience a correction replaces.
- `rendered_refs` names the Experiences a read showed the deciding agent. It
  records exposure, not that the agent relied on them.
- `provenance` names the producer and its version, and optionally the model,
  prompt, snapshot, source, and JSON-safe `extra` metadata.

Both kinds are stored, exported, and synced whole. A read renders only the
knowledge, under a heading that names the Experience so an agent can cite it;
`rendered_refs` and the citations and read a producer records in
`provenance.extra` never reach a prompt. `Experience.knowledge()` returns the
knowledge as the record holds it.

## Notes

A producer that finds data worth keeping after it started writing under a
schema puts it in `notes` instead of changing the schema:

```json
"notes": {"interconnect": "XGMI links saturate during the all-reduce."}
```

Each label is a lowercase name (`[a-z][a-z0-9_.-]*`) and each value a non-empty
text. Notes are knowledge: a read renders them and fuzzy matching searches their
labels and text. They are not declared, so they leave `schema_ref` unchanged,
and they take no part in Repeat Group keys or exact lookup. Credential
screening treats them as prose.

## Declaration

A declaration has `schema_version: 2`, one or more `objectives`, and a field
list for each of `identity`, `baseline`, `rationale`, `change`, `outcome`, and
`reflection`. The rationale's defaults are implicit and cannot be redeclared.
An objective has an `id`, a `description`, and optionally a `unit`, a
`direction` (`higher_is_better` or `lower_is_better`), and a `failure_value`.

A service holds as many declarations as it is sent and stores, searches, and
syncs each by its `schema_ref` (see [Schemas](experience-kb.md#schemas)).

### Packaged declaration: `inference-recipe-v1`

The `hyperloom-sbd-v6` mapping writes Hyperloom's declaration, which is the
service's default read schema:

| Category | Fields |
|---|---|
| objective | `e2e_throughput@v1`: maximize end-to-end output throughput of the declared synthetic workload, keeping only changes that pass the accuracy gate when it is required (token/s, higher is better, failure value 0). |
| identity | Required: `model`, `gpu`, `framework`, `model_type`, `architecture`, `framework_version` (version), `precision`. Optional: `tp`, `ep`, `conc`, `isl`, `osl`, `max_model_len`, `partitions` (numbers), `compute_partition_mode`. |
| baseline | `baseline_fingerprint` (group): the SHA-256 of the exact measured-against configuration; `value` (measurement): the baseline throughput. |
| rationale | The KB defaults; the mapping writes `preconditions` and `reasoning`. |
| change | `change_family` (group, search 4): `config_variant` or `source_patch`; `change_fingerprint` (group): the SHA-256 of the normalized configuration or the patch bytes; `summary` (summary, search 4); `content` (text): the configuration delta, or the source reference, target files, and patches, as canonical JSON. |
| outcome | `decision` (decision, search 0.5): `keep`, `revert`, or `failed`; `value` (measurement); `constraints` (text, many): each gate as `name: passed=…, observed=…`; `error_class` (search 0.5). |
| reflection | `text` (text). |

Hyperloom keeps its change inline: a configuration delta or patch it measured
is small enough for text, and an agent reads it straight from the prompt. A
task whose changes outgrow 32 KiB declares its change content as a file.

## Identity and monotonic updates

The UUIDv5 namespace is fixed by `hyperloom_kb`. The UUID name is the
NUL-delimited string `producer + run_id + seq`. Callers cannot choose an id;
constructing an Experience with a non-matching id fails validation.

The same id evolves only in this direction:

```text
begin snapshot → decision snapshot → complete snapshot
```

An identical replay at any point is a no-op. The begin-time fields (the
identity, objective, baseline, provenance, and lineage) never change. Once a
change is recorded, the rationale, change, and rendered references never
change. A complete record rejects every non-identical same-id write.

A correction allocates a new sequence and id, then sets `supersedes` to the
prior id. It is a new fact, not a mutation of the old fact.

## Example

```json
{
  "kind": "experience",
  "id": "exp-68ac1daab7335dbb9348f2d369ef2c03",
  "schema_ref": "schema:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "schema_version": 2,
  "status": "complete",
  "created_at": "2026-09-04T20:00:00Z",
  "completed_at": "2026-09-04T20:12:00Z",
  "run_id": "run-456",
  "seq": 2,
  "parent_id": "",
  "supersedes": "",
  "objective": "throughput@v1",
  "identity": {"model": "qwen3-8b", "framework": "sglang", "framework_version": "0.5.17", "gpu": "mi355x"},
  "baseline": {"config": "default", "value": 2400.0},
  "rationale": {
    "preconditions": ["Baseline correctness passed."],
    "reasoning": "KV cache bandwidth dominates decode.",
    "alternatives": ["Increase tensor parallelism: the workload fits on one GPU."]
  },
  "change": {
    "knob": "kv_cache_dtype",
    "summary": "Use fp8_e4m3 for the KV cache.",
    "content": "--kv-cache-dtype fp8_e4m3",
    "profile": {"name": "profiles/decode.json", "sha256": "5b1c…", "bytes": 48213}
  },
  "outcome": {"decision": "keep", "value": 2610.5, "accuracy_passed": true},
  "reflection": {"text": "Throughput improved without violating accuracy."},
  "notes": {},
  "rendered_refs": [{"id": "exp-100", "purpose": "starting_point"}],
  "provenance": {
    "producer": "hyperloom",
    "producer_version": "1.0.0",
    "model": "claude-sonnet",
    "prompt_version": "7",
    "snapshot_version": "",
    "source_ref": "",
    "extra": {}
  }
}
```

## Compatibility

Readers reject unknown schema versions and unknown top-level keys. The open
extension points are `identity`, which keeps undeclared scalar keys, `notes`,
and `provenance.extra`. Schema evolution adds a new version; a reader never
silently coerces a document from one version into another.
