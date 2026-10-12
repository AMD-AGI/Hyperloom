---
myst:
    html_meta:
        "description": "The Experience record, its declaration, identity, and immutability rules in Hyperloom's Experience KB."
        "keywords": "Hyperloom, Experience KB, experience schema, declaration, schema_ref, identity, immutability"
---

# Experience schema

An Experience is the canonical record of one autonomous decision and its
measured outcome. Every storage representation must round-trip to this model
without loss.

## State model

An Experience has one of two states:

- `in_progress` contains begin-time fields and may contain decision-time fields.
  It has no `completed_at`, `outcome`, or `reflection`.
- `complete` additionally requires `completed_at`, `reasoning`, `change`,
  `outcome`, and `reflection`.

There is no abandoned state stored on the record. A reader derives abandonment
from an old `in_progress` timestamp. There is also no mutable confidence,
staleness, grouping key, or dedup flag on an Experience; those belong to future
annotations.

A complete failure is still evidence. It has an `error_class` and may lack a
numeric `value`, but its declared objective must define a numeric
`failure_value` so aggregate comparisons remain explicit.

## Top-level fields

- `kind`: fixed string `experience`.
- `id`: `exp-{uuid5}` derived only from producer, `run_id`, and `seq`.
- `schema_ref`: content-addressed `schema:sha256:{digest}` of the exact
  Experience declaration used to interpret this record.
- `schema_version`: integer `1`.
- `status`: `in_progress` or `complete`.
- `created_at`, `completed_at`: RFC3339 UTC timestamps.
- `run_id`, `seq`: producer run identity and non-negative sequence.
- `parent_id`: optional retry/refinement parent.
- `supersedes`: optional prior Experience corrected by this record.
- `identity`: non-empty open map of scalar condition values.
- `objective`: versioned objective identifier such as `throughput@v1`.
- `baseline_identity`: non-empty scalar map identifying the exact starting
  configuration independently of its measured value.
- `baseline_value`: finite numeric comparison baseline.
- `preconditions`: decision-time facts already known before the action.
- `reasoning`: why the action was chosen.
- `alternatives`: considered options and why each was rejected.
- `change`: stable identity, searchable summary, optional kind/content, and
  safe relative resource references.
- `rendered_refs`: Experiences rendered to this decision context. This is
  exposure provenance, not proof that the LLM relied on the item.
- `outcome`: disposition, optional measured value, constraint results, and
  optional error class.
- `reflection`: post-measurement interpretation.
- `provenance`: producer/version plus optional model, prompt, snapshot,
  migration source, and JSON-safe extra metadata.

Identity and change-identity values are scalars. Nested payloads belong in
change content, provenance extras, or referenced resources; allowing nested
identity values would make indexing and deterministic grouping ambiguous.

## Declaration schema

Each producer writes under one versioned declaration; its content-addressed
`schema_ref` is what a service stores, searches, and syncs by, and a service
holds as many declarations as it is sent (see
[Schemas](experience-kb.md#schemas)). A declaration contains:

- ordered identity field declarations;
- ordered baseline-identity field declarations;
- ordered change-identity field declarations;
- one or more objective declarations;
- the allowed decision vocabulary.

A field declaration has a name, description, kind, required flag, and sensitive
flag. Supported kinds are `string`, `number`, `boolean`, `version`, and `json`.
Names and descriptions are part of the contract: descriptions make fields
understandable across contributors, while `sensitive` marks fields that must be
redacted or pseudonymized before they leave the producer.

An objective declares direction (`higher_is_better` or `lower_is_better`),
description, optional unit, and optional failure value.

Declarations validate indexed fields but do not discard undeclared identity
keys. Extra keys are preserved so another consumer can index them later.

### Packaged declaration: `inference-recipe-v1`

The `hyperloom-sbd-v6` mapping writes this declaration, and it is the service's
default read schema:

- identity, required: `model`, `gpu`, `framework`, `model_type`,
  `architecture`, `framework_version`, `precision`;
- identity, optional: `tp`, `ep`, `conc`, `isl`, `osl`, `max_model_len`,
  `compute_partition_mode`, `partitions`;
- baseline identity: `baseline_fingerprint`, the SHA-256 of the exact
  measured-against runtime configuration;
- change identity: `change_family` (`config_variant` or `source_patch`) and
  `change_fingerprint`, the SHA-256 of the normalized config or the patch bytes;
- objective: `e2e_throughput@v1` (higher is better, token/s, failure value 0);
- decisions: `keep`, `revert`, `failed`.

## Identity and monotonic updates

The UUIDv5 namespace is fixed by `hyperloom_kb`. The UUID name is the
NUL-delimited string `producer + run_id + seq`. Callers cannot choose an id;
constructing an Experience with a non-matching id fails validation.

The same id evolves only in this direction:

```text
begin snapshot → decision snapshot → complete snapshot
```

An identical replay at any point is a no-op. Begin-time fields never change.
Once reasoning/change/alternatives/rendered references are captured, they never
change. A complete record rejects every non-identical same-id write.

A correction allocates a new sequence and id, then sets `supersedes` to the
prior id. It is a new fact, not a mutation of the old fact.

## Example

```json
{
  "kind": "experience",
  "id": "exp-68ac1daab7335dbb9348f2d369ef2c03",
  "schema_ref": "schema:sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "schema_version": 1,
  "status": "complete",
  "created_at": "2026-09-04T20:00:00Z",
  "completed_at": "2026-09-04T20:12:00Z",
  "run_id": "run-456",
  "seq": 2,
  "parent_id": "",
  "supersedes": "",
  "identity": {
    "model": "qwen3-8b",
    "framework": "sglang",
    "framework_version": "0.5.17",
    "gpu": "mi355x"
  },
  "objective": "throughput@v1",
  "baseline_identity": {"config": "default"},
  "baseline_value": 2400.0,
  "preconditions": ["Baseline correctness passed."],
  "reasoning": "KV cache bandwidth dominates decode.",
  "alternatives": [
    {
      "option": "Increase tensor parallelism.",
      "why_not": "The workload fits on one GPU."
    }
  ],
  "change": {
    "identity": {"knob": "kv_cache_dtype"},
    "summary": "Use fp8_e4m3 for the KV cache.",
    "kind": "config_delta",
    "content": "--kv-cache-dtype fp8_e4m3",
    "resource_refs": ["artifacts/measurement.json"]
  },
  "rendered_refs": [
    {"id": "exp-100", "purpose": "starting_point"}
  ],
  "outcome": {
    "decision": "keep",
    "value": 2610.5,
    "constraints": [
      {"name": "accuracy_delta", "passed": true, "value": -0.2}
    ],
    "error_class": ""
  },
  "reflection": "Throughput improved without violating accuracy.",
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

Readers reject unknown schema versions and unknown top-level fields. Open
extension points are limited to maps explicitly designed for them:
`identity`, `change.identity`, `constraint.value`, and `provenance.extra`.

Schema evolution adds a new version and a tested converter. A reader never
silently coerces a document from one version into another.
