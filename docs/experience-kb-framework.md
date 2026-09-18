# Framework Experience publication

Framework Experience publication is an opt-in side effect of SBD V6 export.
It is separate from Recipe KB reads and writes.

## Source of truth

The producer reads:

```text
session_breakdown.timeline[type=framework_agent].ext
  ├── proposals
  └── attempts
```

It does not revive the removed pre-V6 `optimizations.attempts` projection.

## Fidelity gate

A complete Experience is emitted only when the V6 record proves:

- all required workload identity fields, with optional EP, compute-partition,
  and max-model-length dimensions retained when present;
- an exact per-attempt measured baseline and measured-against runtime
  configuration;
- a supported terminal outcome;
- a concrete config delta or verifiable source-patch fingerprint;
- non-generic decision reasoning recorded before publication;
- a measured outcome or explicit failure class;
- required accuracy success for a kept result;
- valid Framework-event and attempt timestamps.

Attempts that fail any condition remain in the authoritative SBD V6 timeline
and appear in `reports/experience_v1_publish.json` with an exact skip reason.
The producer does not fill missing identity, measurements, reasoning, or patch
content with placeholders.

The normalized measured-against configuration is preserved in Experience
preconditions alongside its `baseline_fingerprint`. Map order and remove/unset
list order do not change the fingerprint. Credential-shaped args, environment
names, or values fail the publication fidelity gate rather than being
persisted. This is the accepted tuning stack, not the process's unbounded
ambient environment.

Configuration changes retain add/remove/unset/replace controls. Source changes
retain UTF-8 patch bytes (up to 128 KiB per patch and 256 KiB total) inside
`change.content`, not only a workspace path or hash; a source attempt without
durable patch material is skipped. Source and config attempts both retain the
exact measured-against stack and throughput/accuracy gates.

An existing session can be reviewed without configuring or writing a KB:

```bash
python scripts/review_framework_experience_fidelity.py /path/to/session
```

AgentX publication is blocked until benchmark mode and real workload identity
can be represented by the Experience Schema. Recipe KB and Experience gates
remain separate.

## Configuration

Publication is disabled by default:

```bash
export HYPERLOOM_KB_ENABLE=true
export HYPERLOOM_KB_BACKEND=local
export HYPERLOOM_KB_HOME=/path/to/local-kb
export HYPERLOOM_KB_DECL=/path/to/examples/hyperloom-kb-inference.yaml
```

When explicitly enabled, CLI startup validates the external `hyperloom_kb`
configuration before the normal optimizer preflight. Export failures never
replace or invalidate `session_breakdown.json`.

## Reasoning provenance

Configuration variants preserve their authored `note` on the measured attempt.
Their materialized tasks retain the proposal message id so attempts join the
right proposal. Source candidates preserve discovery reasoning, and local
source exploration records its gap rationale before specialist dispatch. A
provenance label such as `llm_direct` is not accepted as decision reasoning.

The current producer generates a deterministic factual reflection from the
recorded outcome and marks its source in provenance. This does not impersonate
an LLM-authored post-outcome explanation.

`rendered_refs` is empty until Local KB retrieval is integrated. Actual LLM use
requires the separate usage-trace design.
