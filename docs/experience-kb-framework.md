# Framework Experience publication

Framework Experience publication is a side effect of SBD V6 export.
It is separate from Recipe KB reads and writes.

The optimizer owns two ends of it: recording each Framework attempt's facts on
the SBD V6 timeline, and handing the written `session_breakdown.json` to the
`hyperloom_kb` package. The projection from attempts to Experiences -- which
fields become identity, baseline, change, and outcome, and which attempts are
fit to publish -- is the `hyperloom-sbd-v6` mapping packaged in `hyperloom_kb`
(see [Experience collection](reference/experience-kb-collect.md)).

## What Hyperloom records

Every Experience comes from one row of:

```text
session_breakdown.timeline[type=framework_agent].ext
  ├── proposals   reasoning, kb_read_id, rendered_refs
  └── attempts    measured_against, measurement, config_delta, gates, accuracy,
                  failure.attribution, reasoning, reasoning_origin,
                  patch_path, patches_applied, patches_reverted, patch_material
```

plus the workload identity in `metadata.task_config`. These fields are
declared in `breakdown/schema.py` (`V6FrameworkAttempt`, `V6FrameworkProposal`).

- **Measured-against stack.** Both arms record the exact configuration an
  attempt was judged on, not the session's current configuration.
- **Failure attribution.** Unmeasured attempts are classified as
  `candidate_caused`, `environment`, `harness`, or `unknown`; only
  `candidate_caused` failures become Experiences.
- **Patch material.** Source attempts record each session-local patch they
  applied or reverted as `{path, sha256, content}`, in the order `patch_path`,
  `patches_applied`, `patches_reverted`. A patch outside the session, over
  128 KiB, not UTF-8, or carrying a credential is left out, and an attempt with
  no recorded patch is not published.

## Reasoning provenance

Configuration variants preserve their authored action-time `reasoning` (and
record which payload field supplied it in `reasoning_origin`) on the measured
attempt. Their materialized tasks retain the proposal message id so attempts
join the right proposal. Source candidates preserve discovery reasoning. Gap
context and post-action result text are recorded when present but are not
accepted as original decision reasoning, and a provenance label such as
`llm_direct` is not a substitute for reasoning.

`rendered_refs` records the Experiences the service injected into the decision
that produced the proposal. Whether the LLM relied on them requires a separate
usage-trace design.

## Publication

When `HYPERLOOM_KB_URL` is set, every SBD V6 export calls:

```python
collect("hyperloom-sbd-v6", breakdown, receipt=session_dir / "reports" / "experience_collect.json")
```

The receipt lists each attempt as collected (with its Experience id and write
status), skipped (with the mapping's reason), or errored. Export failures never
replace or invalidate `session_breakdown.json`; a network failure spools the
write for retry. Re-exporting a session is idempotent.

An existing session can be reviewed without writing anything:

```bash
hyperloom-kb-collect --mapping hyperloom-sbd-v6 \
  --document /path/to/session/session_breakdown.json --dry-run
```

AgentX sessions are skipped until the Experience declaration can represent
their benchmark mode and workload identity.

## Configuration

Publications go to the workspace's local Experience KB service, which every
optimize launch starts when it is not serving; its configuration, the global KB
it can push to and pull from, and its HTTP API are in
[Experience KB service](reference/experience-kb.md).

Publication is disabled only when `HYPERLOOM_KB_URL` is unset. CLI startup
fails when the packaged mapping cannot load or validates a different
declaration than the mapping produces; `cold_start_check.py
--require-experience-kb` also checks the service's health and declaration.
