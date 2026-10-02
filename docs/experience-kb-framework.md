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
                  experience_citations,
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
  `patches_applied`, `patches_reverted`, whatever their size. A patch outside
  the session, not UTF-8, or carrying a credential is left out, and an attempt
  with no recorded patch is not published unless it changed configuration.
- **Configuration from a specialist.** A source attempt records the server args
  and environment variables its specialist delivered as its `config_delta`.
  One that delivered no patch is published as a `config_variant` Experience;
  its `provenance.extra.arm` still says `source`.

## Reasoning provenance

Configuration variants preserve their authored action-time `reasoning` (and
record which payload field supplied it in `reasoning_origin`) on the measured
attempt. Every explore grid the orchestration agent emits, whether proposed
for review or delegated to run directly, records one proposal, and its task
carries that proposal's id so its attempts join it. Source candidates preserve discovery reasoning. Gap
context and post-action result text are recorded when present but are not
accepted as original decision reasoning, and a provenance label such as
`llm_direct` is not a substitute for reasoning.

`rendered_refs` records the Experiences the service injected into the decision
that produced the proposal; it is exposure, not reliance. A read that matched
nothing still records its `kb_read_id`, with no `rendered_refs`. What the deciding
agent says it relied on is `experience_citations`: per variant for the
orchestration agent, per proposal or per written patch for a specialist, each
`{id, stance, claim}` with `stance` one of `adopt`, `adapt`, `avoid`, or
`contrast`. A citation of an Experience that agent was not shown, or with
another stance, is dropped where the agent's output enters the loop. A grid
variant that asks for exactly the change a specialist proposed in the same
macro-cycle also carries that specialist's proposal citations; one that changes
the proposal carries only the orchestration agent's own. Proposal citations do
not yet reach an attempt from a multi-node auto-materialized grid or from an
upstream PR candidate. The
Experience carries its citations in `provenance.extra.experience_citations`;
how often a cited Experience worked out is not stored on any record but
derived by the KB that holds both.

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
their benchmark mode and workload identity. For the same reason an AgentX run
reads no Experience: every one it could be shown was measured on the synthetic
workload. A session graded on anything but output throughput is skipped and
reads nothing too: every Experience records `e2e_throughput@v1`, and its
KEEP/REVERT decisions answer another objective.

## Configuration

Publications go to the workspace's local Experience KB service, which every
optimize launch starts when it is not serving; its configuration, the global KB
it can push to and pull from, and its HTTP API are in
[Experience KB service](reference/experience-kb.md).

Publication is disabled only when `HYPERLOOM_KB_URL` is unset. CLI startup
fails when the packaged mapping cannot load or validates a different
declaration than the mapping produces.
