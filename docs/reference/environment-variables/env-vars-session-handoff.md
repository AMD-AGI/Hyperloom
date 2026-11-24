---
myst:
    html_meta:
        "description": "Session and observability hand-off variables for Hyperloom: Claw session and user IDs, Langfuse live trace push settings, and gateway attribution for per-call spend reconciliation."
        "keywords": "Hyperloom, environment variables, session, Langfuse, HYPERLOOM_LANGFUSE_ENABLE, HYPERLOOM_LLM_ATTRIBUTION, CLAW_SESSION_ID, observability, LiteLLM"
---
# Session / observability hand-off

These are read by `src/hyperloom/inference_optimizer/session/manifest.py` and the `src/hyperloom/inference_optimizer/breakdown/collectors/`
package to populate `session_breakdown.json` for downstream consumers.

| Variable          | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| `CLAW_SESSION_ID` | Hosted SaFE / Claw session id, written to `session.claw_session_id` in `session_breakdown.json`. Set by the Primus-Claw sandbox; unset for local runs. |
| `SANDBOX_USER_ID` | Hosted SaFE / Claw user id, written to `session.sandbox_user_id`. Set by Primus-Claw; unset for local runs.                                            |
| `HYPERLOOM_LANGFUSE_ENABLE` | Primary switch (default **off**) for live Langfuse trace push. See details below. |
| `HYPERLOOM_LLM_ATTRIBUTION` | Names the gateway whose attribution headers every LLM call should carry (default **unset**, emitting none). See details below. |

**`HYPERLOOM_LANGFUSE_ENABLE`** details:

Primary switch (default **off**) for live Langfuse trace push.

- **SDK install**: when this flag is on, `src/hyperloom/inference_optimizer/assets/install.sh` auto-installs the optional `langfuse` SDK on demand and skips it entirely when off — no separate `pip install '...[trace]'` is required.
- **Live push**: when set to `1/true/yes/on` and the three `LANGFUSE_*` credentials are present, every in-process LLM call is mirrored into Langfuse while the run is live. A session-end flush backfills calls still missing their token or text half, the recipe-KB, specialist-intel, Forge-step, GEMM-tuning and trajectory logs, and KEEP/REVERT decision Scores.
- **Local ledger**: `reports/trace/*.jsonl` is always written regardless of this flag. If the SDK is unavailable, live push degrades to a no-op.
- **Correlation**: the Langfuse trace ID and `session_id` grouping are derived from `claw_session_id` (env `CLAW_SESSION_ID`), falling back to the internal session ID for standalone runs. Live push and the offline `backfill_langfuse` CLI collapse onto one trace per Primus-Claw session.
- **Span layout**: `trace → phase span (PRELUDE/ENABLEMENT/FRAMEWORK_AGENT/KERNEL_AGENT/SWEEP/…) → agent span (component: orchestration/kernel/specialist/critic/geak/forge/…) → Generation`. Each KEEP/REVERT/`gain_pct` Score attaches to the agent span that produced the decision, with a trace-level fallback when no matching span exists.
- **Recipe-KB spans**: under the `recipe_kb` agent span, local reads/writes and remote KB Store publish attempts are recorded from `runtime/recipe_snapshot/.audit.jsonl`. Read spans use `kb:recipe_snapshot:<method>`; write spans use `kb:recipe_write:<generator>`, where the generator distinguishes normal `close` from `t4_fallback`. Remote rows report `written`, `skipped`, or `error` without recording credentials or payload bodies.
- **Receipt**: every session records a `langfuse` section in `session_breakdown.json` (and `reports/trace/langfuse_receipt.json`) noting:
  - Whether push was enabled (or the `disabled_reason`)
  - The redacted connection config (host and key-presence booleans — never the keys themselves)
  - The derived `trace_id` and `session_id`
  - How many generations, scores, and spans were sent

  This lets an operator confirm post-hoc whether a run reached Langfuse.

## Langfuse and artifact-package — security and known limitations

* **Sensitive data surface**: When live push is on, `conversations.jsonl`
  (and Langfuse Generations) carry full prompt/response text. `redact_secret_values`
  scrubs common token shapes (Bearer, `sk-`/`pk-`, GitHub tokens, some
  `KEY=value`) but is not a complete data loss prevention (DLP) filter — bare keys without a
  recognizable prefix (for example, raw AWS `AKIA…`) can slip through. The artifact
  packager also copies `reports/trace/*.jsonl` and, with the loose mode on by
  default (`HYPERLOOM_SESSION_PACKAGE_LOOSE`), drops them under the package
  destination `HYPERLOOM_SESSION_PACKAGE_DEST`: `/workspace` for the Claw sync when
  it is unset, and `<session dir>/archive/` (mode 2770, the run's group, on the shared storage) in
  a hosted run, where the entry sets it whatever the host passed. If a session might
  contain customer code or secrets, define an explicit retention + access-control
  policy for both the Langfuse project and the package destination, and consider
  disabling live push or loose packaging for those runs.
* **`live push` + `backfill_langfuse` overlap**: Both derive the same
  `trace_id` from `claw_session_id`, so running the offline backfill *after* a
  live run re-emits the generations and decision scores onto the same trace and can
  duplicate observations. Use one path per session, or treat backfill as a
  recovery tool only when live push did not run.
* **`flush_session` is idempotent, and retries only what failed**: the
  session-end reconcile runs as named steps (leftover halves,
  recipe-KB audit, specialist intel, forge steps, GEMM tuning, trajectory,
  decision scores, span close, final SDK flush). Every call runs every step,
  and each step sends only what is still owed, so a duplicated CLOSE step
  won't double-push, a row that failed is retried by the next call, and a
  later call (the shutdown flush after CLOSE) pushes whatever was recorded
  since the earlier one. `counts_final` is `true` when the latest call
  completed every step — a `false` there means the push is still incomplete,
  not that the session was short-lived. Every leg of a resumed session
  (a crash plus a `--resume`, or two shutdown paths racing) reports into the
  same trace. `rows_sent` maps each
  append-only log, by its path under the session directory, to how many of its
  rows were sent: the `trajectory/*.jsonl` shards, the
  recipe-KB audit, specialist intel, forge steps and GEMM tuning logs. A call
  sends only the rows past that count, advancing it one row at a time and
  stopping at the first row it could not send, so that row is retried by the
  next call or the next leg. A v1.1.3 receipt carries the trajectory shard
  counts as `trajectory_rows_sent`, which is still read. `decision_trace.jsonl` is rewritten ts-sorted on
  every export, so decision scores are tracked by the `decision_id` its writer
  stamps on each row instead (`decisions_sent`). Both record what was handed to
  the Langfuse SDK: its flush does not report a failed export, so a row lost in
  export is not re-pushed. The `*_read` counters count only what the current leg
  read past those cursors. The
  one-shot `session_start` / `session_breakdown` pushes are claimed through an
  exclusive marker file (`reports/trace/.session_start.claim`) rather than
  through the receipt read. The receipt also carries `payload_sha256` over its own body; a
  receipt whose hash does not match is ignored on read, so a torn file cannot
  suppress or replay the one-shot `session_start` / breakdown pushes.
* **Package completeness**: `PACKAGE_MANIFEST` describes what was actually
  written, and `included_files` never names a file the package lacks. Check
  `complete` before treating a package as the whole selection; it is `false`
  whenever anything selected is absent, and the reason is itemized in
  `dropped_files` (the bundle caps at 5000 files / 256 MB and a very long
  session can stop it short, alongside `truncated: true`), `failed_files`
  (the write failed) or `refused_files` (the entry was not a regular file
  inside the session — see below). The zip and the loose tree are written
  independently and each carries a manifest describing its own contents.
* **Session boundary**: a session directory is shared-filesystem state that
  agents write into, so the packager only bundles entries that resolve
  inside the session. A symlink pointing out of the session is refused
  rather than followed, which keeps unrelated file content from being
  copied to the dest root and synced onward.
* **Generation duration is ~0**: Both live and backfill stamp a single
  timestamp (`end == start`), so Langfuse shows no meaningful per-Generation
  duration — counts/usage are accurate, latency is not captured.

## Gateway attribution (`HYPERLOOM_LLM_ATTRIBUTION`)

The per-call ledger (`reports/trace/llm_calls.jsonl`) is Hyperloom's own account
of its spend, assembled from calls that report themselves. The gateway keeps a second account,
from the calls it actually metered, and the two disagree whenever a component
spends without a ledger producer. This variable makes the gateway's account
readable along the same axes, by tagging every outbound call with who made it.

Set it to the name of the gateway in front of the deployment; `litellm` is the
only preset today. Leave it unset (the default) and nothing is emitted.

```bash
export HYPERLOOM_LLM_ATTRIBUTION=litellm
```

Each call then carries, on headers that gateway understands:

| Field | Value |
|-------|-------|
| `application` | Always `hyperloom`, to separate this product's spend on a shared gateway. |
| `session` | `CLAW_SESSION_ID`, the same id `session_breakdown.json` records — this is what joins the two accounts. |
| `component` | The producer, e.g. `orchestration`, `kernel`, `critic`, `specialist`. |
| `phase` | The phase the run had reached. |
| `type` | The action executing inside that phase, e.g. `kernel_opt`. |
| `operation` | What the individual call was for, e.g. `generate_candidate`. |

For LiteLLM these arrive as `x-litellm-tags` (a comma-separated list landing in
the `request_tags` column of `LiteLLM_SpendLogs`) and `x-litellm-trace-id` (the
session id, which sets the `session_id` column and propagates to nested calls).
A spend report can then be grouped by component or phase, or reconciled against
the call ledger for one session.

Fields with per-call cardinality (a task or kernel id) are deliberately not in
the tag list: one tag per task would give the rollup as many buckets as there
are tasks. Supporting another gateway means adding a preset in
`src/hyperloom_common/llm/llm_attribution.py`, not setting a different string here.
