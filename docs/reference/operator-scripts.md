---
myst:
    html_meta:
        "description": "Reference for Hyperloom operator scripts: dump_session_breakdown, dump_session_report, event_counts, and estimate_no_run. Use these utilities to inspect, export, and report on session data, and to estimate uplift before running."
        "keywords": "Hyperloom, operator scripts, session breakdown, session report, event counts, uplift estimate, Recipe KB, LLM inference, AMD GPU, ROCm, debugging, observability, operator tools"
---
# Hyperloom operator scripts

A short reference for the operator-facing scripts under
`src/hyperloom/inference_optimizer/tools/`. These are not part of the agent loop —
they are utilities you run by hand, most of them against a finished or
in-progress session directory. The exception is `estimate_no_run.py`, which
reads the Recipe KB rather than a session.

When no explicit `--session-dir` is given, scripts resolve the active session in
two steps: `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR` when set, otherwise the
workspace root (`USER_DATA_PATH`, falling back to `/workspace/hyperloom`). This
does **not** auto-discover the latest `$USER_DATA_PATH/<model>/<ts>/` per-session
subdir — under the per-model timestamp layout, pass `--session-dir` explicitly
(or rely on `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`, which the CLI sets during
a run). See [Hyperloom authentication and credentials](authentication.md).

---

## `dump_session_breakdown.py`

Produce a `session_breakdown.json` from a session directory. Same
builder as the live Coordinator `session_breakdown` action and the
`cli.py` finally-block safety net.

Use this when:

* You want to (re)produce the breakdown for a historical session on a shared
  filesystem.
* A live session crashed before reaching the closing phase and you
  want the partial breakdown anyway.
* You need to bulk-export breakdowns for downstream indexing.

### Usage

Use these commands to produce a session breakdown.

```bash
# Live session in the current sandbox (USER_DATA_PATH or /workspace/hyperloom)
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown

# Historical session on a shared filesystem
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
    --session-dir /shared/hyperloom-sessions/<user>/<sid>

# Override output path (don't touch session_dir)
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
    --session-dir <SD> --output /tmp/breakdown-<sid>.json

# Bulk historical
for d in /shared/hyperloom-sessions/*/*; do
    [ -d "$d" ] || continue
    python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
        --session-dir "$d" > /dev/null
done
```

The default output path is `<session_dir>/session_breakdown.json`
(overwrites if present; the file is rebuilt deterministically from raw
artifacts).

### Output

`session_breakdown.json` conforming to
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md).
The script exits 0 on success, prints a one-line summary, and writes
collector warnings to the `warnings[]` field rather than failing.

---

## `dump_session_report.py`

Render a markdown session report from a `session_breakdown.json`.
Deterministic by default; optionally large language model (LLM)-polished when an
OpenAI-compatible endpoint is configured.

Use this when:

* You want a human-readable summary to paste into a PR, Slack, or
  email.
* You want to generate the same report for many sessions in bulk.

### Usage

Use the following commands to render a session report.

```bash
# Deterministic only (no LLM):
python -m hyperloom.inference_optimizer.tools.dump_session_report \
    --input  /shared/hyperloom-sessions/<user>/<sid>/session_breakdown.json \
    --output /shared/hyperloom-sessions/<user>/<sid>/session_report.md

# With LLM-polished prose (OpenAI-compatible endpoint):
HYPERLOOM_REPORT_LLM_BACKEND=openai \
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1 \
OPENAI_API_KEY=... \
python -m hyperloom.inference_optimizer.tools.dump_session_report \
    --input  /shared/hyperloom-sessions/<user>/<sid>/session_breakdown.json \
    --output /shared/hyperloom-sessions/<user>/<sid>/session_report.md
```

When `--output` is omitted the report is written to
`<session_dir>/session_report.md` next to the input file. The LLM
user prompt and raw response (when used) are persisted alongside as
`session_report_prompt.json` / `session_report_llm_raw.txt` so
hallucinations can be audited after the fact.

### LLM hardening

The script applies the following safeguards when LLM polishing is enabled.

* The deterministic skeleton (headline numbers, action_path,
  kernel_lifecycle counts) is generated *without* the LLM; the LLM
  only rewrites prose.
* If the LLM call fails (timeout, 5xx, malformed response), the
  script falls back to the deterministic report and exits 0.
* If you do not want any LLM call, leave
  `HYPERLOOM_REPORT_LLM_BACKEND` unset.

---

## `event_counts.py`

Print recent action / proposal / kernel counts from a session's
`coordinator.db`.

Use this when:

* You want a quick "is this session making progress?" check without
  reading logs.
* You are debugging an apparent stall and want to see what kind of
  events are landing.

### Usage

Use the following commands to print event counts for a session.

```bash
python -m hyperloom.inference_optimizer.tools.event_counts            # default session_dir
python -m hyperloom.inference_optimizer.tools.event_counts /path/to/session
```

Reads at most the last 500 events from
`$SESSION_DIR/storage/coordinator.db` and emits a JSON object of
`{category: count}`. Exit code 2 if the database (DB) is missing.

### Example output

The script emits a JSON object of event categories and counts.

```json
{
  "delegated:kernel_optimization:succeeded": 7,
  "delegated:tracelens_analysis:succeeded": 1,
  "kernel_request:kernel_optimization": 7,
  "kernel_request:tracelens_analysis": 1,
  "kernel_response:kernel_optimization:KEEP": 3,
  "kernel_response:kernel_optimization:NEEDS_REVIEW": 4,
  "proposal:explore": 12,
  "proposal:specialist": 4,
  "proposal:kernel_opt": 5
}
```

A long run with healthy progress has roughly proportional
`proposal:*` and `delegated:*:succeeded` counts. A stuck run typically
shows many `kernel_request:*` and few `kernel_response:*`.

---

## `estimate_no_run.py`

Estimate uplift for a target from the Recipe KB, without running a session.
Unlike the scripts above this one reads no session directory at all: it pulls
per-session envelopes from the KB Store and reports what prior sessions already
settled for a scope.

Use this when:

* You are triaging a queue of targets and want to know which ones prior
  sessions suggest are worth GPU hours.
* You want the parallelism layout that won inside a fixed GPU count for a
  model/precision/shape you are about to launch.
* You want a session's expected gain range to sanity-check a result you already
  have.

### Usage

Use these commands to estimate uplift for a scope.

```bash
# Whole board, live store, token from a file
python -m hyperloom.inference_optimizer.tools.estimate_no_run \
    --kb-store-url https://host/knowledge-base \
    --kb-store-token-file ~/.secrets/kb_store_token \
    --hardware mi355x --framework-name sglang

# One replay scope, credentials from the environment
export KB_STORE_URL=... KB_STORE_TOKEN=...
python -m hyperloom.inference_optimizer.tools.estimate_no_run \
    --hardware mi355x --model <MODEL> --precision mxfp4 \
    --tp 8 --isl 1024 --osl 256

# Named identities instead of a search (repeatable; still reads the store)
python -m hyperloom.inference_optimizer.tools.estimate_no_run \
    --canonical-id <CID> --canonical-id <CID2>

# A saved pool of session envelopes, no network at all
python -m hyperloom.inference_optimizer.tools.estimate_no_run \
    --input prior_sessions.json --tp 8 --isl 1024 --osl 256
```

Credentials resolve in the order `--kb-store-token`, then
`--kb-store-token-file`, then `KB_STORE_TOKEN`; prefer the file form so the
secret stays out of shell history and `ps`. Exit code 2 when no store URL is
configured, and no network call is attempted in that case. See
[Hyperloom authentication and credentials](authentication.md).

### Output

A JSON report on stdout, or to `--output`. It echoes the store URL but never
the token. The fields to read first:

* `historical`: p50 and p90 validated end-to-end (E2E) gain across the pool.
* `by_shape`: per `tp/conc/isl/osl` bucket, the p50 gain plus p50 and best
  throughput. Read this instead of `historical` when the pool spans shapes.
* `sharding_whatif`: accepted parallelism layouts ranked per replay scope,
  with the vLLM and SGLang spellings of tp/dp/ep/pp normalized.
* `pool_warnings`: raised whenever the pool mixes models, boards, frameworks,
  versions, precisions, or shapes — a median across those is not a prior for
  any of them.
* `limitations` and `sessions_scored`: how much the report is actually standing
  on. Layout arms are frequently `n=1`, so treat rankings as directional.

A KB record keeps only the layout its session settled on, so `sharding_whatif`
carries `winners_only: true`: a layout that is absent was untried or
unpublished, not beaten.

---

## Additional operator tools

The same tools package also contains smaller utilities that are useful during
incident response or launch validation:

* `backfill_langfuse.py`: replay one finished session's `reports/trace/` into
  Langfuse after the fact:
  `python -m hyperloom.inference_optimizer.tools.backfill_langfuse --session-dir <SD> [--dry-run]`.
* `preflight_optimizer.py`: launcher-side local preflight for stale serving
  processes, torch/ROCm device visibility, GPU VRAM occupancy (exits non-zero
  when any card exceeds 1% of its total capacity), and model path existence:
  `python src/hyperloom/inference_optimizer/tools/preflight_optimizer.py MODEL_PATH`.
  A non-zero exit must abort the launch.
* `read_optimizer_state.py`: concise `state.json` / lifecycle summary:
  `python src/hyperloom/inference_optimizer/tools/read_optimizer_state.py SESSION_DIR`.
* `robustness_monitor.sh.example`: shell example for polling robustness
  findings around a session; copy/adapt it for local operator workflows.

---

## Related topics

Use these resources for related reference information:

* [`session_breakdown.json` integration in Hyperloom](session-breakdown.md): The schema produced by `dump_session_breakdown.py`.
* [Hyperloom self-hosting and operations guide](operations.md): Retention recommendations, including which scripts' outputs to back up long-term.
* [Troubleshooting Hyperloom](troubleshooting.md): Symptoms vs which script to reach for first.
* [Integrate Recipe knowledge base in Hyperloom](integrate-kb.md): The store `estimate_no_run.py` reads, and what a session publishes to it.
