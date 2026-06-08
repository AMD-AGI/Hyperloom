# Operator Scripts Reference

A short reference for the operator-facing scripts under
`inference_optimizer/scripts/`. These are not part of the agent loop —
they are utilities you run by hand against a finished or in-progress
session directory.

All scripts respect the `$USER_DATA_PATH` env (default
`/workspace/hyperloom`) when no explicit `--session-dir` is given. See
[`ENV_AND_AUTH.md`](ENV_AND_AUTH.md) §4.

---

## 1. `dump_session_breakdown.py`

Produce a `session_breakdown.json` from a session directory. Same
builder as the live Coordinator `session_breakdown` action and the
`cli.py` finally-block safety net.

**Use this when:**

* You want to (re)produce the breakdown for a historical WekaFS
  session.
* A live session crashed before reaching the closing phase and you
  want the partial breakdown anyway.
* You need to bulk-export breakdowns for downstream indexing.

### Usage

```bash
# Live session in the current sandbox (USER_DATA_PATH or /workspace/hyperloom)
python -m inference_optimizer.scripts.dump_session_breakdown

# Historical session on WekaFS
python -m inference_optimizer.scripts.dump_session_breakdown \
    --session-dir /wekafs/users/zgong/inference_optimizer-sessions/<sid>

# Override output path (don't touch session_dir)
python -m inference_optimizer.scripts.dump_session_breakdown \
    --session-dir <SD> --output /tmp/breakdown-<sid>.json

# Bulk historical
for d in /wekafs/users/*/inference_optimizer-sessions/*; do
    [ -d "$d" ] || continue
    python -m inference_optimizer.scripts.dump_session_breakdown \
        --session-dir "$d" > /dev/null
done
```

The default output path is `<session_dir>/session_breakdown.json`
(overwrites if present; the file is rebuilt deterministically from raw
artefacts).

### Output

`session_breakdown.json` conforming to
[`INTEGRATION_SESSION_BREAKDOWN.md`](INTEGRATION_SESSION_BREAKDOWN.md).
The script exits 0 on success, prints a one-line summary, and writes
collector warnings to the `warnings[]` field rather than failing.

---

## 2. `dump_session_report.py`

Render a markdown session report from a `session_breakdown.json`.
Deterministic by default; optionally LLM-polished when an
OpenAI-compatible endpoint is configured.

**Use this when:**

* You want a human-readable summary to paste into a PR / Slack /
  email.
* You want to generate the same report for many sessions in bulk.

### Usage

```bash
# Deterministic only (no LLM):
python -m inference_optimizer.scripts.dump_session_report \
    --input  /wekafs/.../session_breakdown.json \
    --output /wekafs/.../session_report.md

# With LLM-polished prose (OpenAI-compatible endpoint):
HYPERLOOM_REPORT_LLM_BACKEND=openai \
OPENAI_BASE_URL=http://127.0.0.1:4002/v1 \
OPENAI_API_KEY=... \
python -m inference_optimizer.scripts.dump_session_report \
    --input  /wekafs/.../session_breakdown.json \
    --output /wekafs/.../session_report.md
```

When `--output` is omitted the report is written to
`<session_dir>/session_report.md` next to the input file. The LLM
user prompt and raw response (when used) are persisted alongside as
`session_report_prompt.json` / `session_report_llm_raw.txt` so
hallucinations can be audited after the fact.

### LLM hardening

* The deterministic skeleton (headline numbers, action_path,
  kernel_lifecycle counts) is generated **without** the LLM; the LLM
  only rewrites prose.
* If the LLM call fails (timeout, 5xx, malformed response), the
  script falls back to the deterministic report and exits 0.
* If you do not want any LLM call, simply leave
  `HYPERLOOM_REPORT_LLM_BACKEND` unset.

---

## 3. `event_counts.py`

Print recent action / proposal / kernel counts from a session's
`coordinator.db`.

**Use this when:**

* You want a quick "is this session making progress?" check without
  reading logs.
* You are debugging an apparent stall and want to see what kind of
  events are landing.

### Usage

```bash
python -m inference_optimizer.scripts.event_counts            # default session_dir
python -m inference_optimizer.scripts.event_counts /path/to/session
```

Reads at most the last 500 events from
`$SESSION_DIR/storage/coordinator.db` and emits a JSON object of
`{category: count}`. Exit code 2 if the DB is missing.

### Example output

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

## 4. A/B helper scripts (advanced)

The same scripts directory also contains:

* `ab_torch_compile_kernels.py`
* `ab_torch_compile_magpie.py`

These are internal A/B harnesses used during torch.compile
investigation work; they are not part of the customer-facing workflow
and are documented in their respective module docstrings. Treat them
as reference implementations rather than supported operator tools.

---

## See also

* [`INTEGRATION_SESSION_BREAKDOWN.md`](INTEGRATION_SESSION_BREAKDOWN.md)
  — the schema produced by `dump_session_breakdown.py`.
* [`OPERATIONS.md`](OPERATIONS.md) — retention recommendations,
  including which scripts' outputs to back up long-term.
* [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — symptoms vs which
  script to reach for first.
