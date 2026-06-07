---
name: robustness-agent
description: Independent guardian daemon for Hyperloom inference optimization. Implements the inference_optimizer "robustness" reactor so the Coordinator can call it as a Backend, plus a standalone loop for dev. Owns continuous health monitoring, RCA, and scheduling-police capabilities (kill_task / force_dispatch / prune_branch / escalate_strategy_change).
---

# Robustness Agent

A Python package that ships the `robustness` agent for the
`inference_optimizer` Coordinator and a CLI for standalone use.

The agent observes shared state, the orchestration agent's inbox, and
cluster telemetry; classifies symptoms; and emits Coordinator-validated
intents (alert / escalate_strategy_change / prune_branch / etc.) plus
on-disk findings.

## Quick start

```bash
cd robustness-agent
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"

# Reactor mode. Auto-discovers session_dir and probes
# robustness-server before falling back to local probes.
.venv/bin/robustness-agent
```

## M1 module layout

```
robustness_agent/
├── runtime/
│   ├── __init__.py         # subprocess transport package
│   └── cli.py              # `python -m robustness_agent.runtime.cli tick` entry point
├── role/
│   ├── envelope.py         # IntentType / Intent / build_* helpers (mirror upstream)
│   ├── prompt_inputs.py    # Coordinator prompt -> ReactorContext
│   └── reactor.py          # Reactor.tick() pipeline driver
├── decision/
│   ├── policy_aware.py     # local PolicyGate-equivalent payload guard
│   ├── action_ladder.py    # symptom -> intent (+ Finding) translation (async)
│   └── rca_engine.py       # NoopRcaEngine | LlmRcaEngine + RcaThrottle (M1.5)
├── signals/
│   ├── stall.py / crash.py / event.py / health.py / local_health.py (M1.5)
│   └── classifier.py       # composes the rules and de-duplicates
├── sources/
│   ├── base.py             # Source / SourceData / DegradeRouter
│   ├── server_client.py    # robustness-server REST + Source adapter
│   └── local_probe.py      # local fallback (conductor.db, ps, df, parsed rocm-smi, http probes, log error patterns)
├── findings/sink.py        # JSONL append sink for Findings
├── factory.py              # Config -> ReactorBundle (build_reactor_components)
├── config.py               # discovery + tunables
├── main.py                 # standalone reactor CLI
└── conductor.py / monitors / checks / providers
```

Use the standalone reactor CLI above for dev/smoke runs, or the
subprocess transport below for Coordinator integration.

## Coordinator integration (subprocess transport)

The architectural blueprint follows `critic-agent`: hosts (the
Coordinator, smoke harnesses, operator tooling) shell out to a CLI
that reads a JSON request and writes a JSON envelope. There is no
in-process Backend adapter — the agent runs in a child Python process
so its dependency tree (httpx, openai SDK, sqlite3) stays isolated
from the host.

```bash
python -m robustness_agent.runtime.cli tick \
    --request request.json \
    --out emit.json
```

`request.json` (host-built):
```json
{
  "kind": "coordinator_inbox",
  "session_id": "sess-1",
  "raw_prompt": "=== Shared session state ===\n...",
  "context": {"tick_index": 0, "now_unix": 1700000000.0},
  "options": {"session_dir": "/tmp/sess-1",
              "robustness_server_url": "http://...",
              "llm_rca_enabled": false,
              "metrics_window_s": 300}
}
```

`emit.json` (CLI-emitted):
```json
{
  "intent_envelope": {"intents": [{"intent_type": "alert", "payload": {...}}, ...]},
  "session_id":   "sess-1",
  "tick_index":   1,
  "parse_warnings": []
}
```

`intent_envelope` follows the same schema as `critic-agent`'s
`commit-review` output, validated host-side by
`inference_optimizer.protocol.intent.validate_envelope`.
Exit code `0` = logical success (zero or more intents); `2` = adapter /
configuration bug.

The host-side wrapper that drives this subprocess lives in
`inference_optimizer/orchestrator/backends/robustness_agent.py:RobustnessAgentBackend`,
mirroring the layout of `CriticAgentBackend`. End-to-end tests in
`inference_optimizer/tests/test_p2_robustness_agent_e2e.py` and
`robustness-agent/tests/test_runtime_cli.py` together cover the full
host -> subprocess -> envelope -> upstream PolicyGate path.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SESSION_DIR` | no | scan known paths | Path containing `storage/conductor.db`; the FindingSink writes under `{session_dir}/agents/robustness/findings/{session_id}.jsonl`. |
| `ROBUSTNESS_SERVER_URL` | no | scan known DNS | M1 primary data source; empty disables the primary path and forces local-only mode. |
| `ROBUST_ANALYZER_URL` | no | scan known DNS | Optional hybrid-provider endpoint used during data-source discovery. |
| `OPENAI_BASE_URL` | no | — | LLM endpoint for RCA (used as `llm_base_url`). |
| `SAFE_API_KEY` | no | — | API key for the LLM proxy (used as `llm_api_key`). |
| `LLM_MODEL` | no | `claude-opus-4-7` | RCA model name. |
| `ROBUSTNESS_LLM_RCA_DISABLED` | no | unset | Set to `1` to forcibly disable the LlmRcaEngine even when credentials are present. |

## Symptom -> intent mapping (M1 / M1.5)

| Symptom | Severity | Intents emitted | Source |
|---------|----------|-----------------|--------|
| `agent_stall` (≥ stall_timeout_s) | medium | `alert(medium)` | M1 |
| `agent_stall` (≥ severity_high_after_s) | high | `alert(high)` + `escalate_strategy_change` | M1 |
| `crash_count_rising` (≥ 2) | medium | `alert(medium)` | M1 |
| `crash_count_high` (≥ 5) | high | `alert(high)` + `escalate_strategy_change` | M1 |
| `crash_count_emergency` (≥ 10) | high | `alert(high)` + `escalate_strategy_change` | M1 |
| `repeated_policy_denied` (≥ 3) | medium | `alert(medium)` | M1 |
| `repeated_failure` (≥ 2 same family) | medium | `alert(medium)` (high tier triggers `prune_branch`) | M1 |
| `pod_not_running` (Failed) | high | `alert(high)` | M1 |
| `pod_not_running` (other non-Running) | medium | `alert(medium)` | M1 |
| `pod_no_metrics` (≥ no_metrics_warn_s) | low | `send_message(observation)` | M1 |
| `local_server_unreachable` (any target down) | medium / high (all down) | `alert(medium)` / `alert(high)` | M1.5 |
| `log_error_pattern` (CUDA OOM / NCCL / segfault) | high | `alert(high)` + `escalate_strategy_change` | M1.5 |
| `log_error_pattern` (RuntimeError / generic) | medium | `alert(medium)` | M1.5 |
| `gpu_thermal_high` (≥ warn_c) | medium | `alert(medium)` | M1.5 |
| `gpu_thermal_high` (≥ crit_c) | high | `alert(high)` + `escalate_strategy_change` | M1.5 |
| (no symptoms) | — | `send_message(heartbeat)` | M1 |

Cooldown: identical `(symptom_name, subject)` keys are silenced for
`config.cooldown_ticks` ticks (default 5) to avoid inbox flooding.

## Data sources (M1 / M1.5)

* **Primary:** `robustness-server`
  * `/api/v1/sessions/{id}/pods`
  * `/api/v1/sessions/{id}/events`
  * `/api/v1/sessions/{id}/summary`
* **Fallback:** local probes
  * `conductor.db` (read-only) for Coordinator events
  * `shutil.disk_usage`, `ps`, parsed `rocm-smi --csv` / `nvidia-smi`
  * `Config.health_probe_targets[]` — local HTTP `/health` probes for
    inference servers running on the same host (M1.5)
  * tail of a configured log file + error-pattern extraction
    (`CUDA out of memory`, `NCCL error`, `Segmentation fault`, etc.)

LocalProbe stays small-scope by design: it only collects what the
agent itself can see. GPU time-series, workload inference health, and
node-level fault detection stay with primus-robust + robustness-server
(see `docs/robustness-agent-implementation-plan.md` §6.1 for the
ownership matrix).

`DegradeRouter` switches to the fallback after
`source_fail_threshold` (default 3) consecutive primary failures and
re-probes the primary every `source_recheck_interval_s` (default 30s).
State transitions emit one WARN log; no spam in steady state.

## LLM RCA (M1.5)

When `llm_base_url` and `llm_api_key` are both set (and
`ROBUSTNESS_LLM_RCA_DISABLED` is not `1`), the factory wires
`LlmRcaEngine` instead of `NoopRcaEngine`. The engine talks to an
OpenAI-compatible chat-completions endpoint and writes the response
to `Finding.rca_text`.

Throttle defaults (override on `Config`):

* `llm_rca_severity_min="high"` — only `high`-severity symptoms call
  the LLM. Set to `"medium"` to include medium-severity findings.
* `llm_rca_cooldown_s=60.0` — same `(symptom_name, subject)` is not
  re-summarized within this window.
* `llm_rca_max_calls_per_tick=1` — hard cap per reactor tick.

If the LLM call times out, returns a non-2xx, or otherwise fails, the
ActionLadder swallows the error and emits the intent without
`rca_text`. RCA never blocks intent delivery.

## Findings on disk

Each tick that emits a non-heartbeat intent writes one
`Finding` JSON line to:

```
{session_dir}/agents/robustness/findings/{session_id}.jsonl
```

Fields: `tick_index`, `timestamp_unix`, `symptom_name`, `severity`,
`summary`, `intents` (envelope dicts), `evidence`, `rca_text`.

These records are the M5 hand-off point: a future milestone POSTs them
to the robustness-server for dashboards / alerting; in M1 they remain
local-only.

## Session-end postmortem (L1 + L2)

When the Coordinator sets `state.json::stop_reason` (run wind-down)
the reactor fires :class:`robustness_agent.finalize.PostmortemFinalizer`
exactly once. It aggregates the in-session findings + per-task
`runs/<action>/<task_id>/result.json` into:

```
{session_dir}/reports/robustness_postmortem.md   # flashpoint + catalogue + per-action summary
{session_dir}/reports/decision_trace.json        # machine-readable per-task ledger
{session_dir}/reports/.robustness_finalized      # idempotency marker
```

Disable via `Config.finalize_enabled=False`. Operators can re-run the
finalizer post-hoc via
`robustness_agent.finalize.postmortem.finalize_session(session_dir, session_id=...)`
(noop when the marker exists).

## Critic feedback loop (L4)

The Critic agent's `prepare-review` phase reads the most recent N HIGH-
severity findings from `findings/<session>.jsonl` and injects them
into `JudgeBundle.robustness_priors` so the LLM sees "what already
broke this session" before producing a fresh proposal. Discovery
order:

1. `$CRITIC_ROBUSTNESS_FINDINGS_DIR` — directory containing
   `<session>.jsonl` (explicit override).
2. `$ROBUSTNESS_AGENT_SESSION_DIR` — robustness-agent's `session_dir`;
   the runtime CLI exports this automatically so co-deployed
   Critic + Robustness picks up findings without operator setup.

Knobs (env): `CRITIC_ROBUSTNESS_PRIORS_LIMIT` (default 5),
`CRITIC_ROBUSTNESS_PRIORS_MIN_SEVERITY` (default `high`),
`CRITIC_ROBUSTNESS_PRIORS_DISABLED=1` (kill switch).

## Roadmap beyond M1.5

* **M2** — robustness-server `/api/v1/cluster/*` proxies (faults / GPU
  time-series / node info). Agent's `signals/{gpu,disk,health}` start
  preferring server data; LocalProbe stays as the disconnected fallback.
* **M3** — multi-cli transport (`inbox.jsonl` / `outbox.jsonl`); same
  reactor, different adapter.
* **M4** — scheduling-police hard actions (`prune_branch`,
  `force_dispatch`, `kill_task`, `delegate{recover|server_lifecycle|accuracy_gate}`),
  gated behind `ROBUSTNESS_AGENT_ENABLE_HARD_ACTIONS`.
* **M5** — findings publisher to robustness-server for cross-session
  reporting and advisory pull-back.
