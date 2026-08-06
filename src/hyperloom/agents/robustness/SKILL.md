---
name: robustness-agent
description: Independent guardian daemon for Hyperloom inference optimization. Implements the inference_optimizer "robustness" reactor so the Coordinator can call it as a Backend, plus a standalone loop for dev. Owns continuous health monitoring, RCA, and scheduling-police capabilities (kill_task / prune_branch / delegate).
---

# Robustness Agent

A Python package that ships the `robustness` agent for the
`inference_optimizer` Coordinator and a CLI for standalone use.

The agent observes shared state, the orchestration agent's inbox, and
cluster telemetry; classifies symptoms; and emits Coordinator-validated
intents (alert / prune_branch / kill_task / delegate / etc.) plus
on-disk findings.

## Quick start

```bash
# from the repo root — the repo is a single distribution
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"

# Reactor mode. Auto-discovers session_dir and probes
# robustness-server before falling back to local probes.
.venv/bin/robustness-agent
```

This console script is the standalone dev path only; the orchestrator
drives the same reactor via
`python -m hyperloom.agents.robustness.runtime.cli`.

## Module layout

```
src/hyperloom/agents/robustness/
├── runtime/
│   ├── __init__.py         # subprocess transport package
│   └── cli.py              # `python -m hyperloom.agents.robustness.runtime.cli tick` entry point
├── role/
│   ├── envelope.py         # IntentType / Intent / build_* helpers (mirror upstream)
│   ├── prompt_inputs.py    # Coordinator prompt -> ReactorContext
│   ├── findings.py         # JSONL append sink for Findings
│   ├── postmortem.py       # session-end postmortem finalizer
│   └── reactor.py          # Reactor.tick() pipeline driver
├── decision/
│   ├── policy_aware.py     # local PolicyGate-equivalent payload guard
│   ├── action_ladder.py    # symptom -> intent (+ Finding) translation (async)
│   └── rca_engine.py       # NoopRcaEngine | LlmRcaEngine | AnthropicRcaEngine + RcaThrottle
├── signals/                # 17 detector modules
│   ├── classifier.py       # composes the rules and de-duplicates
│   └── symptom.py          # Symptom / SymptomSeverity dataclasses
├── sources/
│   ├── base.py             # Source / SourceData / DegradeRouter
│   ├── server_client.py    # robustness-server REST + Source adapter
│   ├── cluster_decoder.py  # cluster pods/GPU/fault payload decoding
│   └── local_probe.py      # local fallback (coordinator.db, ps, df, parsed rocm-smi, http probes, log error patterns)
├── factory.py              # Config -> ReactorBundle (build_reactor_components)
├── config.py               # discovery + tunables
├── state_store.py          # per-detector state persisted across ticks
└── main.py                 # standalone reactor CLI
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
python -m hyperloom.agents.robustness.runtime.cli tick \
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
`hyperloom.inference_optimizer.protocol.intent.validate_envelope`.
Exit code `0` = logical success (zero or more intents); `2` = adapter /
configuration bug.

The host-side wrapper that drives this subprocess lives in
`src/hyperloom/orchestrator/roles/robustness_agent.py:RobustnessAgentBackend`,
mirroring the layout of `CriticAgentBackend`. End-to-end tests in
`src/hyperloom/inference_optimizer/tests/test_robustness_agent_e2e.py` and
`src/hyperloom/agents/robustness/tests/test_runtime_cli.py` together cover the full
host -> subprocess -> envelope -> upstream PolicyGate path.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SESSION_DIR` | no | scan known paths | Path containing `storage/coordinator.db`; the FindingSink writes under `{session_dir}/agents/robustness/findings/{session_id}.jsonl`. |
| `ROBUSTNESS_SERVER_URL` | no | scan known DNS | M1 primary data source; empty disables the primary path and forces local-only mode. |
| `OPENAI_BASE_URL` | no | — | LLM endpoint for RCA (used as `llm_base_url`). |
| `OPENAI_API_KEY` | no | — | API key for the LLM proxy (used as `llm_api_key`). |
| `ROBUSTNESS_LLM_MODEL` | no | — | RCA model name; takes precedence over `LLM_MODEL`. |
| `LLM_MODEL` | no | provider default | RCA model name. With neither override set the chain is `DEEPSEEK_MODEL` / `deepseek-v4-pro` (when DeepSeek env is present), else openai: `OPENAI_MODEL` → `CODEX_MODEL` → `gpt-5.6-sol`, else anthropic: `ANTHROPIC_MODEL` → `CLAUDE_MODEL` → `claude-opus-5`. |
| `ROBUSTNESS_LLM_RCA_DISABLED` | no | unset | Set to `1` to forcibly disable the LlmRcaEngine even when credentials are present. |

## Symptom -> intent mapping (M1 / M1.5)

| Symptom | Severity | Intents emitted | Source |
|---------|----------|-----------------|--------|
| `agent_stall` (≥ stall_timeout_s) | medium | `alert(medium)` | M1 |
| `agent_stall` (≥ severity_high_after_s) | high | `alert(high)` | M1 |
| `crash_count_rising` (≥ 2) | medium | `alert(medium)` | M1 |
| `crash_count_high` (≥ 5) | high | `alert(high)` | M1 |
| `crash_count_emergency` (≥ 10) | high | `alert(high)` | M1 |
| `repeated_policy_denied` (≥ 3) | medium | `alert(medium)` | M1 |
| `repeated_failure` (≥ 2 same family) | medium / high (≥ prune threshold) | `alert(medium)`; HIGH tier also emits `prune_branch(family)` | M1 |
| `pod_not_running` (Failed) | high | `alert(high)` | M1 |
| `pod_not_running` (other non-Running) | medium | `alert(medium)` | M1 |
| `pod_no_metrics` (≥ no_metrics_warn_s) | low | `send_message(observation)` | M1 |
| `local_server_unreachable` (any target down) | medium / high (all down) | `alert(medium)` / `alert(high)` | M1.5 |
| `log_error_pattern` (CUDA OOM / NCCL / segfault) | high | `alert(high)` | M1.5 |
| `log_error_pattern` (RuntimeError / generic) | medium | `alert(medium)` | M1.5 |
| `gpu_thermal_high` (≥ warn_c) | medium | `alert(medium)` | M1.5 |
| `gpu_thermal_high` (≥ crit_c) | high | `alert(high)` | M1.5 |
| `stale_lease` | high | `alert(high)` + `kill_task(task_id)` | M1 |
| `gpu_memory_leaked` | high | `alert(high)` + `delegate(recover, force_gpu_cleanup=True)` | M1 |
| `deadline_warning` / `deadline_imminent` / `deadline_hard_cutoff` / `recover_unsuccessful` | high | `alert(high)` + `delegate(report)` | M1 |
| `same_payload_loop` / `kernel_opt_no_progress` / `geak_budget_starvation` / `amdahl_kernel_ceiling_low` | high | `alert(high)` + `prune_branch(family)` | M1 |
| (no symptoms) | — | `send_message(heartbeat)` | M1 |

Every other HIGH symptom is strategic: the recommendation rides the
alert's `detail.suggestion` field and the ladder never auto-emits
`escalate_strategy_change` — the intent stays PolicyGate-allowed for
explicit drives, but Orchestration owns the phase-advance decision.
`delegate` is constrained to the `ROBUSTNESS_DELEGATE_ACTIONS`
allowlist (`accuracy_gate` / `recover` / `report` / `server_lifecycle`).

Cooldown: identical `(symptom_name, subject)` keys are silenced for
`config.cooldown_ticks` ticks (default 5) to avoid inbox flooding.

## Data sources (M1 / M1.5)

* **Primary:** `robustness-server`
  * `/api/v1/sessions/{id}/pods`
  * `/api/v1/sessions/{id}/events`
  * `/api/v1/sessions/{id}/summary`
  * `/api/v1/cluster/faults` (on by default)
  * `/api/v1/cluster/workloads/{id}/hierarchy`
  * `/api/v1/cluster/pods/{ns}/{name}/metrics` — gated by
    `Config.enable_cluster_pod_metrics` (default `False`, env-settable)
* **Fallback:** local probes
  * `coordinator.db` (read-only) for Coordinator events
  * `shutil.disk_usage`, `ps`, parsed `rocm-smi --csv` / `nvidia-smi`
  * `Config.health_probe_targets[]` — local HTTP `/health` probes for
    inference servers running on the same host (M1.5)
  * tail of a configured log file + error-pattern extraction
    (`CUDA out of memory`, `NCCL error`, `Segmentation fault`, etc.)

LocalProbe stays small-scope by design: it only collects what the
agent itself can see. GPU time-series, workload inference health, and
node-level fault detection stay with primus-robust + robustness-server
ownership.

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

These records are the hand-off point for a future findings publisher
that POSTs them to the robustness-server for dashboards / alerting;
today they remain local-only.

## Session-end postmortem (L1 + L2)

When the Coordinator sets `state.json::stop_reason` (run wind-down)
the reactor fires :class:`hyperloom.agents.robustness.role.postmortem.PostmortemFinalizer`
exactly once. It aggregates the in-session findings + per-task
`runs/<action>/<task_id>/result.json` into:

```
{session_dir}/reports/robustness_postmortem.md   # flashpoint + catalogue + per-action summary
{session_dir}/reports/decision_trace.json        # machine-readable per-task ledger
{session_dir}/reports/.robustness_finalized      # idempotency marker
```

Disable via `Config.finalize_enabled=False`. Operators can re-run the
finalizer post-hoc via
`hyperloom.agents.robustness.role.postmortem.finalize_session(session_dir, session_id=...)`
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

## Roadmap

* **Multi-cli transport** — an `inbox.jsonl` / `outbox.jsonl` adapter
  feeding the same reactor. Not shipped: the only transport today is the
  subprocess one above, and `ReactorContext` is only ever built from a
  rendered Coordinator prompt.
* **Findings publisher** — POST the on-disk findings to
  robustness-server for cross-session reporting and advisory pull-back.
  Nothing ships today; findings stay local-only.
