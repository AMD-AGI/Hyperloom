---
name: robustness-agent
description: Independent guardian daemon for Hyperloom inference optimization. Implements the inference_optimizer "robustness" reactor so the Coordinator can call it as a Backend, plus a standalone loop for dev. Owns continuous health monitoring, RCA, and scheduling-police capabilities (prune_branch / delegate).
---

# Robustness Agent

A Python package that ships the `robustness` agent for the
`inference_optimizer` Coordinator and a CLI for standalone use.

The agent observes shared state, the orchestration agent's inbox, and
cluster telemetry; classifies symptoms; and emits Coordinator-validated
intents (alert / prune_branch / delegate / etc.) plus on-disk findings.

## Quick start

```bash
# from the repo root — the repo is a single distribution
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"

# Reactor mode. Auto-discovers session_dir and runs the local probes.
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
│   └── local_probe.py      # the collector (coordinator.db, ps, df, parsed rocm-smi, http probes, log error patterns)
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
              "llm_rca_enabled": false,
              "disable_local_probe": false}
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
| `OPENAI_BASE_URL` | no | — | LLM endpoint for RCA (used as `llm_base_url`). |
| `OPENAI_API_KEY` | no | — | API key for the LLM proxy (used as `llm_api_key`). |
| `ROBUSTNESS_LLM_MODEL` | no | — | RCA model name; takes precedence over `LLM_MODEL`. |
| `LLM_MODEL` | no | provider default | RCA model name. With neither override set the chain is openai: `OPENAI_MODEL` → `CODEX_MODEL` → `gpt-5.6-sol`, else anthropic: `ANTHROPIC_MODEL` → `CLAUDE_MODEL` → `claude-opus-5`. |
| `ROBUSTNESS_LLM_RCA_DISABLED` | no | unset | Set to `1` to forcibly disable the LlmRcaEngine even when credentials are present. |

`Config.discover()` reads the variables above plus the deployment-shape ones
(`ROBUSTNESS_DISABLE_LOCAL_PROBE`, `ROBUSTNESS_NODES`) and nothing else. Every
threshold — stall timeouts, disk and shm percentages, GPU temperatures — is a
field on `Config` with a default in `config.py`, changed in code or by whoever
constructs the `Config`, not from the environment.

## Symptom -> intent mapping

Complete inventory. The `Rule` column is the `SignalSpec.name` from
`_SIGNAL_REGISTRY` in `signals/classifier.py`, which is also the module the
rule lives in (`signals/<rule>.py`; `ray_pending` and `kernel_pipeline` share
`kernel_pipeline.py`, and the three preflight rules share `preflight.py`).

Severity drives the ladder tier: low emits `send_message(observation)`, medium
emits `alert(medium)`, high emits `alert(high)` plus any remediation intent
listed below.

| Symptom | Severity | Intents emitted | Rule |
|---------|----------|-----------------|------|
| `agent_stall` (≥ stall_timeout_s) | medium | `alert(medium)` | `stall` |
| `agent_stall` (≥ severity_high_after_s) | high | `alert(high)` | `stall` |
| `agent_quiet_work_progressing` (own dispatched work reported within `stall_timeout_s`) | low | `send_message(observation)` | `stall` |
| `crash_count_rising` (≥ 2) | medium | `alert(medium)` | `crash` |
| `crash_count_high` (≥ 5) | high | `alert(high)` | `crash` |
| `crash_count_emergency` (≥ 10) | high | `alert(high)` | `crash` |
| `repeated_policy_denied` (≥ 3) | medium | `alert(medium)` | `event` |
| `repeated_failure` (≥ 2 same family) | medium / high (≥ prune threshold) | `alert(...)`; HIGH tier also emits `prune_branch(family)` | `event` |
| `idempotency_replay` | medium | `alert(medium)` | `event` |
| `recover_unsuccessful` | high | `alert(high)` | `event` |
| `local_server_unreachable` (any target down) | medium / high (all down) | `alert(medium)` / `alert(high)` | `local_health` |
| `local_server_unreachable`, no server process and no benchmark client of this session | — | suppressed (an idle stretch, not an outage) | `local_health` |
| `log_error_pattern` (CUDA OOM / NCCL / segfault) | high | `alert(high)` | `local_health` |
| `log_error_pattern` (RuntimeError / generic) | medium | `alert(medium)` | `local_health` |
| `gpu_thermal_high` (≥ warn_c / ≥ crit_c) | medium / high | `alert(medium)` / `alert(high)` | `local_health` |
| `disk_pressure` (≥ warn_pct / ≥ crit_pct) | medium / high | `alert(medium)` / `alert(high)` | `local_health` |
| `shm_pressure` (≥ warn_pct / ≥ crit_pct) | medium / high | `alert(medium)` / `alert(high)` | `local_health` |
| `fd_pressure` (≥ warn_pct / ≥ crit_pct) | medium / high | `alert(medium)` / `alert(high)` | `local_health` |
| `ray_head_dead` | high | `alert(high)` | `local_health` |
| `gpu_memory_leaked` | high | `alert(high)` + `delegate(recover, force_gpu_cleanup=True)` | `gpu_leak` |
| `deadline_warning` | medium / high | `alert(...)` | `budget` |
| `deadline_imminent` | high | `alert(high)` | `budget` |
| `deadline_hard_cutoff` | high | `alert(high)` | `budget` |
| `budget_burn_no_gain` | medium | `alert(medium)` | `budget` |
| `budget_strategy_drift` | medium | `alert(medium)` | `budget` |
| `phase_budget_nearly_exhausted` | medium | `alert(medium)` | `phase_budget` |
| `conversation_no_progress` | high | `alert(high)` | `conversation_progress` |
| `aiter_jit_regressed` | high | `alert(high)` | `aiter_jit` |
| `aiter_jit_build_stuck` | medium | `alert(medium)` | `aiter_jit` |
| `gain_plateau` | medium | `alert(medium)` | `progress` |
| `no_levers_found` | medium | `alert(medium)` | `progress` |
| `same_payload_loop` | high | `alert(high)` + `prune_branch(family)` | `repeated_payload` |
| `empty_patch_kept` | high | `alert(high)` | `decision_audit` |
| `decision_threshold_violated` | medium | `alert(medium)` | `decision_audit` |
| `kernel_dispatch_bypassed` | high | `alert(high)` | `decision_audit` |
| `kernel_negative_delta_kept` | high | `alert(high)` | `decision_audit` |
| `ci_metrics_baseline_zero` | high | `alert(high)` | `decision_audit` |
| `ci_metrics_schema_drift` | medium | `alert(medium)` | `decision_audit` |
| `model_gpu_infeasible` | high | `alert(high)` | `model_gpu_fit` |
| `amdahl_kernel_ceiling_low` | high | `alert(high)` + `prune_branch(kernel_opt)` | `amdahl_ceiling` |
| `cold_start_budget_exhausted` | high | `alert(high)` | `cold_start` |
| `critic_kb_outage` | high | `alert(high)` | `critic_health` |
| `critic_unavailable_streak` | high | `alert(high)` | `critic_health` |
| `critic_prune_stuck` | medium | `alert(medium)` | `critic_health` |
| `critic_runtime_stuck` | high | `alert(high)` | `critic_health` |
| `ray_pending_starvation` | high | `alert(high)` | `ray_pending` |
| `geak_budget_starvation` | high | `alert(high)` + `prune_branch(kernel_opt)` | `kernel_pipeline` |
| `kernel_opt_no_progress` | high | `alert(high)` + `prune_branch(kernel_opt)` | `kernel_pipeline` |
| `state_json_corrupt` | high | `alert(high)` | `state_integrity` |
| `coordinator_wal_bloat` (≥ warn / ≥ critical bytes) | medium / high | `alert(medium)` / `alert(high)` | `state_integrity` |
| `inbox_bloat` (≥ warn / ≥ critical bytes) | low / medium | `send_message(observation)` / `alert(medium)` | `state_integrity` |
| `coordinator_zombie` | high | `alert(high)` | `state_integrity` |
| `gateway_auth_outage` | high | `alert(high)` | `external_deps` |
| `wekafs_degraded` (unreachable, or ≥ warn / ≥ critical latency) | medium / high | `alert(medium)` / `alert(high)` | `external_deps` |
| `tracelens_cli_missing` | high (once per session) | `alert(high)` | `external_deps` |
| (no symptoms) | — | `send_message(heartbeat)` | — |

Every other HIGH symptom is strategic: the recommendation rides the
alert's `detail.suggestion` field and the ladder never auto-emits
`escalate_strategy_change` — the intent stays PolicyGate-allowed for
explicit drives, but Orchestration owns the phase-advance decision.
`delegate` is constrained to the `ROBUSTNESS_DELEGATE_ACTIONS`
allowlist (`recover` only); all other policing intents ride alerts.

Cooldown: identical `(symptom_name, subject)` keys are silenced for
`config.cooldown_ticks` ticks (default 5) to avoid inbox flooding.

## Data sources

Two of the rule families need no source at all: everything driven by the
rendered Coordinator prompt (`budget`, `phase_budget`,
`conversation_progress`, `progress`, `crash`) and everything driven by the
inbox (`event`, `repeated_payload`, and part of `stall` / `critic_health`).
Those keep working when the probe is off, which is what multi-node relies on.

The rest read `SourceData`, collected by:

* **LocalProbe** — the only collector
  * `coordinator.db` (read-only) for Coordinator events
  * `shutil.disk_usage`, `ps`, parsed `rocm-smi --csv` / `nvidia-smi`
  * `Config.health_probe_targets[]` — local HTTP `/health` probes for
    inference servers running on the same host
  * tail of a configured log file + error-pattern extraction
    (`CUDA out of memory`, `NCCL error`, `Segmentation fault`, etc.)
  * `state.json` / WAL / lease integrity, decision-audit and critic-workdir
    scans, and the external-dependency probes (gateway `/models`, source
    mounts, TraceLens CLI)
* **Quiet stub** — substituted when `Config.disable_local_probe` is set
  (the multi-node default). Returns an empty snapshot without raising, so
  probe-derived rules stay silent instead of false-firing on a single pod.

LocalProbe stays small-scope by design: it only collects what the agent
itself can see, so on multi-node it sees one pod and is therefore disabled.

`DegradeRouter` keeps the collector behind a silent fallback: after
`source_fail_threshold` (default 3) consecutive failures it serves an empty
snapshot instead, and re-probes every `source_recheck_interval_s` (default
30s). A tick therefore degrades to "no data" rather than failing outright.
State transitions emit one WARN log; no spam in steady state.

## LLM RCA

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

These records are the hand-off point for a future findings publisher;
today they remain local-only.

## Session-end postmortem

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

## Critic feedback loop

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
* **Findings publisher** — POST the on-disk findings to a collector for
  cross-session reporting and advisory pull-back. Nothing ships today;
  findings stay local-only.
