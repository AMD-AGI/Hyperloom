# Critic Backend Selection

The Critic role has two backend modes. Default is `--critic-agent` (no flag
needed).

| Flag | Backend class | Behaviour |
|---|---|---|
| (none) / `--critic-agent` | `CriticAgentBackend` | Drives the standalone `critic-agent/` skill runtime via `python -m runtime.cli prepare-review` → Codex chat completion → `python -m runtime.cli commit-review`. Adds Recipe/Cortex KB context when configured, per-session memory + idempotent `reviewed_msg_ids` (no double-verdict), `judge_bundle.review_constraints` injected into the LLM prompt, and `needs_review` / `critic_unavailable` source when context is missing. |
| `--critic-mock` | `MockCriticBackend` | Always-approve adapter. Use for offline / smoke tests when Codex creds aren't available. |

Default is overridable per pod via `INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND`
(one of `mock` / `agent`).

## Required env when `--critic-agent` is active

| Var | Purpose | Default |
|---|---|---|
| `CRITIC_AGENT_ROOT` | Path to the directory containing `runtime/cli.py`. | sibling `$REPO_ROOT/critic-agent/` |
| `CRITIC_KB_CLIENT_MODE` | Critic runtime prior-store mode. Hyperloom sets this to `inmemory` by default; use `live` only when deliberately wiring the standalone critic KB client. | `inmemory` |
| `KB_BASE_URL` | Required only when `CRITIC_KB_CLIENT_MODE=live`; unused in the default Hyperloom path. | unset |
| `CORTEX_KB_URL` | Optional remote Cortex KB URL for best-effort per-proposal `/v2/reasoning/assess` enrichment. Usually injected from `--cortex-kb-url`; unset skips this enrichment. | unset |
| `KB_TIMEOUT_MS` / `KB_RETRY_MAX` / `KB_DEAD_LETTER_DIR` | Optional runtime tuning for the critic KB client and dead-letter handling. | runtime defaults |
| `CRITIC_SESSION_MEMORY_DIR` | Where the runtime persists per-session decisions / reviewed_msg_ids. | `$SESSION_DIR/critic-session-memory` (auto-set by the optimizer; co-located with the Coordinator session and cleaned up alongside it). |
| `WORKSPACE_PATH` | Skill root the critic-agent runtime resolves prompt assets against. | `$REPO_ROOT` (auto-set). |

`_preflight()` checks `CRITIC_AGENT_ROOT` resolves to a real directory with
`runtime/cli.py`, then runs `python -m runtime.cli --help` (5s timeout) before
the Coordinator boots. Missing or broken runtime aborts the run with a clear
error pointing at `--critic-mock` as the offline bypass.

## Per-turn artefacts (audit trail)

Each Critic turn writes a 6-digit workdir under
`$SESSION_DIR/critic-workdir/<turn_idx>/` (`request.json` / `judge_bundle.json`
/ `review.json` / `emit.json`) plus session memory under
`$SESSION_DIR/critic-session-memory/<session_id>/`. The backend prunes to the
latest 50 turn workdirs each tick. Inspect these when debugging critic verdicts
(see `troubleshooting.md`).
