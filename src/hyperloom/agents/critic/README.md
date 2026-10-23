# Critic runtime

The Critic does not need a dedicated long-running service. Each turn,
Hyperloom's `CriticAgentBackend` writes a `coordinator_inbox` request to
`request.json`, runs `prepare-review` to build `judge_bundle.json`,
reasons over the bundle with the Critic SKILL prompts, writes
`review.json`, runs `commit-review`, and forwards
`emit.json["intent_envelope"]` to the Coordinator.

## Required environment

| Variable | Purpose |
|---|---|
| `WORKSPACE_PATH` | Skill-asset root the runtime resolves prompts against (NOT an artefact root). Hyperloom sets this to `$REPO_ROOT` automatically; defaults to `/workspace` for standalone use. |
| `CRITIC_SESSION_MEMORY_DIR` | Persistent volume for per-session memory (default `/var/lib/critic-session-memory`). Mount as PVC. |
| `CRITIC_KB_CLIENT_MODE` | Critic runtime prior-store mode. `inmemory` is the default used by Hyperloom; set `live` only when deliberately wiring the standalone critic KB client. |
| `KB_BASE_URL` | Required only when `CRITIC_KB_CLIENT_MODE=live`; unused by the default `inmemory` path. |
| `KB_TIMEOUT_MS` | HTTP timeout for the standalone live KB client (default 10000). |
| `KB_RETRY_MAX` | KB retry budget (default 3). |
| `KB_DEAD_LETTER_DIR` | PVC mount for dead-letter JSONL files. |
| `KB_WRITE_ENABLED` | Set to `false` to disable KB writes globally. |
| `KB_READ_ENABLED` | Set to `false` to disable KB priors lookup. |
| `KB_SERVICE_TOKEN` | Bearer credential for live KB requests (sent as `Authorization: Bearer <token>`); omitted when unset, unused by the default `inmemory` path. |
| `CRITIC_PRIOR_CACHE_TTL_SECONDS` | Per-session KB prior cache TTL. |
| `CRITIC_KB_BREAKER_THRESHOLD` | Consecutive transport errors before the circuit breaker opens (default `1` — first failure short-circuits the rest of the request). |
| `CRITIC_KB_BREAKER_COOLDOWN_SECONDS` | How long the breaker stays open before allowing another KB attempt (default `60`). |

## KB unreachable behaviour (default)

The runtime treats KB unreachability as "skip KB, keep reviewing" by
default — you do not need to flip a flag when the KB service is down:

1. The first KB transport / network / 5xx-after-retries failure trips an
   in-process circuit breaker per `KBWriter` instance.
2. While the breaker is open (`CRITIC_KB_BREAKER_COOLDOWN_SECONDS`):
   - `list_priors` returns `cache="kb_unreachable"` with empty priors and
     never makes another transport call.
   - `write_verdict` returns
     `WriteResult(status="disabled", reason="kb_unreachable")` so the
     review pipeline keeps emitting verdicts.
   - `prepare-review` reports `judge_bundle.kb_read_skipped_reason =
     "kb_unreachable"` and adds a note so the SKILL knows priors are
     missing because of an outage rather than a clean miss.
3. A successful KB call resets the breaker (`_record_kb_success`).
4. 4xx errors (`KBValidationError`) do **not** open the breaker — they
   indicate a client bug, not service unavailability, and are surfaced
   as `error` on the response.

`KB_READ_ENABLED=false` is still honoured for operator-driven kill
switches; the breaker is the automatic equivalent for transient
outages.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Logical success (including `dead_lettered` write outcomes). |
| `2` | Adapter bug; surface to the host so the SKILL can produce `needs_review`. |

The CLI emits a single JSON object on stdout (or to `--out`) for every
command, so the host can parse the response without regex.

## Recommended dead-letter cron

`replay-dead-letter` is an operator command, not for LLM use.

```bash
*/15 * * * *  python -m hyperloom.agents.critic.runtime.cli replay-dead-letter \
                --dir "$KB_DEAD_LETTER_DIR" \
                --keep-on-success
```

## Optional: local Cursor / IDE workflow

When developing locally without a Hyperloom session, you can drive the
same flow by hand:

```bash
echo '{"kind": "coordinator_inbox", ...}' > req.json
python -m hyperloom.agents.critic.runtime.cli prepare-review --request req.json --out judge.json
# craft review.json based on judge.json + skill prompts
python -m hyperloom.agents.critic.runtime.cli commit-review --request req.json --review review.json --out emit.json
```

Tests for the runtime live under `runtime/tests/`.
