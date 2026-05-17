# Hosting the Critic skill in an A2A / Codex chat server

The Critic does not need a dedicated long-running service. The intended
delivery is:

- A Codex-based **A2A chat server** owns the agent identity (allowlists,
  rate limits, auth).
- The Critic SKILL is mounted into that server.
- The server permits exactly two Bash commands for Critic turns:
  `python -m runtime.cli prepare-review` and `python -m runtime.cli commit-review`.
- For session lifecycle: `init-session` and `close-session`.

## Per-turn flow

1. The host receives a request payload (Coordinator-style or
   dialogue-style) and writes it to `${CRITIC_WORKDIR}/request.json`.
2. Codex runs `prepare-review`, reads `judge_bundle.json` and the
   Critic SKILL prompts, then writes `review.json`.
3. Codex runs `commit-review` and uses `emit.json` as the response back
   to the host.

The host should consider `emit.json["intent_envelope"]` (when present)
the canonical response for a Coordinator caller; for dialogue callers
return `emit.json["critic_decision_review"]` instead.

## Required environment

| Variable | Purpose |
|---|---|
| `WORKSPACE_PATH` | Skill-asset root the runtime resolves prompts against (NOT an artefact root). Hyperloom sets this to `$REPO_ROOT` automatically; defaults to `/workspace` for standalone use. |
| `CRITIC_SESSION_MEMORY_DIR` | Persistent volume for per-session memory (default `/var/lib/critic-session-memory`). Mount as PVC. |
| `CRITIC_KB_CLIENT_MODE` | `live` to talk to the KB service, `inmemory` for dry-run / local tests. |
| `KB_BASE_URL` | KB service URL when `CRITIC_KB_CLIENT_MODE=live`. |
| `KB_TIMEOUT_MS` | HTTP timeout for KB calls (default 10000). |
| `KB_RETRY_MAX` | KB retry budget (default 3). |
| `KB_DEAD_LETTER_DIR` | PVC mount for dead-letter JSONL files. |
| `KB_WRITE_ENABLED` | Set to `false` to disable KB writes globally. |
| `KB_READ_ENABLED` | Set to `false` to disable KB priors lookup. |
| `KB_SERVICE_TOKEN` | Reserved for v2 auth. |
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
   - `write_verdict`, `write_kb_drafts`, and `add_contradiction` return
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

## Bash allowlist

Configure the chat server's Bash gate to allow exactly:

```text
python3 -m runtime.cli init-session ...
python3 -m runtime.cli prepare-review ...
python3 -m runtime.cli commit-review ...
python3 -m runtime.cli close-session ...
python3 -m runtime.cli list-priors ...
python3 -m runtime.cli replay-dead-letter ...
```

Reject `write-verdict`, `write-kb-drafts`, `add-contradiction` outside
trusted operators — they are exposed for tooling, not for direct LLM use.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Logical success (including `dead_lettered` write outcomes). |
| `2` | Adapter bug; surface to the host so the SKILL can produce `needs_review`. |

The CLI emits a single JSON object on stdout (or to `--out`) for every
command, so the host can parse the response without regex.

## Recommended dead-letter cron

```bash
*/15 * * * *  python -m runtime.cli replay-dead-letter \
                --dir "$KB_DEAD_LETTER_DIR" \
                --keep-on-success
```

## Optional: local Cursor / IDE workflow

When developing locally without the chat server, you can drive the same
flow by hand:

```bash
echo '{"kind": "coordinator_inbox", ...}' > req.json
python -m runtime.cli prepare-review --request req.json --out judge.json
# craft review.json based on judge.json + skill prompts
python -m runtime.cli commit-review --request req.json --review review.json --out emit.json
```

Tests for the runtime live under `runtime/tests/`; the SKILL fixtures
under `tests/`.
