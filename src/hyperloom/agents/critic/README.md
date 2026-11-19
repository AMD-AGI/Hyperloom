# Critic runtime

The Critic does not need a dedicated long-running service. Each turn,
Hyperloom's `CriticAgentBackend` writes the Coordinator inbox prompt to
`request.json`, runs `prepare-review` to build `judge_bundle.json`,
reasons over the bundle with the Critic SKILL prompts, writes
`review.json`, runs `commit-review`, and forwards
`emit.json["intent_envelope"]` to the Coordinator.

## Required environment

| Variable | Purpose |
|---|---|
| `WORKSPACE_PATH` | Skill-asset root the runtime resolves prompts against (NOT an artefact root). Hyperloom sets this to `$REPO_ROOT` automatically; defaults to `/workspace` for standalone use. |
| `CRITIC_SESSION_MEMORY_DIR` | Persistent volume for per-session memory (default `/var/lib/critic-session-memory`). Mount as PVC. |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Logical success. |
| `2` | Adapter bug; surface to the host so the SKILL can produce `needs_review`. |

The CLI emits a single JSON object on stdout (or to `--out`) for every
command, so the host can parse the response without regex.

## Optional: local Cursor / IDE workflow

When developing locally without a Hyperloom session, you can drive the
same flow by hand:

```bash
echo '{"session_id": "local", "raw_prompt": "..."}' > req.json
python -m hyperloom.agents.critic.runtime.cli prepare-review --request req.json --out judge.json
# craft review.json based on judge.json + skill prompts
python -m hyperloom.agents.critic.runtime.cli commit-review --request req.json --review review.json --out emit.json
```

Tests for the runtime live under `runtime/tests/`.
