---
myst:
    html_meta:
        "description": "Multi-branch prompt-cache TTL variables for Hyperloom: Claude CLI and subagent cache TTLs, defaulting to 1h on InferenceX and the CLI default on AgentX workloads."
        "keywords": "Hyperloom, environment variables, prompt cache, CLAUDE_CODE_PROMPT_CACHE_TTL, CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL, multi-branch, TTL"
---
# Hyperloom Multi-branch prompt-cache TTL

A multi-branch session sets the prompt-cache TTL of every branch's Claude CLI. On an InferenceX workload a branch
waits for its test in the same conversation for longer than the CLI's default 5-minute TTL, so the session
defaults to `1h` there; on an AgentX workload, whose test outlasts even an hour, it keeps the CLI's default. A
variable set in the session's environment wins on either workload, and reaches a single-lineage specialist too.

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_CODE_PROMPT_CACHE_TTL` | `1h` on InferenceX; unset (the CLI's `5m`) on AgentX | The TTL of a branch CLI's own prompt cache, `5m` or `1h`. |
| `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL` | As `CLAUDE_CODE_PROMPT_CACHE_TTL` | The TTL of its subagents' prompt cache. |
