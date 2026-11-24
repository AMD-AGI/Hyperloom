---
myst:
    html_meta:
        "description": "Reactor budget variables for Hyperloom: total wall-clock limit per reactor stage, Claude Agent SDK turn timeout for orchestration, and DeepSeek Harness turn timeout."
        "keywords": "Hyperloom, environment variables, reactor budgets, INFERENCE_OPTIMIZER_REACTOR_TURN_TIMEOUT_SEC, INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC, INFERENCE_OPTIMIZER_DEEPSEEK_CALL_TIMEOUT_SEC"
---
# Reactor budgets

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_OPTIMIZER_REACTOR_TURN_TIMEOUT_SEC` | `1800` | Total wall-clock limit for each reactor stage, including backend startup, streamed output, retries, backoff, and cleanup. This is independent of backend `*_CALL_TIMEOUT_SEC` settings: for streamed Claude turns those settings bound idle time between SDK messages, and activity resets that idle timer. Reaching this total limit cancels the stage and records a crash; a shorter remaining session bound still ends the stage without recording a crash. A malformed value raises `EnvValueError`; a non-positive or non-finite value falls back to the default. |
| `INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC` | `1500` | Total wall-clock limit for one Claude Agent SDK call made by the orchestration backend (reactor turns, including the SWEEP cycle-directive handoff), covering every retry and the backoff between them; tearing the CLI down afterwards adds at most 30 s. Unlike the idle bound of `INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC`, the CLI's own API-retry messages do not extend it. Past it the call is cancelled, the CLI and every process it started are killed, and the turn fails as a backend error the loop retries on its next tick. |
| `INFERENCE_OPTIMIZER_DEEPSEEK_CALL_TIMEOUT_SEC` | `300` | Wall-clock limit for one DeepSeek Harness turn made by the orchestration backend. Past it the harness runtime is closed and the next turn starts a new runtime and a new conversation. |
