---
myst:
    html_meta:
        "description": "TraceLens analysis budget variables for Hyperloom: stream idle timeout and tool-call idle timeout that bound inactivity in the Claude-backed TraceLens analysis session."
        "keywords": "Hyperloom, environment variables, TraceLens budgets, HYPERLOOM_TRACELENS_STREAM_IDLE_TIMEOUT_SEC, HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC, idle timeout"
---
# Hyperloom TraceLens analysis budgets

On the Claude backend, the TraceLens analysis session bounds each wait for the
next SDK message: the in-process SDK has no read timeout, so a stalled gateway
would otherwise hold the session. Both limits measure inactivity, not total
time. The Codex runner is one-shot and exposes no stream to bound.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_TRACELENS_STREAM_IDLE_TIMEOUT_SEC` | `300` | Seconds to wait for the next SDK message while no tool call is running. Values below `30` are raised to `30`; `0` or a negative value removes the bound. |
| `HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC` | `3600` | The same bound while a tool call is running, when the SDK is silent by design: a TraceLens report over a large trace can run for many minutes. When unset it is never tighter than the stream bound. Values below `30` are raised to `30`; `0` or a negative value removes the bound while a tool runs. |
