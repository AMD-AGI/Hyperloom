---
myst:
    html_meta:
        "description": "Phase tuning variables for Hyperloom: macro-cycle duration, backend error streak threshold, minimum remaining time for a new cycle, kernel idle ticks and seconds, and SWEEP variant timeout."
        "keywords": "Hyperloom, environment variables, phase tuning, INFERENCE_OPTIMIZER_CYCLE_HOURS, INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS, SWEEP_VARIANT_TIMEOUT_SEC, macro-cycle"
---
# Phase tuning

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_CYCLE_HOURS` | Optional | `24.0` | Duration of one macro-cycle in hours. Multiplied by 60 to obtain `cycle_minutes` on the first session start. A malformed value raises `EnvValueError`. |
| `INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD` | Optional | `5` | Consecutive backend errors per agent role before a `backend_unhealthy` event is recorded. A malformed value raises `EnvValueError`. |
| `INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC` | Optional | `10800` | Absolute minimum remaining session seconds to justify opening a new macro-cycle. For bounded sessions the effective floor is `min(this, max_minutes * 60 * 0.15)` so shorter sessions are not unconditionally blocked. |
| `INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS` | Optional | `3` | Consecutive no-work KERNEL_AGENT ticks before the phase winds down to SWEEP. Values below `1` are treated as `3`. A malformed value raises `EnvValueError`. |
| `INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS` | Optional | `600` | Wall-clock seconds a KERNEL idle streak must last before winding down to SWEEP. Values at or below `0` are treated as `600`. A malformed value raises `EnvValueError`. |
| `SWEEP_VARIANT_TIMEOUT_SEC` | Optional | `2400` | Per-variant wall-clock timeout in seconds for GEAK-backed SWEEP runs. A malformed value raises `EnvValueError`. |
