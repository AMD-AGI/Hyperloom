---
myst:
    html_meta:
        "description": "Benchmark deadline and lifecycle variables for Hyperloom: hard wall-clock timeout per benchmark spawn, output-silence timeout after server readiness, and session lifecycle notes."
        "keywords": "Hyperloom, environment variables, benchmark deadlines, INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC, INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC, session lifecycle"
---
# Hyperloom benchmark deadlines

Each actual benchmark subprocess spawn uses the same finite, positive limits,
including baseline, explore, sweep, and rebench measurements:

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` | `7800`; under `HYPERLOOM_AGENTIC_BACKEND=mlperf`, sized to the trajectory count, and under native AgentX sized to the round's `CONC` (see [MLPerf agentic backend](env-vars-agentx-performance.md#mlperf-agentic-backend)) | Hard wall-clock seconds per actual spawn, including server boot and accuracy evaluation. Output cannot extend it. An explicit value always wins. |
| `INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC` | `600` | Output-silence seconds after real server readiness. Not armed before readiness or for server-less scriptable workloads. |

Session `--max-hours` and cancellation apply independently of benchmark limits.
Task and lease age do not expire work; admission and phase budgets, ordinary
release, and dead-PID cleanup remain. Session exhaustion is cooperative, not a
guarantee that a frozen Coordinator will terminate. There is no out-of-band
supervision or automatic resume. SIGHUP uses the normal terminal drain; an
operator may later explicitly use `--resume-from` on the same session.
Historical stop reasons remain interpretable and are not restart requests.

`PYTHONUNBUFFERED=1` affects Python output only, not shell/native buffering or
upstream subprocess capture. Active logs do not prove useful progress.

Profile, KernelForge, GEAK, LLM-call, and SGLang's own watchdog budgets are separate
from these benchmark limits and retain their existing contracts.
