---
myst:
  html_meta:
    "description": "Hyperloom release notes: headline capabilities for version 0.8.0, including the agentic optimization loop, multi-agent runtime, TraceLens integration, and session artifacts."
    "keywords": "Hyperloom, release notes, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, Primus-Claw, bare metal, kernel optimization"
---

# Hyperloom release notes

The current packaged version is **0.8.0** (`pyproject.toml`). For the
per-change history since the initial snapshot, see
[`CHANGELOG.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/CHANGELOG.md);
this page summarizes the headline capabilities.

## Hyperloom (initial capabilities)

The first public snapshot introduced the following features, which remain the
core of the current runtime:

### Added

- **Agentic optimization loop** — Hyperloom treats LLM inference optimization as
  a search problem. Given a workload, the agent autonomously explores candidates
  one change at a time — backend swaps, server parameters, GEMM tuning, kernel
  rewrites, parallelism configs — measuring against the real workload
  before accepting any change.

- **Multi-agent runtime** — A single-mode 4-agent architecture
  (Orchestration / Kernel / Critic / Robustness) drives the loop, with
  additional framework-agent (upstream framework PR discovery / authoring) and
  quantization-agent (AMD Quark PTQ prelude) roles available through the
  `--quantize` / `--framework`-driven paths.

- **TraceLens integration** — Agentic trace analysis that captures bottlenecks
  and roofline targets from real workload traces, giving the optimizer a
  hardware-grounded picture of where performance is being left on the table.

- **Kernel optimization** — Hot kernels are optimized asynchronously in parallel
  with the main loop, so kernel work doesn't block forward progress. The default
  backend is GEAK (Triton / HIP / FlyDSL).

- **Session artifacts and `session_breakdown.json`** — Each run produces
  reproducible session artifacts and a machine-readable `session_breakdown.json`
  file that records the final throughput, cumulative validated gain, and the ordered
  action path — designed for dashboard and downstream delivery integrations.

- **Bare-metal and Docker setup for self-hosted deployments** — Install
  Hyperloom on your own AMD GPU hardware and run the full optimization loop
  with an agent-driven setup flow.
