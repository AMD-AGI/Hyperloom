---
myst:
  html_meta:
    "description": "Hyperloom release notes"
    "keywords": "Hyperloom, release notes, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, Primus-Claw, Local Mode, kernel optimization"
---

# Hyperloom release notes

## Hyperloom 0.1.0

Initial release of Hyperloom.

### Added

This release introduces the following features:

- **Agentic optimization loop** — Hyperloom treats LLM inference optimization as
  a search problem. Given a workload, the agent autonomously explores candidates
  one change at a time — backend swaps, server parameters, GEMM tuning, kernel
  rewrites, parallelism configs — measuring against the real workload
  before accepting any change.

- **TraceLens integration** — Agentic trace analysis that captures bottlenecks
  and roofline targets from real workload traces, giving the optimizer a
  hardware-grounded picture of where performance is being left on the table.

- **GEAK kernel optimization** — GPU kernel generation and optimization using
  Triton, HIP, and FlyDSL. Hot kernels are optimized asynchronously in parallel
  with the main optimization loop, so kernel work doesn't block forward progress.

- **Session artifacts and `session_breakdown.json`** — Each run produces
  reproducible session artifacts and a machine-readable `session_breakdown.json`
  that records the final throughput, cumulative validated gain, and the ordered
  action path — designed for dashboard and downstream delivery integrations.

- **Primus-Claw hosted UI** — AMD-internal users and approved partners can run
  Hyperloom from a browser with no local GPU setup. Jobs run in isolated
  sandboxed containers; multi-node workloads fan out using RayJob. Every run feeds
  results back through a data flywheel that continuously improves the agent's KB
  and scoring heuristics.

- **Local Mode for self-hosted deployments** — External users can install
  Hyperloom on their own AMD GPU hardware and run the full optimization loop
  locally, with Cursor as the agent interface and the same phase structure as the
  hosted tier.
