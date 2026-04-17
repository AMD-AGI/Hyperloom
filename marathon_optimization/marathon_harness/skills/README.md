# Inference Optimization Skill

Autonomous DFS-guided inference optimization for LLM serving on AMD Instinct GPUs.
The agent profiles, identifies bottlenecks, explores optimization actions via
depth-first search, and verifies improvements — all autonomously.

**Start here:** Read [`SKILL.md`](SKILL.md) — it contains the complete protocol.

## File Layout

| File | Description |
|------|-------------|
| `SKILL.md` | **The protocol** — DFS orchestrator, heuristic scoring, constraints |
| `actions/*.md` | Self-contained action modules (12 actions) dispatched by the DFS loop |
| `kernel-opt/*.md` | Per-backend kernel optimization references (GEAK, Codex, Claude, LLM) |
| `kb/` | Knowledge base — JSONL store + query/ingest scripts |
| `scripts/common.sh` | Shared functions — kill_server, wait_for_health, filter_trace, check_gpu_memory |
| `scripts/run_baseline.sh` | Launch server + baseline benchmark + profiling (single script) |
| `scripts/run_sweep.sh` | Parameter sweep (CONC/ISL/OSL grid with server reuse) |
| `scripts/eval_accuracy.sh` | GSM8K accuracy evaluation for the accuracy gate |
| `scripts/patch_inductor.py` | AST-safe kernel patching for Inductor standalone files |
| `modes/LOCAL.md` | Local mode (Cursor IDE, direct shell) execution details |
| `modes/CLAW.md` | Claw mode (SaFE RayJob) execution details |
| `KNOWLEDGE-BASE.md` | Legacy KB (archived, seeded into `kb/entries.jsonl`) |

## Quick Start

```
@inference-optimization Optimize <model> inference on MI355X.
Model: /shared_nfs/models/<model>
InferenceX: /shared_nfs/InferenceX
Target: compare with reference GPU at <X> tok/s/GPU
```

The agent will execute the full DFS protocol from SKILL.md:
setup → classify → target analysis → KB warm-up → baseline → profile →
build action stack → DFS loop (optimize) → sweep → report.

## Prerequisites

- **GPU**: AMD Instinct MI355X / MI325X / MI300X (ROCm 7.0+)
- **Framework**: SGLang v0.5.8+ or vLLM v0.17+
- **InferenceX**: Cloned for benchmarking
- **Model**: Downloaded to local/NFS path

## How It Differs from training-optimization

| Aspect | training-optimization | inference-optimization |
|--------|----------------------|----------------------|
| Scenario | Training | Inference serving |
| Benchmark | `torchrun` (training) | InferenceX `benchmark_serving` |
| Primary lever | Config/code changes | Backend switches + kernel optimization |
| After optimization | Report | Parameter sweep → Pareto curves → report |
