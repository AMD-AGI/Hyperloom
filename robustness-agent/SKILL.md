---
name: robustness-agent
description: Independent guardian daemon for Hyperloom inference optimization. Provides continuous health monitoring (GPU/process/disk/server), event-driven anomaly detection, root cause analysis, and scheduling police capabilities (kill_task/force_dispatch/prune_branch/escalate_strategy_change).
---

# Robustness Agent

Independent Python daemon that monitors the health of an inference optimization
session. Designed to run alongside Conductor + Orchestration + Kernel agents,
providing L0-L2 observability and automated intervention.

## Quick Start

```bash
cd robustness-agent
pip install -e .

SESSION_DIR=/path/to/session \
  robustness-agent
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SESSION_DIR` | yes | — | Path to the session directory containing `storage/conductor.db` |
| `ROBUST_ANALYZER_URL` | no | — | Primus-Robust-Internal analyzer URL (e.g. `http://robust-analyzer:8085`). If unset, uses local shell commands for GPU metrics |
| `OPENAI_BASE_URL` | no | — | LLM endpoint for RCA diagnosis |
| `SAFE_API_KEY` | no | — | API key for LLM |
| `LLM_MODEL` | no | `claude-opus-4-7` | Model to use for RCA |

## Architecture

```
robustness-agent (independent daemon)
├── Monitors (async tasks, different intervals)
│   ├── ProcessMonitor    (10s) — sglang/vllm/benchmark process tracking
│   ├── GpuMonitor        (15s) — VRAM/temp/ECC/utilization via Provider
│   ├── ServerHealthMonitor(30s) — HTTP /health endpoint probing
│   ├── LogTailer         (5s)  — server.log error pattern matching
│   └── DiskCheck         (60s) — disk space monitoring
│
├── Checks (event-driven from Conductor SQLite)
│   ├── EventCheck   — error patterns, KEEP/REVERT bouncing, family failures
│   └── StallCheck   — agent heartbeat timeout detection
│
├── Providers (pluggable metrics backend)
│   ├── LocalProvider   — rocm-smi / nvidia-smi / ps / df
│   ├── RobustProvider  — Primus-Robust-Internal REST API
│   └── HybridProvider  — auto-fallback: Robust when available, Local otherwise
│
├── RCA Engine (LLM, invoked only on critical alert accumulation)
│   └── Structured evidence → LLM diagnosis → action recommendation
│
└── Conductor Integration
    ├── ConductorReader — poll SQLite events table (read-only)
    └── IntentEmitter   — write alerts and scheduling police intents
```

## Scheduling Police Intents (v0.6 §19)

| Intent | When |
|--------|------|
| `kill_task` | Task stuck or repeatedly failing |
| `force_dispatch` | High-value task blocked behind low-priority queue |
| `prune_branch` | 3+ failures in same action family |
| `escalate_strategy_change` | Systemic issue requiring Orchestration to change approach |

## Deployment Modes

- **Local mode** (`ROBUST_ANALYZER_URL` unset): All metrics collected via shell
  commands. GPU history limited to in-memory ring buffer (default 1h).
- **Cluster mode** (`ROBUST_ANALYZER_URL` set): GPU/RDMA/fault metrics from
  Primus-Robust-Internal (5s granularity, 30-day history). Process/disk/server
  health still collected locally.
