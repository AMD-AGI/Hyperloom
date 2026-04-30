---
name: inference-optimizer
description: |
  Single-mode autonomous inference optimization for LLM serving on AMD GPUs.
  A Python Conductor coordinates 4 persistent LLM agents — Orchestration / Kernel
  (layer experts) plus Critic / Robustness (cross-layer) — over a shared NFS
  session directory, driving end-to-end inference optimization (kernel selection,
  GEAK / OOB submission, integration, accuracy gating, recovery) inside a single
  GPU sandbox. State is persisted to a single SQLite WAL database (events /
  cursors / leases / tasks); resource contention is mediated by 4 mutual-exclusion
  lanes (server_lifecycle / workspace_mutation / benchmark_lane / profile_lane).
globs:
  - "**/inference*optim*"
  - "**/inference-optimizer*"
---

# Inference Optimizer v0.6 — Single-Mode 4-Agent Autonomous Skill

> **Specification**: `inference-optimizer-DESIGN-v2.md` (v0.6 Final).
> **Code layout**: this directory is the canonical home of both the Python
> Conductor runtime and the agent assets. The Python entry point is
> `from inference_optimizer.orchestrator import …` (the directory name has
> a hyphen for skill convention, the module name has an underscore for
> Python — `pyproject.toml` maps between them).

## Roles (4 persistent agents, no mode gating)

| Agent | Model | Role |
|---|---|---|
| **Orchestration** | Claude `claude-opus-4-7` | Layer expert — proposes actions, delegates sub-agents, REQUESTs Kernel |
| **Kernel** | Claude `claude-opus-4-7` | Layer expert — owns 5 deep-kernel actions, responder-only via REQUEST/RESPONSE |
| **Critic** | Codex `gpt-5.4` (no-tools + KB exception) | Cross-layer — reviews proposals (approve/reject/redirect/advise), owns KB |
| **Robustness** | Claude `claude-opus-4-7` | Cross-layer — always-on Watchdog + RCA + Handle, scheduling police |

`Framework` and `Comm` layer experts are placeholders (DESIGN §7.7); not implemented in v0.6.

## Directory layout

```
inference-optimizer/
├── SKILL.md                         ← this file (skill entry point)
├── README.md                        ← human-facing overview
├── orchestrator/                    ← Conductor + agent_role + policy + scheduler ...
│   ├── conductor.py
│   ├── agent_role.py
│   ├── policy.py
│   ├── intent_parser.py
│   ├── message_bus.py
│   ├── resource_lock.py
│   ├── task_registry.py
│   ├── cursor_store.py
│   └── system_prompts/
│       ├── orchestration.md
│       ├── kernel.md
│       ├── critic.md
│       └── robustness.md
├── agents/                          ← per-agent assets (agent_card / SKILL / scripts)
│   ├── orchestration/
│   ├── kernel/
│   ├── critic/
│   └── robustness/
├── storage/                         ← SQLite WAL connection + schema
│   ├── connection.py
│   └── schema.py
├── actions/                         ← OptimizationAction catalogue + _meta/*.yaml
├── kernel_opt/                      ← per-backend GEAK / OOB prompt templates
├── scripts/                         ← shell tools (run_baseline.sh / patch_inductor.py / ...)
└── tests/                           ← pytest (unit + integration + e2e)
```

## Implementation status

P0 milestone (current): Conductor + Orchestration + Kernel main path. Critic and
Robustness ship as mock adapters (non-blocking). See repo-root TODO list and the
`feature/xiaofei/kernel-agent` branch.
