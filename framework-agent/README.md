# Framework Agent

Standalone PR/ref exploration agent for Hyperloom serving frameworks
(vLLM / SGLang).

## Architecture

- **Sibling skill** under `Hyperloom/framework-agent/` - does NOT
  integrate with inference_optimizer's 5-role mesh.
- **Dual data source** for PR discovery: Primus Cortex (AMD internal
  REST) + GitHub Search (public fallback).
- **Isolated execution**: git worktree + venv per candidate.
- **KB sediment**: contributes measured findings to a 4-file KB
  partition.

See `SKILL.md` for the operation protocol and `AGENTS.md` for hosting
modes.

## Quick Start

```bash
cd framework-agent
bash scripts/install.sh
. runtime/env.sh
fa explore --request examples/explore_request.json --out /tmp/fe/summary.json
```

Without `--execute` the command writes a plan + audit material. With
`--execute` it runs the trusted command templates from the request.

## CLI subcommands

```text
fa explore --request <path> [--out <path>] [--execute]
fa candidates --request <path> [--out <path>]
fa schema
fa kb {list|show|search|synthesize}
```

PR 1 ships only `fa schema`; other subcommands are added in subsequent
PRs (see implementation plan).

## Comparison with sibling agents

| Sibling | Protocol layer | Manual promotion | Subprocess of |
|---|:---:|:---:|---|
| kernel-agent | yes (kernel-owned actions) | no (auto integrate) | inference_optimizer |
| critic-agent | yes (review verdicts) | no (verdict drives) | inference_optimizer |
| robustness-agent | yes (KILL_TASK etc.) | no (auto recover) | inference_optimizer |
| **framework-agent** | **no** | **yes** | **standalone (any LLM tool / CLI)** |

## Design references

- `claw-dev/docs-zh/framework-explorer-merged-design.md` - upstream
  fusion design (Arbor + zhenggong).
- `claw-dev/docs-zh/framework-agent-hyperloom-implementation-plan.md`
  - this skill's implementation plan and PR breakdown.
