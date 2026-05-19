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

## Used by inference_optimizer as a pre-stage

`inference_optimizer` can drive `fa explore` as a one-shot
*before-baseline* step via `--framework-pr-discover` (or apply an
explicit ref via `--framework-pr PR:N`). The IO side handles the
hand-off (resolve PR `head_sha`, `git checkout` sglang, `pip install
-e python/`); framework-agent itself stays standalone and only
produces the winner record:

```bash
# Auto-discover via fa (Primus Cortex + GitHub) and apply before baseline
inference_optimizer optimize \
    --model "$MODEL_PATH" \
    --framework sglang \
    --framework-pr-discover \
    --framework-gap "improve sglang fp8 MoE on MI300X" \
    --max-hours 2

# Or explicit PR ref - skips fa, jumps straight to git checkout + pip install
inference_optimizer optimize \
    --model "$MODEL_PATH" \
    --framework sglang \
    --framework-pr PR:25748 \
    --max-hours 2
```

See `inference_optimizer/SKILL.md` "Optional: Framework-Agent
Pre-stage" for the full IO-side contract and
`inference_optimizer/orchestrator/framework_pr_discover.py` for the
implementation.

## Design references

- `claw-dev/docs-zh/framework-explorer-merged-design.md` - upstream
  fusion design (Arbor + zhenggong).
- `claw-dev/docs-zh/framework-agent-hyperloom-implementation-plan.md`
  - this skill's implementation plan and PR breakdown.
