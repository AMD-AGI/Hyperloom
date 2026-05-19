# framework-agent

vllm/sglang source-layer optimisation companion for
[`inference_optimizer`](../inference_optimizer/). Two protocols share
this package:

1. **PR exploration** (`fa candidates` / `fa explore`) — discovers and
   imports upstream PRs for ad-hoc A/B benchmark. Used by IO's
   `framework_pr` bandit arm.
2. **5th-role sibling skill** (`fa agent`, PR-D+) — AST scan + patch
   propose + apply lifecycle invoked by IO's Framework agent role via
   subprocess bridge.

See [`SKILL.md`](./SKILL.md) for the full architectural overview.

## Quick start

```bash
# Install (idempotent; PR-D adds scripts/install.sh)
cd Hyperloom/framework-agent
pip install -e '.[test]'

# Legacy PR exploration
fa schema
fa candidates --request /path/to/request.json
fa explore --request /path/to/request.json [--execute]

# KB management
fa kb list
fa kb search --domain framework_optimization --query "fp8 kv cache"

# Sibling-skill subprocess (PR-D+)
fa agent prepare-task --task task.json --output-bundle bundle.json
fa agent commit-result --envelope envelope.json --task-id <task_id>
```

## Tests

```bash
pytest -q                      # all 160+ unit tests
pytest -q tests/test_logging_setup.py tests/test_isolation.py \
          tests/test_decision.py tests/test_explore_modes.py
```

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

- [`framework-explorer-merged-design.md`](../claw-dev/docs-zh/framework-explorer-merged-design.md)
  — PR exploration tool (existing).
- [`hyperloom-framework-agent-design.md`](../claw-dev/docs-zh/hyperloom-framework-agent-design.md)
  — 5th-role design (v1.3).
- [`hyperloom-framework-agent-implementation-plan.md`](../claw-dev/docs-zh/hyperloom-framework-agent-implementation-plan.md)
  — PR-A/B/C/D/E/F/G/H/I execution plan.
