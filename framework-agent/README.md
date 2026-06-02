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

## Used by inference_optimizer as a bandit arm

`inference_optimizer` drives PR discovery via the `framework_pr` bandit
arm (not startup CLI flags). After `baseline` completes, the
Orchestration agent proposes `framework_pr`; the executor calls `fa
candidates`, applies the winning PR to sglang, runs a sub-baseline, and
KEEP/DISCARDs based on throughput gain.

Gap / keywords are auto-composed from SharedState (`framework`,
`gpu_type`, `model_class`, `precision`). Override per tick via
`proposal.params.gap_override` from the Orchestration prompt.

```bash
inference_optimizer optimize \
    --model "$MODEL_PATH" \
    --framework sglang \
    --model-class dense \
    --gpu-type mi300x \
    --no-kernel \
    --max-hours 2
```

See `inference_optimizer/SKILL.md` "Framework-Agent as Bandit Arm"
and `inference_optimizer/orchestrator/action_executors/framework_pr.py`.

## Design references

- [`framework-explorer-merged-design.md`](../claw-dev/docs-zh/framework-explorer-merged-design.md)
  — PR exploration tool (existing).
- [`hyperloom-framework-agent-design.md`](../claw-dev/docs-zh/hyperloom-framework-agent-design.md)
  — 5th-role design (v1.3).
- [`hyperloom-framework-agent-implementation-plan.md`](../claw-dev/docs-zh/hyperloom-framework-agent-implementation-plan.md)
  — PR-A/B/C/D/E/F/G/H/I execution plan.
