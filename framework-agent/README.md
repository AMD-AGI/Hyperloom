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

## Design references

- [`framework-explorer-merged-design.md`](../claw-dev/docs-zh/framework-explorer-merged-design.md)
  — PR exploration tool (existing).
- [`hyperloom-framework-agent-design.md`](../claw-dev/docs-zh/hyperloom-framework-agent-design.md)
  — 5th-role design (v1.3).
- [`hyperloom-framework-agent-implementation-plan.md`](../claw-dev/docs-zh/hyperloom-framework-agent-implementation-plan.md)
  — PR-A/B/C/D/E/F/G/H/I execution plan.
