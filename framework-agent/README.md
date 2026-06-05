# framework-agent

vllm/sglang source-layer optimisation companion for
[`inference_optimizer`](../inference_optimizer/). The live Hyperloom
integration uses the FRAMEWORK_PR discovery path:

- **FRAMEWORK_PR discovery** (`fa phase-discover`) — returns upstream PR
  candidates to `inference_optimizer`, which owns Critic review, diff
  apply, benchmark, KEEP, and REVERT.
- **Standalone PR exploration** (`fa candidates` / `fa explore`) —
  ad-hoc tooling outside the `inference_optimizer` runtime path.

See [`SKILL.md`](./SKILL.md) for the full architectural overview.

## Quick start

```bash
# Install (idempotent)
cd Hyperloom/framework-agent
bash scripts/install.sh        # or: pip install -e '.[test]'

# Live IO discovery path
fa phase-discover --request /path/to/request.json --out -

# Standalone PR exploration
fa schema
fa candidates --request /path/to/request.json
fa explore --request /path/to/request.json [--execute]

# KB management
fa kb list
fa kb search --domain framework_optimization --query "fp8 kv cache"
```

## Tests

```bash
pytest -q                      # all 160+ unit tests
pytest -q tests/test_logging_setup.py tests/test_isolation.py \
          tests/test_decision.py tests/test_explore_modes.py
```

## Used by inference_optimizer

`inference_optimizer` drives PR discovery in the Coordinator-owned
FRAMEWORK_PR phase. After `baseline` completes, the Coordinator calls
`fa phase-discover`, routes each candidate through Critic review, then
uses `FrameworkPrExecutor` to apply the diff to the live framework tree,
benchmark it, and KEEP/REVERT based on throughput and correctness gates.

Gap / keywords are auto-composed from SharedState (`framework`,
`gpu_type`, `model_class`, `precision`).

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
