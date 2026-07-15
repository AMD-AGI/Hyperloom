# framework-agent

vllm/sglang source-layer optimisation companion for
[`inference_optimizer`](../../inference_optimizer/). The live Hyperloom
integration uses the FRAMEWORK discovery path:

- **FRAMEWORK discovery** (`fa phase-discover`) — returns upstream PR
  candidates to `inference_optimizer`, which owns Critic review, diff
  apply, benchmark, KEEP, and REVERT.
- **Standalone PR exploration** (`fa candidates` / `fa explore`) —
  ad-hoc tooling outside the `inference_optimizer` runtime path.

- **Enablement** (opt-in) — make a currently **non-runnable**
  `(model, backend)` combo *run at all* by authoring a bridging patch. Unlike
  the perf path (gated on throughput), enablement is gated on **runnability**
  (server boots + minimal correctness). See
  [Enablement path](#enablement-path-non-runnable-model--backend-combos).

See [`SKILL.md`](./SKILL.md) for the full architectural overview.

## Enablement path (non-runnable model + backend combos)

When a `(model, backend)` combo will not start, the enablement building
blocks turn the failure into an authored bridging patch, gated on *does it
run* rather than *is it faster*:

1. **Classify** — `hyperloom.agents.framework.enablement.classify_failure(log)` parses a
   launch/import/build log into a `FailureSignature`
   (`missing_model_arch` / `unsupported_dtype` / `hip_kernel_missing` /
   `import_error` / `shape_mismatch` / `not_implemented` /
   `capability_disabled`) with the offending file/symbol and a `bridge_layer`.
2. **Discover** — `hyperloom.agents.framework.enablement_ops.build_search_plan(...)`
   picks the repos to scout (the framework repo, plus ROCm/HIP/aiter via
   `repo_map.bridge_repo_urls` for the failure's bridge layer) and ranks
   candidate PR titles for *enablement* intent (`enable` / `support` / `add` /
   `fix` / `port`).
3. **Author** — `hyperloom.agents.framework.enablement_ops.build_mandate(...)`
   produces the `EnablementMandate` (allowed source roots + task description +
   patch invariants) handed to Hyperloom's `enablement_specialist` /
   `SpecialistRunner`, which writes the patch into an isolated worktree.
4. **Verify** — `hyperloom.agents.framework.enablement.runnable_decision(...)` is the
   KEEP/REVERT gate: the launch probe must exit 0 (no timeout) and any minimal
   correctness check must pass; the same failure re-appearing is a reject.

Editing ROCm/HIP source (`/opt/rocm`) is a **default-on** part of the
enablement path — the IO-side allowlist always surfaces those roots (alongside
the always-allowed `aiter`).

## Quick start

```bash
# Install from the repo root (idempotent)
cd Hyperloom
bash src/hyperloom/agents/framework/scripts/install.sh        # or: pip install -e '.[test]'

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
          tests/test_decision.py tests/test_explorer.py
```

## Used by inference_optimizer

`inference_optimizer` drives PR discovery in the Coordinator-owned
FRAMEWORK_AGENT phase. After `baseline` completes, the Coordinator calls
`fa phase-discover`, routes each candidate through Critic review, then
uses `FrameworkAgentExecutor` to apply the diff to the live framework tree,
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

See `src/hyperloom/inference_optimizer/SKILL.md` "Framework-Agent as Bandit Arm"
and `src/hyperloom/orchestrator/actions/executors/framework_agent.py`.

## Design references

The current runtime contract is documented in this directory's `SKILL.md`,
`AGENTS.md`, and `references/` files. Historical design notes live outside the
packaged Hyperloom tree.
