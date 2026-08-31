# framework-agent

vllm/sglang source-layer optimisation companion for
[`inference_optimizer`](../../inference_optimizer/).

- **Standalone PR exploration** (`fa candidates` / `fa explore`) —
  ad-hoc tooling outside the `inference_optimizer` runtime path.
- **Shared tables** — repo map, PR-KB adapters and the KB partitions the
  Coordinator reads through `hyperloom.orchestrator.framework`.

- **Enablement** (opt-in) — make a `(model, backend)` combo that is
  **non-runnable**, or that boots but **fails its accuracy eval**, *run
  correctly* by authoring a bridging patch. Unlike the perf path (gated on
  throughput), enablement is gated on **runnability** (server boots + minimal
  correctness) or, for an eval-origin trigger, meeting the accuracy floor. See
  [Enablement path](#enablement-path-non-runnable-model--backend-combos).

See [`SKILL.md`](./SKILL.md) for the full architectural overview.

## Enablement path (non-runnable model + backend combos)

When a `(model, backend)` combo will not start, or it starts but fails its
accuracy eval, the enablement building blocks turn the failure into an authored
bridging patch, gated on *does it run correctly* rather than *is it faster*:

1. **Classify** — `hyperloom.agents.framework.enablement.classify_failure(log)` parses a
   launch/import/build/eval log into a `FailureSignature`
   (`missing_model_arch` / `unsupported_dtype` / `hip_kernel_missing` /
   `import_error` / `shape_mismatch` / `not_implemented` /
   `capability_disabled` / `accuracy_below_floor` /
   `eval_generation_pathology` / `eval_runtime_failure`)
   with the offending file/symbol and a `bridge_layer`.
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
   correctness check must pass; the same failure re-appearing is a reject. For an
   eval-origin trigger the gate additionally re-runs the accuracy eval and
   REVERTs a patch that boots but still misses the accuracy floor.

Editing ROCm/HIP source (`/opt/rocm`) is a **default-on** part of the
enablement path — the IO-side allowlist always surfaces those roots (alongside
the always-allowed `aiter`).

## Quick start

```bash
# Install from the repo root; the fa CLI ships with the distribution.
cd Hyperloom
pip install -e '.[test]'

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
pytest -q src/hyperloom/agents/framework/tests/   # all framework-agent unit tests
pytest -q src/hyperloom/agents/framework/tests/test_logging_setup.py \
          src/hyperloom/agents/framework/tests/test_isolation.py \
          src/hyperloom/agents/framework/tests/test_decision.py \
          src/hyperloom/agents/framework/tests/test_explorer.py
```

## Used by inference_optimizer

`inference_optimizer` discovers upstream PR candidates with a
`candidate_discovery_specialist` inside the FRAMEWORK_AGENT phase, and lands
each one through `integrate_patch` with `patch_source='upstream_pr'` — the
same apply / vet / bench / KEEP-REVERT pipeline every other patch source
uses. This package supplies the repo map and PR-KB adapters that path reads;
it is not itself invoked as a subprocess.

Gap / keywords are auto-composed from SharedState (`framework`,
`gpu_type`, `model_class`, `precision`).

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
    --model "$MODEL_PATH" \
    --framework sglang \
    --model-class dense \
    --gpu-type mi300x \
    --no-kernel \
    --max-hours 2
```

See `src/hyperloom/inference_optimizer/SKILL.md` "FRAMEWORK_AGENT phase — the
optimisation phase" and `src/hyperloom/orchestrator/phases/framework.py`.

## Design references

The current runtime contract is documented in this directory's `SKILL.md`.
Historical design notes live outside the packaged Hyperloom tree.
