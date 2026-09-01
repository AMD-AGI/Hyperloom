# Benchmark Config

Default configs live in
`src/hyperloom/inference_optimizer/assets/configs/`, one
`baseline_<framework>.yaml` + `profile_<framework>.yaml` pair per supported
framework (`sglang`, `vllm`, `atom`, `xdit`, `custom`). The resolvers are the
source of truth: `_workload_envs.py` for the baseline map and
`_default_profile_config()` in
`src/hyperloom/orchestrator/actions/executors/profile.py` for the profile map.

Two fields in each YAML are **fallback only** — the optimizer overrides them at
runtime:

- `benchmark.model` <- `--model` / `$MODEL_PATH`
- `benchmark.runner_type` <- `--gpu-type` / `$GPU_TYPE` / rocm-smi auto-detect

`benchmark.benchmark_script` is deliberately NOT set in the shipped YAMLs. At
materialize time Hyperloom pins it to `{framework}_{runner_type}.sh` (e.g.
`sglang_mi300x.sh` / `sglang_mi355x.sh`) so Magpie's resolver hits priority 1
(explicit user override) and uses the generic script — which respects
`RESULT_DIR` and `EXTRA_*_ARGS`. Each shipped YAML has a commented
`# benchmark_script: ...` template right under `framework:` for manual debug
overrides; Orchestration can also route per-task via `params.benchmark_script`
(sanitized).

Before a new model run, verify these fields match the environment:

- `benchmark.model`: model path.
- `benchmark.envs.TP`: tensor parallel size.
- `benchmark.envs.CONC`, `ISL`, `OSL`: workload.
- `benchmark.envs.ROCR_VISIBLE_DEVICES`: GPU pinning.
- `benchmark.envs.PATH`: must lead with the launcher Python's bin dir
  (`$(dirname "$PYTHON")`).

## Magpie leak-path salvage (`INFERENCE_OPTIMIZER_RESCUE_PATHS`)

In-loop, defense-in-depth — the launcher does not touch this. Magpie shell
wrappers hardcode artifacts under `/workspace/` (`inferencex_result.json`,
`server.log`, `gpu_metrics.csv`, `profile_*.trace.json.gz`). When a task's
in-workspace search finds no usable measurement, the executors run an
mtime-gated salvage pass over `$INFERENCE_OPTIMIZER_RESCUE_PATHS` (unset =
no salvage) and copy fresh matches into the task workspace, tagged in
`nonfatal_warnings` (`rescued_from_leaked_path:` / `harvested_leaked_artifact:`).
Extend the scan roots via `$INFERENCE_OPTIMIZER_LEAK_ROOTS` if a script leaks
elsewhere; the default `{framework}_{runner_type}.sh` already respects
`$RESULT_DIR` so salvage normally never fires.

Operators only interact through two `task.params` knobs:
`params.benchmark_script` (bare sanitized `*.sh`
name; overrides the gpu_type auto-pick) and `params.result_dir` (forwarded as
`$RESULT_DIR`). A baseline retry after a failure MUST change at least one of
`params.benchmark_script` / `params.result_dir` / `params.extra_server_args` /
`params.extra_envs` (prompt RULE F1 — LLM-side judgement, not a PolicyGate
deny); a proposal repeating a recent failing params fingerprint is dropped as a
duplicate. Three consecutive baseline failures with no enablement engaged stop
the run with `stop_reason='baseline_failed'` and route PRELUDE to CLOSE.

Operator-supplied server flags have a first-class CLI surface:
`optimize --server-args "<framework serve flags>"`. The CLI persists this in
`state.json` and exports it as `INFERENCE_OPTIMIZER_SERVER_ARGS` (a `--resume`
re-exports the persisted value), and YAML materialization routes it into the
framework-specific Magpie env (`EXTRA_VLLM_ARGS`, `EXTRA_SGLANG_ARGS`, or
`EXTRA_ATOM_ARGS`) for baseline, profile, explore, and sweep. Explicit
`--max-model-len` / `$MAX_MODEL_LEN` wins over the auto `ISL+OSL+headroom`
calculation. A comma `$CONC` value such as `4,16,128` is accepted for
compatibility; the single baseline CONC becomes the first value. Use
`--conc-sweep-concs` for the explicit sweep ladder.

The configuration arm may deliberately ablate operator-supplied server flags. A grid variant
can carry `remove_args` to delete inherited CLI flags before its own
`extra_args` are appended, or `unset_envs` to remove inherited environment
keys before `extra_envs` are applied. Use these fields when testing whether a
pinned operator/base knob is harmful; do not approximate deletion by adding
another unrelated flag. Removal controls are part of the variant fingerprint
and are recorded in `explore_search`.

## Workload-contract reuse (baseline → explore/sweep)

`baseline` materializes its YAML once with the operator's process env (`CONC` /
`ISL` / `OSL` / `TP` / `MAX_MODEL_LEN` / `PRECISION` / `RUN_EVAL` /
`ROCR_VISIBLE_DEVICES` + adaptive `NUM_PROMPTS` / `NUM_WARMUPS`), saves it as
`baseline_config.with_envs.yaml`, and forwards the path as
`task.params["config_path"]` to every `explore` / `sweep` task — so variants
benchmark the **same workload baseline ran** (without it they'd fall back to the
YAML's fallback defaults `TP=1`/`CONC=64`/`ISL=1024`/`OSL=1024`). Per-variant
`extra_envs` still win (applied last).
