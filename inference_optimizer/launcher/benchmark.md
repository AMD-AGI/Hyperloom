# Benchmark Config

Default configs live here:

```bash
inference_optimizer/scripts/configs/baseline_sglang.yaml
inference_optimizer/scripts/configs/baseline_vllm.yaml
inference_optimizer/scripts/configs/profile_sglang.yaml
inference_optimizer/scripts/configs/profile_vllm.yaml
```

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
mtime-gated salvage pass over `$INFERENCE_OPTIMIZER_RESCUE_PATHS` (default
`/workspace/`) and copy fresh matches into the task workspace, tagged in
`nonfatal_warnings` (`rescued_from_leaked_path:` / `harvested_leaked_artifact:`).
Extend the scan roots via `$INFERENCE_OPTIMIZER_LEAK_ROOTS` if a script leaks
elsewhere; the default `{framework}_{runner_type}.sh` already respects
`$RESULT_DIR` so salvage normally never fires.

Operators only interact through two `task.params` knobs (full schema in each
`actions/_meta/<action>.yaml`): `params.benchmark_script` (bare sanitized `*.sh`
name; overrides the gpu_type auto-pick) and `params.result_dir` (forwarded as
`$RESULT_DIR`). The Coordinator's `baseline_no_param_change` PolicyGate rule
denies any baseline proposal that changes params after a failure — the agent
must retry with identical params and the run terminates after 3 consecutive
failures.

## Workload-contract reuse (baseline → explore/sweep)

`baseline` materializes its YAML once with the operator's process env (`CONC` /
`ISL` / `OSL` / `TP` / `MAX_MODEL_LEN` / `PRECISION` / `RUN_EVAL` /
`ROCR_VISIBLE_DEVICES` + adaptive `NUM_PROMPTS` / `NUM_WARMUPS`), saves it as
`baseline_config.with_envs.yaml`, and forwards the path as
`task.params["config_path"]` to every `explore` / `sweep` task — so variants
benchmark the **same workload baseline ran** (without it they'd fall back to the
YAML's smoke defaults `TP=1`/`CONC=8`/`ISL=256`/`OSL=256`, ~10x lower tput).
Per-variant `extra_envs` still win (applied last).
