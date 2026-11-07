# Benchmark Config

Default configs live in
`src/hyperloom/inference_optimizer/assets/configs/`, one
`baseline_<framework>.yaml` + `profile_<framework>.yaml` pair per supported
framework (`sglang`, `vllm`, `atom`, `xdit`, `custom`). The names come from
`framework_registry.shipped_config_name()`: a framework registered in
`src/hyperloom/inference_optimizer/framework_registry.py` must ship both files, and
an unset `$FRAMEWORK` resolves to the sglang pair.

`benchmark.model` is **fallback only**: the optimizer overrides it at runtime
with `--model` / `$MODEL_PATH`.

These YAMLs carry the workload, not the server launch. A fresh session resolves
its base launch once, before the first round: `--reference-script`, else
InferenceX main, else the framework cookbook, else (fixed workloads only) the
default table, with a warning. It is frozen as `<session>/base_launch.json`;
resume reads that file and resolves nothing. Every round composes the base, the
operator's `--server-args` / `--extra-env`, and the candidate's change, then
`inference_suite.run_round` runs it into its own run directory (`config.yaml`,
`launch.json`, `report.json`, `server.log`, and the client, eval and profile
artefacts). The round refuses a flag the installed framework's parser does not
know (a candidate's flag fails that candidate as `serve_flag`; a base or
operator flag stops the run) and a framework older than the recipe image's
version.

Before a new model run, verify these fields match the environment:

- `benchmark.model`: model path.
- `benchmark.envs.TP`: tensor parallel size.
- `benchmark.envs.CONC`, `ISL`, `OSL`: workload.
- `benchmark.envs.ROCR_VISIBLE_DEVICES`: GPU pinning.

## Benchmark deadlines

Every actual benchmark subprocess spawn, including baseline, explore, sweep, and
rebench, uses `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` (default `7800`
seconds). It includes server boot and accuracy evaluation; output cannot extend
it. `INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC` (default `600` seconds)
starts only after real server readiness, never during pre-ready startup or for
server-less scriptable workloads. Both settings must be finite and positive.
Session `--max-hours`, cancellation, and admission/phase budgets remain separate;
the whole-sweep budget can end a sweep before all its measurements run.

`PYTHONUNBUFFERED=1` only unbuffers Python output. It does not make shell/native
programs or upstream subprocess capture stream, and busy logs are not proof of
useful progress. Session exhaustion is cooperative, not protection against a frozen
Coordinator. Profile, KernelForge, GEAK, LLM-call, and SGLang's own watchdog
budgets retain their existing contracts.

## Baseline retries and operator server flags

A baseline retry after a failure MUST change its launch
(`params.extra_server_args` / `params.extra_envs`) (prompt RULE F1 — LLM-side
judgement, not a PolicyGate deny); a proposal repeating a recent failing params
fingerprint is dropped as a duplicate. Three consecutive baseline failures with
no enablement engaged stop the run with `stop_reason='baseline_failed'` and
route PRELUDE to CLOSE.

Operator-supplied server flags have a first-class CLI surface:
`optimize --server-args "<framework serve flags>"`. The CLI persists this in
`state.json` and exports it as `INFERENCE_OPTIMIZER_SERVER_ARGS` (a `--resume`
re-exports the persisted value), and every round composes it into the launch
above the base for baseline, profile, explore, and sweep. Explicit
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
`ISL` / `OSL` / `TP` / `MAX_MODEL_LEN` / `PRECISION` /
`ROCR_VISIBLE_DEVICES` + adaptive `NUM_PROMPTS` / `NUM_WARMUPS`), saves it as
`baseline_config.with_envs.yaml`, and forwards the path as
`task.params["config_path"]` to every `explore` / `sweep` task — so variants
benchmark the **same workload baseline ran** (without it they'd fall back to the
YAML's fallback defaults `TP=1`/`CONC=64`/`ISL=1024`/`OSL=1024`). Per-variant
`extra_envs` still win (applied last).
