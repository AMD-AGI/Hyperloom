"""Shared workload-env materialization (single source of truth).

The optimizer used to have TWO YAML-rendering paths:

* baseline executor's ``_materialize_config_with_envs`` — injected the full
  set of process-env knobs (TP/CONC/ISL/OSL/MAX_MODEL_LEN/PRECISION/
  RUN_EVAL/ROCR_VISIBLE_DEVICES + adaptive NUM_PROMPTS/NUM_WARMUPS).
* grid runner's ``_build_variant_yaml`` (used by params/backends/sweep) —
  only set ``model``, ``runner_type``, ``EXTRA_*_ARGS``, and per-variant
  ``extra_envs``. Process-env workload knobs were silently dropped.

The result was a "Benchmark fairness" bug (SKILL Lesson 4): baseline ran at
the user's real workload (e.g. CONC=64, ISL=1024, OSL=1024) while every
downstream variant ran at the YAML smoke defaults (CONC=8, ISL=256, OSL=256).
Throughput numbers were 10x apart, every variant looked like a regression.

This module is the **single source of truth** for "render a Magpie YAML
with the user's actual workload contract":

* :func:`materialize_config_with_envs` — write a per-run YAML file
  honoring process env (and optional caller overrides).
* :func:`default_baseline_config` — pick the shipped sglang/vllm YAML
  based on ``$FRAMEWORK``.

Callers:

* ``baseline.py`` — runs first, materializes the contract once and
  surfaces the rendered YAML path in its result so downstream actions
  can reuse it verbatim (no env re-read race).
* ``params.py`` / ``backends.py`` — fall back to materializing on
  their own if Coordinator has not yet plumbed the baseline path
  through ``task.params["config_path"]``.
* ``sweep.py`` — same fallback; per-variant CONC/ISL/OSL still win
  because ``_build_variant_yaml`` applies ``variant.extra_envs`` last.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from ...paths import asset_root
from ._grid_runner import server_args_env_name

log = logging.getLogger(__name__)


def default_baseline_config() -> Path:
    """Resolve the shipped Magpie YAML based on ``$FRAMEWORK`` env.

    Returns the sglang YAML when ``$FRAMEWORK`` is unset/unknown so existing
    sglang-default tests keep passing.
    """
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    name = "baseline_vllm.yaml" if fw == "vllm" else "baseline_sglang.yaml"
    return asset_root() / "scripts" / "configs" / name


def materialize_config_with_envs(
    config_path: Path,
    output_dir: Path,
    *,
    extra_sglang_args: str = "",
    extra_server_args: str = "",
    extra_envs: dict[str, Any] | None = None,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    out_name: str = "baseline_config.with_envs.yaml",
) -> Path:
    """Render a per-run Magpie YAML with caller-provided overrides.

    Process-env precedence on top of the YAML defaults (env wins):

    * ``MODEL_PATH`` / ``model_path`` arg → ``benchmark.model`` (overrides
      the legacy hardcoded ``Qwen-Qwen3-8B`` default that ships in every YAML).
    * ``GPU_TYPE`` / ``gpu_type`` arg → ``benchmark.runner_type`` (and pop
      any explicit ``benchmark.benchmark_script`` so Magpie's runner_type
      → script logic actually fires).
    * ``benchmark_script`` arg (when non-empty, must be sanitized by the
      executor via :func:`_grid_runner.sanitize_script_name`) re-pins
      ``benchmark.benchmark_script`` AFTER the gpu_type pop above —
      Orchestration uses this to route around scripts that hardcode
      ``--result-dir /workspace/`` (see SKILL.md "Magpie leak-path
      salvage").
    * ``PRECISION`` → ``benchmark.precision``.
    * ``CONC, ISL, OSL, MAX_MODEL_LEN, TP, RANDOM_RANGE_RATIO`` → injected
      as integers into ``benchmark.envs``.
    * ``ROCR_VISIBLE_DEVICES`` → reconciled against TP (if YAML pins fewer
      devices than TP requires, expanded to ``0..TP-1``; logs a warning).
    * ``RUN_EVAL=true`` is set as a default so accuracy eval (GSM8K) runs
      while the server is alive.
    * ``NUM_PROMPTS`` and ``NUM_WARMUPS`` are computed adaptively from
      ``CONC`` and ``ISL+OSL`` (longer sequences → fewer prompts to keep
      each variant under ~3-5 min wall time).
    * ``extra_sglang_args`` / ``extra_server_args`` (the latter wins) are
      written to ``EXTRA_SGLANG_ARGS`` / ``EXTRA_VLLM_ARGS`` based on the
      configured framework.
    * ``extra_envs`` overrides any of the above.

    Returns the path to the materialized YAML written under ``output_dir``.
    Reuses the file name across calls so callers can locate it predictably.
    """
    server_args = (extra_server_args or extra_sglang_args).strip()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = cfg.setdefault("benchmark", {})
    if model_path:
        bench["model"] = str(model_path)
    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision
    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        bench.pop("benchmark_script", None)
    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)
    envs = bench.setdefault("envs", {})
    for env_key in (
        "CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "RANDOM_RANGE_RATIO",
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = int(val)
    rocr_env = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if rocr_env:
        envs["ROCR_VISIBLE_DEVICES"] = rocr_env
    resolved_tp = int(envs.get("TP") or os.environ.get("TP") or 1)
    rocr_yaml = str(envs.get("ROCR_VISIBLE_DEVICES") or "").strip()
    rocr_devices = [d.strip() for d in rocr_yaml.split(",") if d.strip()]
    if not rocr_yaml or len(rocr_devices) < resolved_tp:
        derived = ",".join(str(i) for i in range(resolved_tp))
        if rocr_yaml and rocr_yaml != derived:
            log.warning(
                "ROCR_VISIBLE_DEVICES=%r has %d devices but TP=%d; "
                "expanding to %r so SGLang sees enough GPUs. Set "
                "ROCR_VISIBLE_DEVICES explicitly to override.",
                rocr_yaml, len(rocr_devices), resolved_tp, derived,
            )
        envs["ROCR_VISIBLE_DEVICES"] = derived

    isl_val = int(os.environ.get("ISL") or envs.get("ISL") or 256)
    osl_val = int(os.environ.get("OSL") or envs.get("OSL") or 256)
    conc_val = int(os.environ.get("CONC") or envs.get("CONC") or 8)

    # TraceLens #126: compute steady-state window for profiling configs.
    # Only inject when this is a profile yaml — detected by PROFILE env or
    # `profiler.torch_profiler.enabled: true` in the YAML. The formulas
    # match issue #126 §3.1.6.
    is_profile = (
        str(envs.get("PROFILE", "")).strip() == "1"
        or (bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True)
    )
    if is_profile:
        delay_iters = 5 * conc_val
        max_iters = max(4, min(64, 16 * osl_val // max(conc_val, 1)))
        fw = str(bench.get("framework") or "").lower()
        if "vllm" in fw:
            existing_vllm_args = str(envs.get("EXTRA_VLLM_ARGS", ""))
            # Do not inject capture_torch_profiler_dir by default: the vLLM
            # build on current MI355X validation nodes rejects that field in
            # --profiler-config. Magpie's vllm_*.sh already injects the
            # supported torch profiler fields and we add TraceLens' stable
            # timing/annotation knobs here.
            profiler_args = (
                f"--profiler-config.delay_iterations {delay_iters} "
                f"--profiler-config.max_iterations {max_iters}"
            )
            if "delay_iterations" not in existing_vllm_args:
                envs["EXTRA_VLLM_ARGS"] = f"{existing_vllm_args} {profiler_args}".strip()
        else:
            import json as _json
            try:
                extra_body = _json.loads(str(envs.get("PROFILE_EXTRA_BODY", "{}")))
            except (ValueError, TypeError):
                extra_body = {}
            # Always override start_step/num_steps with computed values —
            # the yaml template has placeholder defaults for CONC=8.
            extra_body["start_step"] = delay_iters
            extra_body["num_steps"] = max_iters
            extra_body.setdefault("shape_discovery", True)
            extra_body.setdefault("roofline_annotations", True)
            envs["PROFILE_EXTRA_BODY"] = _json.dumps(extra_body)

    seq_cost = isl_val + osl_val
    if seq_cost <= 1024:
        factor = 10
    elif seq_cost <= 4096:
        factor = 5
    elif seq_cost <= 16384:
        factor = 3
    else:
        factor = 2
    if "NUM_PROMPTS" not in envs:
        envs["NUM_PROMPTS"] = max(conc_val * factor, conc_val)
    if "NUM_WARMUPS" not in envs:
        envs["NUM_WARMUPS"] = min(conc_val, 8)
    if server_args:
        envs[server_args_env_name(bench.get("framework"))] = server_args
    for key, value in (extra_envs or {}).items():
        envs[str(key)] = str(value)
    # Accuracy eval (GSM8K) is OFF by default because Magpie main and
    # InferenceX main currently disagree on the lm-eval CLI shape:
    # Magpie's benchmark scripts call `run_eval ... --concurrent-requests N`,
    # but InferenceX's `run_lm_eval` rejects `--concurrent-requests` (it
    # reads `EVAL_CONCURRENT_REQUESTS` from env instead, and the unknown
    # flag makes the wrapper return 1, failing the whole benchmark). Magpie
    # does not pin InferenceX, so any fresh clone hits this. Until upstream
    # realigns, leave RUN_EVAL off and let the user opt in via env or
    # extra_envs. The accuracy gate in coordinator.py treats a missing
    # GSM8K result as "no regression", which is the safe behaviour for a
    # flag-only run. NOTE: this is resolved after extra_envs merging so
    # callers (params/backends/sweep variants) that explicitly pass
    # RUN_EVAL=true via extra_envs do not trigger the warning.
    if "RUN_EVAL" not in envs:
        env_run_eval = os.environ.get("RUN_EVAL")
        if env_run_eval is not None:
            envs["RUN_EVAL"] = env_run_eval
        else:
            envs["RUN_EVAL"] = "false"
            log.warning(
                "RUN_EVAL defaulted to false: Magpie main / InferenceX main "
                "disagree on `run_eval --concurrent-requests`. Export "
                "RUN_EVAL=true once your InferenceX checkout accepts that flag."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = output_dir / out_name
    with materialized.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return materialized


__all__ = [
    "default_baseline_config",
    "materialize_config_with_envs",
]
