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
* :func:`default_baseline_config` — pick the shipped sglang / vllm /
  atom YAML based on ``$FRAMEWORK``.

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
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ...paths import asset_root
from ._grid_runner import server_args_env_name
from ._server_patcher import (
    ensure_sglang_patched_for_tracelens,
    ensure_vllm_patched_for_tracelens,
)

log = logging.getLogger(__name__)

# Module-level flag — the RUN_EVAL=false default is the expected steady
# state on most checkouts (see docstring at materialize_config_with_envs);
# the warning is for first-time awareness, not per-variant noise. Emit it
# once per process to keep the log readable.
_RUN_EVAL_DEFAULT_WARN_EMITTED = False


def _visible_gpu_count() -> int:
    """Return how many GPUs are visible to this pod (0 = none / unknown).

    Prefers ``torch.cuda.device_count`` because that's the count sglang /
    vllm actually see (and it doesn't shell out, which keeps the
    BaselineExecutor unit tests that mock ``subprocess.run`` happy). Falls
    back to ``rocm-smi --showid`` (matches ``cli._check_gpu_visibility``)
    when torch is missing or its driver probe hits a fluke. Returns 0 on
    every failure path so callers can skip the clamp and let downstream
    surface the real "no GPU" error instead of inventing a wrong TP.

    Override (escape hatch for hosts where torch + rocm-smi disagree, or
    for unit tests that exercise the clamp without GPU hardware):
    ``$INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT=N``.
    """
    override = os.environ.get("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "").strip()
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            log.warning(
                "INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT=%r is not an int; "
                "ignoring override.", override,
            )
    try:
        import torch  # type: ignore[import-not-found]
        count = int(torch.cuda.device_count() or 0)
        if count > 0:
            return count
    except Exception:
        pass
    if shutil.which("rocm-smi"):
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, PermissionError, OSError):
            return 0
        if proc.returncode == 0:
            # ``rocm-smi --showid`` emits 6+ lines per device
            # (``GPU[N] : Device Name``, ``GPU[N] : Device ID``, etc.).
            # Counting raw lines that start with ``GPU[`` over-counts by 6x
            # (a 4-GPU pod reports 24 "visible" GPUs and the clamp never
            # fires). Deduplicate by GPU index so 4 distinct ``GPU[0..3]``
            # prefixes return 4.
            indices: set[str] = set()
            for line in (proc.stdout or "").splitlines():
                stripped = line.strip()
                if stripped.startswith("GPU["):
                    idx, _, _ = stripped[4:].partition("]")
                    if idx:
                        indices.add(idx)
            return len(indices)
    return 0


def _tracelens_patch_enabled() -> bool:
    """Read the ``HYPERLOOM_ENABLE_PATCH`` kill switch (default on).

    Set ``HYPERLOOM_ENABLE_PATCH=0`` to disable runtime patching of
    vLLM / SGLang — Hyperloom then keeps today's safe behaviour
    (no TraceLens-only profiler flags injected). Useful when:

    * the user runs a custom vLLM/SGLang fork they don't want touched,
    * the patcher itself ships a bug and the user needs an escape hatch
      without a Hyperloom release,
    * compliance / audit forbids modifying installed packages at runtime.

    Default is ``"1"`` (patching on) because the patches are backward-
    compatible and add no flags by themselves — they only enable
    capabilities Hyperloom requests via ``EXTRA_VLLM_ARGS`` /
    ``EXTRA_SGLANG_ARGS`` when patching succeeds.
    """
    return os.environ.get("HYPERLOOM_ENABLE_PATCH", "1").strip() != "0"


def default_baseline_config() -> Path:
    """Resolve the shipped Magpie YAML based on ``$FRAMEWORK`` env.

    Returns the sglang YAML when ``$FRAMEWORK`` is unset/unknown so existing
    sglang-default tests keep passing.
    """
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    if fw == "atom":
        name = "baseline_atom.yaml"
    elif fw == "vllm":
        name = "baseline_vllm.yaml"
    else:
        name = "baseline_sglang.yaml"
    return asset_root() / "scripts" / "configs" / name


def materialize_config_with_envs(
    config_path: Path,
    output_dir: Path,
    *,
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
    * ``GPU_TYPE`` / ``gpu_type`` arg → ``benchmark.runner_type`` AND pin
      ``benchmark.benchmark_script`` to the generic
      ``{framework}_{gpu_type}.sh`` so Magpie's resolver hits priority 1
      (explicit user override) and never silently falls through to the
      InferenceX native script (e.g. ``dsr1_fp8_mi300x.sh``) which
      hardcodes ``--result-dir /workspace/`` and ignores
      ``EXTRA_SGLANG_ARGS`` / ``EXTRA_VLLM_ARGS``. See
      ``design/magpie-generic-script-and-user-data-path.md``.
    * ``benchmark_script`` arg (when non-empty, must be sanitized by the
      executor via :func:`_grid_runner.sanitize_script_name`) re-pins
      ``benchmark.benchmark_script`` AFTER the gpu_type-derived generic
      script above so an operator-supplied override (typically a model
      script the operator deliberately wants tested) wins. See SKILL.md
      "Magpie leak-path salvage" for use cases.
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
    * ``extra_server_args`` is the framework-neutral payload-surface
      slot (the legacy name was ``extra_sglang_args``). The
      materializer routes its value into the framework-specific env
      name (``EXTRA_SGLANG_ARGS`` / ``EXTRA_VLLM_ARGS`` /
      ``EXTRA_ATOM_ARGS``) based on the framework declared in the
      YAML's ``benchmark.framework``.
    * ``extra_envs`` overrides any of the above.

    Returns the path to the materialized YAML written under ``output_dir``.
    Reuses the file name across calls so callers can locate it predictably.
    """
    server_args = (extra_server_args or "").strip()
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
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{gpu_type}.sh"
        else:
            bench.pop("benchmark_script", None)
    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        # Magpie resolves an empty benchmark.inferencex_path to its sibling
        # checkout (usually $MAGPIE_DIR/InferenceX). Hyperloom's profile path
        # patches the checkout addressed by $INFERENCEX_PATH, so persist the
        # same path into the YAML to keep Magpie's runtime and Hyperloom's
        # patch target aligned.
        bench["inferencex_path"] = inferencex_path
    envs = bench.setdefault("envs", {})
    for env_key in (
        "CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP",
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = int(val)
    # RANDOM_RANGE_RATIO is a float (skill default 1.0; common values include
    # 0.5). It feeds the steady-state delay/max formulas below; do not coerce
    # to int or the prefill window estimate collapses for fractional ratios.
    r_env = os.environ.get("RANDOM_RANGE_RATIO", "").strip()
    if r_env:
        envs["RANDOM_RANGE_RATIO"] = float(r_env)
    rocr_env = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if rocr_env:
        envs["ROCR_VISIBLE_DEVICES"] = rocr_env
    tp_from_env = os.environ.get("TP", "").strip()
    tp_from_yaml = envs.get("TP")
    rocr_yaml = str(envs.get("ROCR_VISIBLE_DEVICES") or "").strip()
    rocr_devices = [d.strip() for d in rocr_yaml.split(",") if d.strip()]
    if tp_from_env:
        resolved_tp = int(tp_from_env)
    elif rocr_yaml and not tp_from_yaml:
        # Derive TP from user-pinned GPU list when the YAML template
        # doesn't set TP — avoids inheriting a stale TP from templates
        # built for a different GPU count.
        resolved_tp = len(rocr_devices)
        envs["TP"] = resolved_tp
    else:
        resolved_tp = int(tp_from_yaml or 1)
    # Auto-clamp TP to the pod's visible GPU count. The shipped YAML defaults
    # to TP=8 (full DGX-style node), so a 4-GPU sandbox would otherwise launch
    # sglang/vllm with `--tensor-parallel-size=8` and crash with `HIP error:
    # invalid device ordinal`. Override via
    # $INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP=1.
    if os.environ.get("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "").strip() != "1":
        visible = _visible_gpu_count()
        if visible and resolved_tp > visible:
            log.warning(
                "TP=%d but only %d GPU(s) visible to this pod; clamping "
                "TP=%d so sglang/vllm can actually load weights. Export "
                "INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP=1 to opt out (the "
                "subprocess will then fail at server launch).",
                resolved_tp, visible, visible,
            )
            resolved_tp = visible
    envs["TP"] = resolved_tp
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

    # TraceLens #194: compute steady-state window for profiling configs.
    # Only inject when this is a profile yaml — detected by PROFILE env or
    # `profiler.torch_profiler.enabled: true` in the YAML. The formulas
    # match the TraceLens magpie-benchmark-profiling skill (Option A:
    # targeted steady-state window) so that Hyperloom-driven profiles
    # capture the same decode-heavy slice as a manual TraceLens run.
    #
    # Skill formulas (RANDOM_RANGE_RATIO defaults to 1.0 if absent):
    #   max_iters   = min(1024, max(256, OSL * 16 / CONC))
    #   delay_iters = OSL * (R + 1) * 3 - max_iters / 2
    #
    # Example OSL=1024 / CONC=32 / R=1 → max=512, delay=5888 (vs. the
    # previous 5*CONC / clamp(16*OSL/CONC,4,64) which gave 160 / 64 —
    # roughly 1/8 of the steady-state window the skill recommends).
    is_profile = (
        str(envs.get("PROFILE", "")).strip() == "1"
        or (bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True)
    )
    profile_num_prompts: int | None = None
    if is_profile:
        try:
            r_val = float(envs.get("RANDOM_RANGE_RATIO", 1.0))
        except (TypeError, ValueError):
            r_val = 1.0
        safe_conc = max(conc_val, 1)
        safe_osl = max(osl_val, 1)
        max_iters = min(1024, max(256, (osl_val * 16) // safe_conc))
        delay_iters = int(osl_val * (r_val + 1) * 3 - max_iters / 2)
        # Guard against pathological inputs (tiny OSL / huge R producing a
        # negative delay collapses to "profile from step 0", which still
        # captures warmup). Clamp to >= 0.
        if delay_iters < 0:
            delay_iters = 0
        # TraceLens #194 §2: NUM_PROMPTS must be large enough that the
        # benchmark engine reaches `delay_iters + max_iters` decode steps
        # before it runs out of prompts. With continuous batching at
        # concurrency CONC and average output length OSL, processing N
        # prompts costs roughly N * OSL / CONC engine iterations; invert
        # to get the prompt floor, then apply a 2x safety buffer for
        # prefill / scheduling drift. Hyperloom owns this — we ignore any
        # NUM_PROMPTS the caller passes when PROFILE is on, otherwise an
        # under-sized value silently empties the trace.
        required_iters = delay_iters + max_iters
        iters_to_prompts = max(
            1, (required_iters * safe_conc + safe_osl - 1) // safe_osl,
        )
        profile_num_prompts = max(safe_conc, iters_to_prompts * 2)
        fw = str(bench.get("framework") or "").lower()
        # atom check must precede vllm/sglang branches: atom doesn't
        # contain a vllm/sglang substring today but the explicit
        # ordering keeps a future framework name (e.g. "atom-vllm") from
        # accidentally falling into the wrong branch. atom's HTTP
        # start_profile/stop_profile path is driven by the InferenceX
        # bench client's --profile flag (added by benchmark_lib.sh when
        # PROFILE=1), and the trace directory is wired via
        # atom_mi*x.sh's --torch-profiler-dir, so this Python layer has
        # no profiler envs to set for atom — and must NOT inject
        # --profiler-config.* style flags (atom argparse rejects them).
        is_atom = "atom" in fw
        # Issue #194 §4 / §5: TraceLens-required profiler flags exist
        # only in patched vLLM / SGLang builds. Try to apply the
        # TraceLens patch set to the in-container install; on success
        # we inject the extra flags below, on failure we silently fall
        # back to today's safe set so vanilla images keep working.
        # Default-on, disable via HYPERLOOM_ENABLE_PATCH=0. atom has no
        # TraceLens patch set (atom's torch_profiler integration is
        # native), so we skip the patcher entirely for atom — calling
        # the sglang patcher would no-op but spam install warnings.
        tracelens_patch_ok = False
        if _tracelens_patch_enabled() and not is_atom:
            if "vllm" in fw:
                tracelens_patch_ok = ensure_vllm_patched_for_tracelens()
            else:
                tracelens_patch_ok = ensure_sglang_patched_for_tracelens()
        if is_atom:
            # atom writes trace files to
            # <torch_profiler_dir>/rank_<N>/*.pt.trace.json.gz which our
            # _candidate_trace_dirs probe matches unchanged.
            #
            # atom's profiler is HTTP-driven (POST /start_profile at server-up,
            # /stop_profile at run end) and — unlike sglang's start_step/num_steps
            # or vLLM's delay_iterations/max_iterations — has NO internal capture
            # window: it records the ENTIRE bench-client run. The only lever that
            # bounds atom's profiled decode-iteration count is the *workload
            # length* (OSL) + prompt count, and OSL can only be clamped at the
            # Magpie client layer (`--output-len`), never from this Python layer.
            #
            # Therefore the atom profile window lives in ONE place — Magpie's
            # atom_mi*x.sh, which clamps OSL + NUM_PROMPTS when PROFILE=1 (see
            # ATOM_PROFILE_OSL / ATOM_PROFILE_NUM_PROMPTS). We must NOT also force
            # the sglang/vllm steady-state NUM_PROMPTS here: at OSL=1024 that
            # ~780-prompt window is ~4096 decode iters, which starves aiter's
            # shared-memory broadcast ring ("No available shared memory broadcast
            # block found in 60s") until a tiny BROADCAST collective trips the
            # 600s NCCL watchdog and aborts the run *before* /stop_profile flushes
            # — leaving rank_*/ empty. Defer to Magpie (single source of truth).
            profile_num_prompts = None
        elif "vllm" in fw:
            existing_vllm_args = str(envs.get("EXTRA_VLLM_ARGS", ""))
            profiler_args_parts = [
                f"--profiler-config.delay_iterations {delay_iters}",
                f"--profiler-config.max_iterations {max_iters}",
            ]
            if tracelens_patch_ok:
                # #194 §4: a TraceLens-patched vLLM exposes
                # capture_torch_profiler_dir + detailed_trace_annotation
                # on its ProfilerConfig. Both are new dataclass fields
                # with safe defaults — unpatched vLLM rejects them as
                # unknown JSON keys, which is why we gate on the
                # patcher result.
                capture_dir = output_dir / "capture_traces"
                profiler_args_parts.append(
                    "--profiler-config.capture_torch_profiler_dir "
                    f"{capture_dir}"
                )
                profiler_args_parts.append(
                    "--profiler-config.detailed_trace_annotation True"
                )
            profiler_args = " ".join(profiler_args_parts)
            if "delay_iterations" not in existing_vllm_args:
                envs["EXTRA_VLLM_ARGS"] = (
                    f"{existing_vllm_args} {profiler_args}".strip()
                )
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
            if tracelens_patch_ok:
                # #194 §5: a TraceLens-patched SGLang exposes
                # --enable-shape-discovery-for-cuda-graph-profile as a
                # server CLI flag. Without the patch SGLang's argparse
                # errors out on this flag — gate strictly on the
                # patcher result.
                existing_sglang = str(envs.get("EXTRA_SGLANG_ARGS", ""))
                if "shape-discovery-for-cuda-graph-profile" not in existing_sglang:
                    envs["EXTRA_SGLANG_ARGS"] = (
                        f"{existing_sglang} "
                        "--enable-shape-discovery-for-cuda-graph-profile"
                    ).strip()

    if profile_num_prompts is not None:
        # Profile mode: force-override caller/YAML NUM_PROMPTS — we need a
        # specific minimum to reach the steady-state window, and an
        # under-sized value silently empties the trace.
        envs["NUM_PROMPTS"] = profile_num_prompts
    else:
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
            global _RUN_EVAL_DEFAULT_WARN_EMITTED
            if not _RUN_EVAL_DEFAULT_WARN_EMITTED:
                log.warning(
                    "RUN_EVAL defaulted to false (no per-variant accuracy "
                    "gate): Magpie main / InferenceX main disagree on "
                    "`run_eval --concurrent-requests`. Export RUN_EVAL=true "
                    "(or pass extra_envs={'RUN_EVAL': 'true'}) once your "
                    "InferenceX checkout accepts that flag. This warning "
                    "fires once per process."
                )
                _RUN_EVAL_DEFAULT_WARN_EMITTED = True
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = output_dir / out_name
    with materialized.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return materialized


__all__ = [
    "default_baseline_config",
    "materialize_config_with_envs",
]
