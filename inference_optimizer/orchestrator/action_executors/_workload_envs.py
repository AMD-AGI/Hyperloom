# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared workload-env materialization (single source of truth).

Two YAML-rendering paths used to diverge — baseline injected the full process-
env workload contract while the grid runner dropped it — so downstream variants
ran at YAML smoke defaults and every one looked like a regression (SKILL Lesson
4 "Benchmark fairness"). This module is the single source of truth for rendering
a Magpie YAML with the user's actual workload contract:

* :func:`materialize_config_with_envs` — write a per-run YAML honoring process
  env (+ optional caller overrides).
* :func:`default_baseline_config` — pick the shipped YAML by ``$FRAMEWORK``.

Used by ``baseline.py`` (materializes once, surfaces the path) and the
``explore`` / ``sweep`` grid runs (fall back to materializing on their own).
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
from ._grid_runner import (
    inject_sglang_attention_backend,
    inject_sglang_context_length,
    inject_sglang_watchdog_timeout,
    server_args_env_name,
)
from ._server_patcher import (
    ensure_sglang_patched_for_tracelens,
    ensure_vllm_patched_for_tracelens,
)

log = logging.getLogger(__name__)

# Emit the RUN_EVAL=false default warning once per process to keep logs
# readable.
_RUN_EVAL_DEFAULT_WARN_EMITTED = False


def _visible_gpu_count() -> int:
    """Return how many GPUs are visible to this pod (0 = none / unknown).

    Prefers ``torch.cuda.device_count`` (no shell-out, keeps subprocess-mock
    tests happy), falls back to ``rocm-smi --showid``. Returns 0 on every
    failure so callers skip the clamp. Override via
    ``$INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT``.
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
            # ``rocm-smi --showid`` emits multiple ``GPU[N]`` lines per device;
            # dedup by index so 4 GPUs return 4, not 24.
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

    Set ``HYPERLOOM_ENABLE_PATCH=0`` to disable runtime patching of vLLM /
    SGLang (keeps today's safe behaviour, no TraceLens-only flags injected).
    Default on because the patches are backward-compatible.
    """
    return os.environ.get("HYPERLOOM_ENABLE_PATCH", "1").strip() != "0"


def default_baseline_config() -> Path:
    """Resolve the shipped Magpie YAML based on ``$FRAMEWORK`` env.

    Returns the sglang YAML when ``$FRAMEWORK`` is unset/unknown so existing
    sglang-default tests keep passing.

    Returns:
        Path: The shipped Magpie YAML config path for the resolved framework.
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

    Process env wins over YAML defaults: ``MODEL_PATH`` → ``benchmark.model``;
    ``GPU_TYPE`` → ``runner_type`` + pinned generic ``{framework}_{gpu_type}.sh``
    (so Magpie doesn't fall through to a native script hardcoding
    ``--result-dir /workspace/``); ``benchmark_script`` (pre-sanitized) re-pins
    after that; ``PRECISION`` → ``precision``; ``CONC/ISL/OSL/MAX_MODEL_LEN/TP/
    RANDOM_RANGE_RATIO`` → ``benchmark.envs``; ``ROCR_VISIBLE_DEVICES``
    reconciled against TP; ``RUN_EVAL`` defaulted; ``NUM_PROMPTS`` /
    ``NUM_WARMUPS`` computed adaptively. ``extra_server_args`` routes into the
    framework env; ``extra_envs`` overrides any of the above.

    Returns the materialized YAML path (stable file name across calls).
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
        # Persist $INFERENCEX_PATH into the YAML so Magpie's runtime checkout
        # matches Hyperloom's patch target.
        bench["inferencex_path"] = inferencex_path
    envs = bench.setdefault("envs", {})
    for env_key in (
        "CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP",
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = int(val)
    # RANDOM_RANGE_RATIO is a float feeding the steady-state formulas below; do
    # NOT coerce to int or fractional ratios collapse the prefill estimate.
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
        # Derive TP from the user-pinned GPU list when the YAML doesn't set TP.
        resolved_tp = len(rocr_devices)
        envs["TP"] = resolved_tp
    else:
        resolved_tp = int(tp_from_yaml or 1)
    # Auto-clamp TP to the visible GPU count so a 4-GPU sandbox doesn't launch
    # with the YAML's TP=8 and crash on `invalid device ordinal`. Override via
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

    # Steady-state window for profiling configs (detected by PROFILE env or
    # ``profiler.torch_profiler.enabled``). Formulas match the TraceLens
    # profiling skill (RANDOM_RANGE_RATIO defaults to 1.0):
    #   max_iters   = min(1024, max(256, OSL * 16 / CONC))
    #   delay_iters = OSL * (R + 1) * 3 - max_iters / 2
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
        # Clamp >= 0 (tiny OSL / huge R can produce a negative delay).
        if delay_iters < 0:
            delay_iters = 0
        # Operator override for a small eager FlyDSL profile: the default
        # >=256-step capture is unsavable in eager mode for a 30B MoE, but
        # eager is the only mode recording the flydsl_moe frames PR#668 keys
        # on. Set HYPERLOOM_PROFILE_MAX_ITERS small (e.g. 8) with the eager
        # profile patch.
        _ovr = os.environ.get("HYPERLOOM_PROFILE_MAX_ITERS", "").strip()
        if _ovr.isdigit() and int(_ovr) > 0:
            max_iters = int(_ovr)
            try:
                delay_iters = int(
                    os.environ.get("HYPERLOOM_PROFILE_DELAY_ITERS", "8") or "8"
                )
            except (TypeError, ValueError):
                delay_iters = 8
            if delay_iters < 0:
                delay_iters = 0
        # TraceLens #194 §2: NUM_PROMPTS must let the engine reach
        # ``delay_iters + max_iters`` decode steps before running out of
        # prompts (N prompts ≈ N * OSL / CONC iters; invert + 2x buffer).
        # Hyperloom owns this under PROFILE; a caller value is ignored.
        required_iters = delay_iters + max_iters
        iters_to_prompts = max(
            1, (required_iters * safe_conc + safe_osl - 1) // safe_osl,
        )
        profile_num_prompts = max(safe_conc, iters_to_prompts * 2)
        fw = str(bench.get("framework") or "").lower()
        # atom checked first so a future overlapping name can't fall into the
        # wrong branch. atom's profiler is HTTP-driven via atom_mi*x.sh, so
        # this layer sets no atom profiler envs and must NOT inject
        # --profiler-config.* flags (atom argparse rejects them).
        is_atom = "atom" in fw
        # #194 §4/§5: TraceLens profiler flags exist only in patched vLLM /
        # SGLang builds; try to patch, fall back to the safe set on failure.
        # Default-on (HYPERLOOM_ENABLE_PATCH=0 disables); skip for atom (native
        # profiler, no patch set).
        tracelens_patch_ok = False
        if _tracelens_patch_enabled() and not is_atom:
            if "vllm" in fw:
                tracelens_patch_ok = ensure_vllm_patched_for_tracelens()
            else:
                tracelens_patch_ok = ensure_sglang_patched_for_tracelens()
        if is_atom:
            # atom's profiler records the entire bench-client run (no internal
            # window); its profile window lives only in Magpie's atom_mi*x.sh
            # (ATOM_PROFILE_OSL / ATOM_PROFILE_NUM_PROMPTS). Forcing the
            # sglang/vllm steady-state NUM_PROMPTS here would starve aiter's
            # broadcast ring and abort before /stop_profile flushes. Defer to
            # Magpie.
            profile_num_prompts = None
        elif "vllm" in fw:
            existing_vllm_args = str(envs.get("EXTRA_VLLM_ARGS", ""))
            profiler_args_parts = [
                f"--profiler-config.delay_iterations {delay_iters}",
                f"--profiler-config.max_iterations {max_iters}",
            ]
            if tracelens_patch_ok:
                # #194 §4: TraceLens-patched vLLM exposes
                # capture_torch_profiler_dir + detailed_trace_annotation;
                # unpatched vLLM rejects them, so gate on the patcher result.
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
            # Always override start_step/num_steps (the template has CONC=8
            # placeholders).
            extra_body["start_step"] = delay_iters
            extra_body["num_steps"] = max_iters
            # shape_discovery balloons an eager+with_stack trace; the splitter's
            # per-step annotations come from roofline_annotations independently,
            # so allow disabling shape_discovery via env for eager profiles.
            _shape_disc = os.environ.get(
                "HYPERLOOM_PROFILE_SHAPE_DISCOVERY", "1",
            ).strip().lower() not in {"0", "false", "no", "off"}
            extra_body["shape_discovery"] = _shape_disc
            extra_body.setdefault("roofline_annotations", True)
            envs["PROFILE_EXTRA_BODY"] = _json.dumps(extra_body)
            if tracelens_patch_ok and _shape_disc:
                # #194 §5: TraceLens-patched SGLang exposes
                # --enable-shape-discovery-for-cuda-graph-profile; unpatched
                # SGLang errors on it, so gate strictly on the patcher result.
                existing_sglang = str(envs.get("EXTRA_SGLANG_ARGS", ""))
                if "shape-discovery-for-cuda-graph-profile" not in existing_sglang:
                    envs["EXTRA_SGLANG_ARGS"] = (
                        f"{existing_sglang} "
                        "--enable-shape-discovery-for-cuda-graph-profile"
                    ).strip()

    if profile_num_prompts is not None:
        # Profile mode: force-override NUM_PROMPTS to reach the steady-state
        # window (an under-sized value empties the trace).
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
        # Merge into (not overwrite) the framework env so the profile path's
        # upstream-injected graph-capture flags aren't dropped when the caller
        # also supplies extra args.
        from ._grid_runner import merge_server_args
        framework_env = server_args_env_name(bench.get("framework"))
        existing = str(envs.get(framework_env, "")).strip()
        if existing:
            envs[framework_env] = merge_server_args(existing, server_args)
        else:
            envs[framework_env] = server_args
    for key, value in (extra_envs or {}).items():
        envs[str(key)] = str(value)
    # ── Per-model MI300X baseline work-arounds ─────────────────────────
    # A handful of flagship models SIGABRT during CUDA-graph capture on the
    # sglang ROCm image because their DEFAULT fused kernels are buggy on
    # gfx942. Inject the verified per-model work-around UNLESS the caller
    # already pinned it (explore variants may legitimately re-try the fused
    # path once the agent knows the model loads — hence setdefault/merge,
    # never overwrite). Matched on the model basename so it fires for both
    # the HF repo id and the /wekafs/models/<org>-<repo> local path.
    _model_basename = Path(
        str(model_path or os.environ.get("MODEL_PATH", ""))
    ).name.lower()
    if "kimi-k2" in _model_basename:
        # Kimi K2.x at tp8 (8 heads/GPU) takes sglang's ROCm
        # fused-decode-MLA path, whose RoPE kernel aborts during CUDA-graph
        # capture (forward_mla_fused_rope_rocm.py: "cannot unpack
        # non-iterable ForwardMetadata"). Disabling the fused decode
        # pipeline keeps the configured tp8 + the clean aiter MLA path.
        # Verified on MI300X: capture passes, decode correct.
        envs.setdefault("SGLANG_ROCM_FUSED_DECODE_MLA", "0")
    if "mimo-v2" in _model_basename:
        # MiMo-V2.x (moe_swa) loads MiMoV2ForCausalLM fine but its DEFAULT
        # aiter attention backend SIGABRTs during CUDA-graph capture on
        # gfx942 (mimo_v2.py forward -> GPU coredump -> "Rank N scheduler
        # died during initialization (exit code: -6)"). Pin the triton
        # attention backend, which sidesteps the buggy aiter fused-attention
        # path. Pairs with the mimo-profilerfix image (the undated v0.5.11
        # profilerfix base does not register MiMoV2ForCausalLM at all; the
        # image picked in optimize_submit._sglang_image_for carries the
        # dated 20260508 arch). Merge (never overwrite) and skip when the
        # caller already pinned an --attention-backend so explore variants
        # can re-test the fused path once the model is known to load.
        from ._grid_runner import merge_server_args
        _mimo_fw_env = server_args_env_name(bench.get("framework"))
        _mimo_existing = str(envs.get(_mimo_fw_env, "")).strip()
        if "attention-backend" not in _mimo_existing:
            envs[_mimo_fw_env] = (
                merge_server_args(_mimo_existing, "--attention-backend triton")
                if _mimo_existing
                else "--attention-backend triton"
            )
    # sglang server-arg guards, applied at the FINAL framework env (after the
    # server_args + extra_envs merges above) so any operator-pinned flag (via
    # extra_server_args, extra_envs, or the YAML) is honored and never doubled.
    # Both are no-ops for vllm/atom. This is the single choke point every
    # benchmark path (baseline / profile / sweep / explore / framework_pr /
    # conc_sweep) funnels through before the YAML is handed to Magpie, so the
    # flags reach every sglang launch.
    #
    # 1. --context-length cap: sglang defaults context_length=None and sizes
    #    max_total_tokens off the model's max_position_embeddings; a huge
    #    native window (e.g. Mistral-Nemo's 1024000) balloons the aiter
    #    workspace_buffer past GPU memory -> HIP OOM -> baseline_failed. We cap
    #    to ISL+OSL+headroom (floored, clamped to the native window). vllm
    #    already passes --max-model-len $MAX_MODEL_LEN, so this only fixes the
    #    sglang asymmetry; sglang ignores MAX_MODEL_LEN entirely.
    # 2. MI300X cold-compile guard: ensure sglang's scheduler watchdog is long
    #    enough to survive the first-request aiter ``mha_batch_prefill`` JIT
    #    compile. sglang's 300s default fires SIGQUIT mid-warmup on a cold
    #    aiter cache and the server dies -> baseline_failed / throughput 0.
    framework_env = server_args_env_name(bench.get("framework"))
    resolved_server_args = str(envs.get(framework_env, "")).strip()
    resolved_server_args = inject_sglang_context_length(
        resolved_server_args, bench.get("framework"),
        bench.get("model"), isl_val, osl_val,
    )
    resolved_server_args = inject_sglang_watchdog_timeout(
        resolved_server_args, bench.get("framework"),
    )
    # 3. Dual-chunk attention backend: Qwen 1M models declare
    #    dual_chunk_attention_config; sglang rejects the default aiter
    #    backend for them and demands dual_chunk_flash_attn. Inject it
    #    unless the operator already pinned --attention-backend.
    resolved_server_args = inject_sglang_attention_backend(
        resolved_server_args, bench.get("framework"), bench.get("model"),
        gpu_type=gpu_type or bench.get("runner_type"),
    )
    if resolved_server_args:
        envs[framework_env] = resolved_server_args
    # Accuracy eval (GSM8K) is OFF by default: Magpie's scripts pass
    # `run_eval --concurrent-requests N` but InferenceX's `run_lm_eval`
    # rejects that flag (fails the whole benchmark). Until upstream realigns,
    # the user opts in via env / extra_envs; the accuracy gate treats a missing
    # result as "no regression". Resolved after extra_envs merging so an
    # explicit RUN_EVAL=true doesn't trigger the warning.
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
