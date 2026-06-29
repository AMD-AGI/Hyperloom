# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared workload-env materialization (single source of truth).

This module is the single source of truth for rendering a Magpie YAML with the
user's actual process-env workload contract, so the baseline and grid-runner
paths render identical YAML:

* :func:`materialize_config_with_envs` — write a per-run YAML honoring process
  env (+ optional caller overrides).
* :func:`default_baseline_config` — pick the shipped YAML by ``$FRAMEWORK``.

Used by ``baseline.py`` (materializes once, surfaces the path) and the
``explore`` / ``sweep`` grid runs (fall back to materializing on their own).
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ...paths import asset_root
from ._grid_runner import (
    compact_json_server_args,
    dedup_vllm_server_args,
    inject_sglang_attention_backend,
    inject_sglang_context_length,
    inject_sglang_moe_runner_backend,
    inject_sglang_watchdog_timeout,
    server_args_env_name,
)
from ._server_patcher import (
    ensure_sglang_patched_for_ck_blockscale,
    ensure_sglang_patched_for_tracelens,
    ensure_vllm_patched_for_tracelens,
)
from ...model_config_utils import (
    _fp8_is_per_channel_per_token,
    _load_model_config_dict,
    _model_is_gemma2,
)

log = logging.getLogger(__name__)

# gfx942 / CDNA3 dies (MI300X and its MI308X/MI325X siblings) that ship the
# aiter CK gemm_a8w8_bpreshuffle kernel. MI355X is gfx950 and excluded.
_GFX942_GPU_TYPES = frozenset({"mi300x", "mi308x", "mi325x"})

_MOE_RUNNER_BACKEND_RE = re.compile(r"(?:^|\s)--moe-runner-backend(?:[=\s]+)\S+")

# Profile-phase capture defaults (issue #571 / #570). Trace size scales with
# captured decode steps; an oversized capture serializes too slowly and kills
# the engine (EngineCore RPC timeout). 128 captured steps was measured
# serialization-safe (~160 MB/rank, ~29s) on a large TP=8 MoE, so smaller
# models stay within budget. Tunable via HYPERLOOM_PROFILE_MAX_STEPS_CAP.
_DEFAULT_PROFILE_MAX_STEPS = 128
# Default profile OSL ceiling when --profile-osl / PROFILE_OSL is unset: the
# profile reuses min(served OSL, this) so its trace stays light without
# distorting the served workload more than necessary.
_PROFILE_DEFAULT_OSL = 1024


def _remove_moe_runner_backend_arg(args: str) -> str:
    """Remove any existing SGLang MoE runner backend flag from an args string."""
    return " ".join(_MOE_RUNNER_BACKEND_RE.sub(" ", str(args or "")).split())

# Warn once per process when the accuracy gate is disabled.
_RUN_EVAL_DISABLED_WARN_EMITTED = False

# Truthy-false spellings that disable the accuracy gate.
_RUN_EVAL_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


def _model_requires_remote_code(model_path: str | None) -> bool:
    """Return whether benchmark server/client must trust custom HF code.

    Kimi K2.6 exposes its tokenizer through custom model code
    (``model_type=kimi_k25`` / ``KimiK25ForConditionalGeneration``). The
    benchmark client and the server must both pass trust-remote-code or baseline
    fails before producing a usable measurement. Generalize the guard to any
    local config that advertises a custom AutoTokenizer so future custom-code
    tokenizers do not need per-model special cases.
    """
    model = str(model_path or "").strip()
    if not model:
        return False
    data = _load_model_config_dict(model)
    basename = Path(model).name.lower()
    if data is None:
        # Fallback for mounted model dirs whose config is temporarily
        # unreadable. Keep this narrow so ordinary models are untouched.
        return "kimi-k2" in basename or "kimi_k2" in basename
    model_type = str(data.get("model_type") or "").lower()
    archs = {str(a).lower() for a in data.get("architectures") or []}
    if model_type == "kimi_k25" or "kimik25forconditionalgeneration" in archs:
        return True
    auto_map = data.get("auto_map")
    return isinstance(auto_map, dict) and bool(auto_map.get("AutoTokenizer"))


def inject_vllm_expert_parallel(
    server_args: str | None,
    framework: Any,
    ep: Any,
) -> str:
    """Append vLLM expert-parallel flag when EP is enabled."""
    args = str(server_args or "").strip()
    if "vllm" not in str(framework or "").lower():
        return args
    try:
        ep_int = int(ep if ep not in (None, "") else 1)
    except (TypeError, ValueError):
        return args
    if ep_int <= 1:
        return args
    if re.search(r"(?:^|\s)--enable-expert-parallel(?:\s|$)", args):
        return args
    return f"{args} --enable-expert-parallel".strip()


class FrameworkScriptMismatchError(ValueError):
    """Raised when benchmark_script targets a different framework than the run.

    Subclasses ValueError so callers can catch it specifically and turn it
    into a structured action failure instead of an uncaught exception.
    """


def _visible_gpu_count() -> int:
    """Return how many GPUs are visible to this pod (0 = none / unknown).

    Prefers ``torch.cuda.device_count`` (no shell-out, keeps subprocess-mock
    tests happy), falls back to ``rocm-smi --showid``. Returns 0 on every
    failure so callers skip the clamp. Override via
    ``$INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT``.

    Returns:
        The number of visible GPUs, or 0 when none/unknown.
    """
    override = os.environ.get("INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT", "").strip()
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            log.warning(
                "INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT=%r is not an int; ignoring override.",
                override,
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
                capture_output=True,
                text=True,
                timeout=5,
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

    Returns:
        True when runtime patching is enabled (default), else False.
    """
    return os.environ.get("HYPERLOOM_ENABLE_PATCH", "1").strip() != "0"


def _coerce_workload_int_env(env_key: str, raw: str) -> int:
    """Coerce workload env values, accepting ``CONC`` comma ladders."""
    text = str(raw or "").strip()
    if env_key == "CONC" and "," in text:
        values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
        if not values or any(v <= 0 for v in values):
            raise ValueError(f"{env_key}={raw!r} must contain positive integers")
        os.environ.setdefault(
            "INFERENCE_OPTIMIZER_CONC_SWEEP_CONCS",
            ",".join(str(v) for v in values),
        )
        return values[0]
    value = int(text)
    if value <= 0:
        raise ValueError(f"{env_key}={raw!r} must be positive")
    return value


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
    elif fw == "xdit":
        name = "baseline_xdit.yaml"
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
    inferencex_path: str | None = None,
    benchmark_script: str | None = None,
    reference_server_args: str = "",
    reference_envs: dict[str, Any] | None = None,
    out_name: str = "baseline_config.with_envs.yaml",
    establish_quality_ref: bool = False,
) -> Path:
    """Render a per-run Magpie YAML with caller-provided overrides.

    Process env wins over YAML defaults: ``MODEL_PATH`` → ``benchmark.model``;
    ``GPU_TYPE`` → ``runner_type`` + pinned generic ``{framework}_{gpu_type}.sh``
    (so Magpie doesn't fall through to a native script hardcoding
    ``--result-dir /workspace/``); ``benchmark_script`` (pre-sanitized) re-pins
    after that; ``PRECISION`` → ``precision``; ``CONC/ISL/OSL/MAX_MODEL_LEN/TP/
    RANDOM_RANGE_RATIO`` → ``benchmark.envs``; ``ROCR_VISIBLE_DEVICES``
    reconciled against TP; ``RUN_EVAL`` defaulted; ``NUM_PROMPTS`` /
    ``NUM_WARMUPS`` computed adaptively. ``inferencex_path`` explicitly pins
    ``benchmark.inferencex_path`` for one task (falling back to
    ``$INFERENCEX_PATH`` for existing callers). ``extra_server_args`` routes
    into the framework env; ``extra_envs`` overrides any of the above.
    ``reference_server_args`` / ``reference_envs`` seed a lowest-priority base
    from a reference recipe (below the YAML base and extra_server_args; empty =
    no-op, byte-for-byte identical to omitting them).

    Args:
        config_path: Path to the source Magpie YAML to render.
        output_dir: Directory the materialized YAML is written into.
        extra_server_args: Extra framework server args merged into the env.
        extra_envs: Overrides applied last over any computed env values.
        model_path: Model path/id; overrides ``benchmark.model`` when set.
        gpu_type: GPU type; sets ``runner_type`` and pins the generic script.
        inferencex_path: Explicit InferenceX checkout to pin into the YAML.
        benchmark_script: Pre-sanitized benchmark script name to re-pin.
        out_name: File name for the materialized YAML.
        establish_quality_ref: When True (baseline only) the scriptable
            image-quality reference is ESTABLISHED (written) by this run;
            otherwise the run only COMPARES against it. See the quality-
            reference wiring block below.

    Returns:
        The materialized YAML path (stable file name across calls).

    Raises:
        FrameworkScriptMismatchError: If ``benchmark_script`` targets a
            different known framework than the run's framework.
    """
    server_args = (extra_server_args or "").strip()
    operator_server_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
    if operator_server_args:
        if server_args:
            from ._grid_runner import merge_server_args

            server_args = merge_server_args(operator_server_args, server_args)
        else:
            server_args = operator_server_args
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
    # Fail fast on framework/script mismatch (e.g. vllm image + sglang script):
    # guards the QRWKV-72B bug where $FRAMEWORK fell back to sglang and booted
    # `sglang.launch_server` in a vllm-only image -> ModuleNotFoundError. Only
    # trip when the script carries a DIFFERENT known framework's prefix, so
    # custom/non-prefixed scripts are not falsely rejected.
    _script = str(bench.get("benchmark_script") or "").lower()
    _fw = str(bench.get("framework") or "").lower()
    from ... import framework_registry

    _known_fw = framework_registry.names()
    if _script and _fw in _known_fw:
        _other = [k for k in _known_fw if k != _fw and _script.startswith(f"{k}_")]
        if _other:
            raise FrameworkScriptMismatchError(
                f"framework/script mismatch: framework={_fw!r} but "
                f"benchmark_script={_script!r} targets {_other[0]!r}; refusing "
                f"to boot server (would launch the wrong framework's entrypoint)"
            )
    effective_inferencex_path = str(inferencex_path or "").strip() or os.environ.get("INFERENCEX_PATH", "").strip()
    if effective_inferencex_path:
        # Persist the resolved InferenceX checkout into the YAML so Magpie's
        # runtime checkout matches Hyperloom's patch target. Baseline/Profile
        # pass the per-task local mirror explicitly to avoid process-wide env
        # races; legacy callers still fall back to $INFERENCEX_PATH.
        bench["inferencex_path"] = effective_inferencex_path
    envs = bench.setdefault("envs", {})
    for env_key in (
        "CONC",
        "ISL",
        "OSL",
        "MAX_MODEL_LEN",
        "TP",
    ):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = _coerce_workload_int_env(env_key, val)
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
                resolved_tp,
                visible,
                visible,
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
                rocr_yaml,
                len(rocr_devices),
                resolved_tp,
                derived,
            )
        envs["ROCR_VISIBLE_DEVICES"] = derived

    isl_val = int(envs.get("ISL") or 256)
    osl_val = int(envs.get("OSL") or 256)
    conc_val = int(envs.get("CONC") or 8)

    # Steady-state window for profiling configs (detected by PROFILE env or
    # ``profiler.torch_profiler.enabled``). The captured-step count is capped at
    # a serialization-safe budget; the profile OSL is resolved (and lowered if
    # needed) so the steady-state floor fits that cap (RANDOM_RANGE_RATIO
    # defaults to 1.0):
    #   max_iters    = HYPERLOOM_PROFILE_MAX_STEPS_CAP (default 128)
    #   steady_floor = ceil(OSL * (1 + R) / (2 * CONC))   # must be <= max_iters
    #   delay_iters  = OSL * (R + 1) * 3 - max_iters / 2
    is_profile = str(envs.get("PROFILE", "")).strip() == "1" or (
        bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True
    )
    profile_num_prompts: int | None = None
    if is_profile:
        try:
            r_val = float(envs.get("RANDOM_RANGE_RATIO", 1.0))
        except (TypeError, ValueError):
            r_val = 1.0
        safe_conc = max(conc_val, 1)
        # --- Profile capture window cap (issue #571 / #570) ----------------
        # Cap captured decode steps at a serialization-safe default so the
        # torch-profiler trace can be written without starving the engine RPC
        # (the EngineCore timeout crash). Operator-tunable.
        try:
            cap = int(
                os.environ.get("HYPERLOOM_PROFILE_MAX_STEPS_CAP", "").strip()
                or _DEFAULT_PROFILE_MAX_STEPS
            )
        except (TypeError, ValueError):
            cap = _DEFAULT_PROFILE_MAX_STEPS
        if cap < 1:
            cap = _DEFAULT_PROFILE_MAX_STEPS

        # --- Resolve the profile-scoped OSL --------------------------------
        # The profile/roofline phase may run a lighter OSL than the served
        # workload so its trace stays serializable; baseline/optimize keep the
        # global OSL. PROFILE_OSL (via --profile-osl) is an explicit operator
        # choice and is honored as-is; otherwise default to
        # min(served OSL, _PROFILE_DEFAULT_OSL). Scoped to is_profile so
        # baseline/optimize configs are never affected.
        _profile_osl_raw = os.environ.get("PROFILE_OSL", "").strip()
        profile_osl_explicit = _profile_osl_raw.isdigit() and int(_profile_osl_raw) > 0
        if profile_osl_explicit:
            osl_val = int(_profile_osl_raw)
        else:
            osl_val = min(osl_val, _PROFILE_DEFAULT_OSL)
        safe_osl = max(osl_val, 1)

        # Steady-state floor: minimum captured decode steps for the splitter to
        # isolate a steady-state window (mirrors TraceLens
        # find_steady_state_window). The capture must be >= this or the splitter
        # reports trace_split_no_steady_state.
        steady_floor = math.ceil(safe_osl * (1.0 + r_val) / (2.0 * safe_conc))
        if steady_floor > cap:
            if profile_osl_explicit:
                # Honor the operator's explicit OSL; warn that the window may
                # not contain a steady-state segment at this OSL + cap.
                log.warning(
                    "PROFILE_OSL=%d needs %d captured steps to reach steady "
                    "state, above the profile cap of %d; the trace may lack a "
                    "steady-state window (trace_split_no_steady_state). Lower "
                    "--profile-osl or raise HYPERLOOM_PROFILE_MAX_STEPS_CAP.",
                    osl_val, steady_floor, cap,
                )
            else:
                # Auto path: lower the profile OSL so the floor fits the cap
                # (largest OSL whose steady floor stays <= cap).
                fitted_osl = max(1, int(cap * 2 * safe_conc / (1.0 + r_val)))
                log.warning(
                    "profile OSL %d would need %d captured steps to reach "
                    "steady state (> cap %d); lowering profile OSL to %d so the "
                    "capture stays serializable. Baseline/optimize unaffected.",
                    osl_val, steady_floor, cap, fitted_osl,
                )
                osl_val = fitted_osl
                safe_osl = max(osl_val, 1)
                steady_floor = math.ceil(safe_osl * (1.0 + r_val) / (2.0 * safe_conc))

        # Profile server runs at the resolved (possibly reduced) profile OSL,
        # decoupled from the served --osl.
        envs["OSL"] = osl_val

        # Capture up to the cap (>= steady_floor in the auto path). delay_iters
        # keeps the established warmup formula.
        max_iters = cap
        delay_iters = int(osl_val * (r_val + 1) * 3 - max_iters / 2)
        # Clamp >= 0 (tiny OSL / huge R can produce a negative delay).
        if delay_iters < 0:
            delay_iters = 0
        # Operator hard-override of captured steps for a small eager FlyDSL
        # profile (e.g. 8), which is unsavable in eager mode at the default
        # capture but is the only mode recording the flydsl_moe frames PR#668
        # keys on. Honored verbatim; warn when outside the safe band rather
        # than silently clamping.
        _ovr = os.environ.get("HYPERLOOM_PROFILE_MAX_ITERS", "").strip()
        if _ovr.isdigit() and int(_ovr) > 0:
            max_iters = int(_ovr)
            try:
                delay_iters = int(os.environ.get("HYPERLOOM_PROFILE_DELAY_ITERS", "8") or "8")
            except (TypeError, ValueError):
                delay_iters = 8
            if delay_iters < 0:
                delay_iters = 0
            if max_iters < steady_floor:
                log.warning(
                    "HYPERLOOM_PROFILE_MAX_ITERS=%d is below the steady-state "
                    "floor of %d; the trace may lack a steady-state window "
                    "(trace_split_no_steady_state).",
                    max_iters, steady_floor,
                )
            elif max_iters > cap:
                log.warning(
                    "HYPERLOOM_PROFILE_MAX_ITERS=%d exceeds the serialization-"
                    "safe cap of %d; the trace may be too large to serialize "
                    "(EngineCore RPC timeout).",
                    max_iters, cap,
                )
        # NUM_PROMPTS must let the engine reach
        # ``delay_iters + max_iters`` decode steps before running out of
        # prompts (N prompts ≈ N * OSL / CONC iters; invert + 2x buffer).
        # Hyperloom owns this under PROFILE; a caller value is ignored.
        required_iters = delay_iters + max_iters
        iters_to_prompts = max(
            1,
            (required_iters * safe_conc + safe_osl - 1) // safe_osl,
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
        patch_attempted = _tracelens_patch_enabled() and not is_atom
        if patch_attempted:
            if "vllm" in fw:
                tracelens_patch_ok = ensure_vllm_patched_for_tracelens()
            else:
                tracelens_patch_ok = ensure_sglang_patched_for_tracelens()
            if not tracelens_patch_ok:
                envs["HYPERLOOM_TRACELENS_PATCH_STATUS"] = "unavailable"
                envs["HYPERLOOM_PROFILE_DEGRADED_REASON"] = (
                    "tracelens_runtime_patch_unavailable"
                )
                log.warning(
                    "TraceLens runtime patch unavailable for framework=%s; "
                    "profile will omit annotation-only flags and roofline "
                    "analysis may be degraded.",
                    fw or "<unset>",
                )
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
                profiler_args_parts.append(f"--profiler-config.capture_torch_profiler_dir {capture_dir}")
                profiler_args_parts.append("--profiler-config.detailed_trace_annotation True")
            profiler_args = " ".join(profiler_args_parts)
            if "delay_iterations" not in existing_vllm_args:
                envs["EXTRA_VLLM_ARGS"] = f"{existing_vllm_args} {profiler_args}".strip()
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
                "HYPERLOOM_PROFILE_SHAPE_DISCOVERY",
                "1",
            ).strip().lower() not in {"0", "false", "no", "off"}
            # Gemma2 + shape-discovery crashes CUDA-graph capture (host
            # torch.tensor in forward during HIP stream capture). Disable
            # shape-discovery for Gemma2 so capture/roofline still run.
            # Escape hatch HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 only skips
            # the Gemma2 gate (for debugging the TraceLens root-cause fix); it
            # does NOT override a global HYPERLOOM_PROFILE_SHAPE_DISCOVERY=0.
            _force_shape_disc = os.environ.get(
                "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE",
                "0",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if _shape_disc and not _force_shape_disc:
                _model = str(bench.get("model") or "")
                if _model_is_gemma2(_model):
                    _shape_disc = False
                    log.info(
                        "Gemma2 roofline: disabling shape-discovery to avoid "
                        "CUDA-graph capture crash (hipErrorStreamCapture"
                        "Unsupported); CUDA graph + profiling kept. Set "
                        "HYPERLOOM_PROFILE_SHAPE_DISCOVERY_FORCE=1 to override.",
                    )
                    if _load_model_config_dict(_model) is None:
                        log.warning(
                            "Gemma2 detected via path heuristic (no readable "
                            "config.json at %r); shape-discovery skip may be "
                            "imprecise.",
                            _model,
                        )
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
                        f"{existing_sglang} --enable-shape-discovery-for-cuda-graph-profile"
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
    # ── reference-script base (lowest priority) ────────────────────────────
    # Seed the framework server-args env + envs from a reference recipe BELOW
    # the YAML base and any per-task extra_server_args. Reference flags are
    # leftmost so merge_server_args' last-wins lets the YAML / extra args / the
    # per-model workarounds below override them; the final dedup collapses dups.
    ref_args = (reference_server_args or "").strip()
    if ref_args:
        from ._grid_runner import merge_server_args
        _ref_fw_env = server_args_env_name(bench.get("framework"))
        _ref_existing = str(envs.get(_ref_fw_env, "")).strip()
        envs[_ref_fw_env] = (
            merge_server_args(ref_args, _ref_existing) if _ref_existing else ref_args
        )
    for _rk, _rv in (reference_envs or {}).items():
        envs.setdefault(str(_rk), str(_rv))  # never clobber YAML/CLI envs
    if server_args:
        # Merge into (not overwrite) the framework env so the profile path's
        # upstream-injected graph-capture flags aren't dropped when the caller
        # also supplies extra args.
        from ._grid_runner import merge_server_args

        framework_env = server_args_env_name(bench.get("framework"))
        existing = str(envs.get(framework_env, "")).strip()
        if framework_env == "EXTRA_SGLANG_ARGS" and "--moe-runner-backend" in str(server_args):
            # For MoE backend exploration/tuning, the candidate value must
            # replace the baseline's injected default (usually triton) rather
            # than relying on duplicate last-wins flags.
            existing = _remove_moe_runner_backend_arg(existing)
        if existing:
            envs[framework_env] = merge_server_args(existing, server_args)
        else:
            envs[framework_env] = server_args
    for key, value in (extra_envs or {}).items():
        envs[str(key)] = str(value)
    # ── Quality-reference wiring (scriptable / server-less workloads) ──────
    # Magpie forwards ONLY ``benchmark.envs`` to the wrapper subprocess, so an
    # operator's ``XDIT_QUALITY_REF`` set in the process env never reaches it:
    # the shipped YAML default (``XDIT_QUALITY_REF: ""``) wins and the image-
    # quality gate is silently SKIPPED on every variant (fail-open). Re-inject
    # the reference here — the single choke point every scriptable bench path
    # funnels through — so the wrapper actually compares. Authoritative over the
    # YAML/caller because the empty YAML default is precisely the bug.
    #   * BASELINE (establish_quality_ref=True): force COMPARE off + WRITE the
    #     fresh reference, so a stale file from a previous session cannot make
    #     the baseline gate against the wrong truth.
    #   * Every other variant: COMPARE only and force the write path empty so a
    #     degraded variant can never overwrite the baseline reference and pass
    #     itself (benchmark.envs overrides the inherited process env).
    #   * Profiling / roofline (is_profile): no correctness gate AND must never
    #     write — an inherited write path would let a reduced-step profile image
    #     clobber the baseline reference.
    if framework_registry.is_scriptable(bench.get("framework")):
        _qref = os.environ.get("XDIT_QUALITY_REF", "").strip()
        if is_profile:
            envs["XDIT_QUALITY_REF"] = ""
            envs["XDIT_QUALITY_REF_WRITE"] = ""
        elif _qref:
            if establish_quality_ref:
                envs["XDIT_QUALITY_REF"] = ""
                envs["XDIT_QUALITY_REF_WRITE"] = (
                    os.environ.get("XDIT_QUALITY_REF_WRITE", "").strip() or _qref
                )
            else:
                envs["XDIT_QUALITY_REF"] = _qref
                envs["XDIT_QUALITY_REF_WRITE"] = ""
        # ── Model-arg wiring (scriptable xDiT registry resolution) ────────
        # The xDiT runner resolves models via MODEL_REGISTRY keys (e.g.
        # "Qwen-Image", "FLUX.1-dev"), NOT arbitrary filesystem paths. The
        # bench wrapper's XDIT_MODEL_ARG selects whether it passes the model
        # basename ("name", registry-correct) or the full path ("path", which
        # fails registry lookup -> "Model <path> not found in registry"). The
        # operator pins the correct mode in the process env; force it onto
        # benchmark.envs here (the single scriptable choke point) so per-task
        # agent overrides cannot silently break model resolution. Default to
        # "name" because registry lookup keys on the basename.
        envs["XDIT_MODEL_ARG"] = (
            os.environ.get("XDIT_MODEL_ARG", "").strip() or "name"
        )
        # ── Baseline attention-backend guard (scriptable xDiT) ────────────
        # The baseline must measure the clean, verified reference config. The
        # orchestration agent sometimes injects experimental extra_envs while
        # trying to escape a failure loop (e.g. XDIT_ATTENTION_BACKEND=torch,
        # which xDiT rejects: "Invalid attention backend: torch"). For the
        # baseline only, force the operator-pinned backend (default 'aiter',
        # the MI300X-verified path) so an invalid agent override cannot
        # poison the reference measurement. Explore/sweep variants keep their
        # freedom to try alternative backends.
        if establish_quality_ref:
            envs["XDIT_ATTENTION_BACKEND"] = (
                os.environ.get("XDIT_ATTENTION_BACKEND", "").strip() or "aiter"
            )
    # ── Per-model MI300X baseline work-arounds ─────────────────────────
    # A handful of flagship models SIGABRT during CUDA-graph capture on the
    # sglang ROCm image because their DEFAULT fused kernels are buggy on
    # gfx942. Inject the verified per-model work-around UNLESS the caller
    # already pinned it (explore variants may legitimately re-try the fused
    # path once the agent knows the model loads — hence setdefault/merge,
    # never overwrite). Matched on the model basename so it fires for both
    # the HF repo id and the /wekafs/models/<org>-<repo> local path.
    _model_basename = Path(str(model_path or os.environ.get("MODEL_PATH", ""))).name.lower()
    if "kimi-k2" in _model_basename:
        # Kimi K2.x at tp8 (8 heads/GPU) takes sglang's ROCm
        # fused-decode-MLA path, whose RoPE kernel aborts during CUDA-graph
        # capture (forward_mla_fused_rope_rocm.py: "cannot unpack
        # non-iterable ForwardMetadata"). Disabling the fused decode
        # pipeline keeps the configured tp8 + the clean aiter MLA path.
        # Verified on MI300X: capture passes, decode correct.
        envs.setdefault("SGLANG_ROCM_FUSED_DECODE_MLA", "0")
        # Client trust-remote-code is handled model-agnostically by the
        # "Client trust-remote-code" block after the server-arg guards below.
    if "mimo-v2" in _model_basename:
        # MiMo-V2.x (moe_swa) loads MiMoV2ForCausalLM fine but its DEFAULT
        # aiter attention backend SIGABRTs during CUDA-graph capture on
        # gfx942 (mimo_v2.py forward -> GPU coredump -> "Rank N scheduler
        # died during initialization (exit code: -6)"). Pin the triton
        # attention backend, which sidesteps the buggy aiter fused-attention
        # path. Pairs with the default sglang image picked in
        # optimize_submit._sglang_image_for (v0.5.12 profilerfix now registers
        # MiMoV2ForCausalLM, so the old v0.5.11 mimo-profilerfix override was
        # dropped). Merge (never overwrite) and skip when the
        # caller already pinned an --attention-backend so explore variants
        # can re-test the fused path once the model is known to load.
        from ._grid_runner import merge_server_args

        _mimo_fw_env = server_args_env_name(bench.get("framework"))
        _mimo_existing = str(envs.get(_mimo_fw_env, "")).strip()
        _mimo_is_vllm = "vllm" in str(bench.get("framework") or "").lower()
        # sglang accepts the lowercase backend name `triton`; vLLM's
        # AttentionBackendEnum (config/attention.py validate_backend_before ->
        # value.upper()) only knows TRITON_ATTN — plain `triton`/`TRITON` is
        # rejected as "Unknown attention backend" and the server never boots
        # -> baseline_failed. Pick the framework-correct spelling.
        _mimo_attn_backend = "TRITON_ATTN" if _mimo_is_vllm else "triton"
        if "attention-backend" not in _mimo_existing:
            envs[_mimo_fw_env] = (
                merge_server_args(_mimo_existing, f"--attention-backend {_mimo_attn_backend}")
                if _mimo_existing
                else f"--attention-backend {_mimo_attn_backend}"
            )
        # vLLM registers this checkpoint's implementation under the arch name
        # MiMoV2FlashForCausalLM (model_executor/models/mimo_v2_flash.py), but
        # the read-only HF config declares architectures=["MiMoV2ForCausalLM"],
        # which the pod-local vLLM build does not recognize -> ModelConfig
        # ValidationError "architectures ['MiMoV2ForCausalLM'] are not supported"
        # at server boot -> baseline_failed (server never comes up). Remap the
        # arch name via --hf-overrides so every `vllm serve` accepts the
        # checkpoint untouched. vLLM-only: the sglang/RayJob path uses the dated
        # image that registers the arch natively. The JSON is kept space-free so
        # it survives Magpie's unquoted `vllm serve ... $EXTRA_VLLM_ARGS` splice
        # (a single shell word). Merge (never overwrite) and skip when the
        # caller already pinned an --hf-overrides so explore variants can
        # re-test alternative overrides.
        if "vllm" in str(bench.get("framework") or "").lower():
            _mimo_hf_existing = str(envs.get(_mimo_fw_env, "")).strip()
            if "hf-overrides" not in _mimo_hf_existing and "hf_overrides" not in _mimo_hf_existing:
                _mimo_arch_override = '--hf-overrides {"architectures":["MiMoV2FlashForCausalLM"]}'
                envs[_mimo_fw_env] = (
                    merge_server_args(_mimo_hf_existing, _mimo_arch_override)
                    if _mimo_hf_existing
                    else _mimo_arch_override
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
    #    to ISL+OSL+headroom (floored, clamped to the native window AND to the
    #    run's MAX_MODEL_LEN). sglang sizes its window off --context-length, so
    #    when MAX_MODEL_LEN is below the workload cap we must clamp here or the
    #    injected --context-length would exceed the configured max-model-len
    #    (#697: --context-length 84048 > --max-model-len 82000).
    # 2. MI300X cold-compile guard: ensure sglang's scheduler watchdog is long
    #    enough to survive the first-request aiter ``mha_batch_prefill`` JIT
    #    compile. sglang's 300s default fires SIGQUIT mid-warmup on a cold
    #    aiter cache and the server dies -> baseline_failed / throughput 0.
    framework_env = server_args_env_name(bench.get("framework"))
    resolved_server_args = str(envs.get(framework_env, "")).strip()
    resolved_server_args = inject_sglang_context_length(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
        isl_val,
        osl_val,
        max_model_len=envs.get("MAX_MODEL_LEN") or os.environ.get("MAX_MODEL_LEN"),
    )
    resolved_server_args = inject_sglang_watchdog_timeout(
        resolved_server_args,
        bench.get("framework"),
    )
    # 3. Dual-chunk attention backend: Qwen 1M models declare
    #    dual_chunk_attention_config; sglang rejects the default aiter
    #    backend for them and demands dual_chunk_flash_attn. Inject it
    #    unless the operator already pinned --attention-backend.
    resolved_server_args = inject_sglang_attention_backend(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
        gpu_type=gpu_type or bench.get("runner_type"),
    )
    # 4. MoE runner backend: MoE models on ROCm route through aiter's CK
    #    2-stage fused-MoE kernel, whose first-request JIT build is broken in
    #    some images (missing cub header -> hipcc fail -> stale lock -> 600s
    #    warmup timeout). Inject the ROCm-capable triton MoE runner unless the
    #    operator already pinned --moe-runner-backend.
    resolved_server_args = inject_sglang_moe_runner_backend(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
        gpu_type=gpu_type or bench.get("runner_type"),
    )
    resolved_server_args = inject_vllm_expert_parallel(
        resolved_server_args,
        bench.get("framework"),
        os.environ.get("EP", "").strip() or envs.get("EP"),
    )
    # 5. vLLM/atom argparse dedup (#520): the YAML EXTRA_VLLM_ARGS base and a
    #    sweep/kernel variant can each inject --attention-backend, and
    #    merge_server_args keeps both. vLLM v0.21.0 crashes EngineCoreProc on a
    #    duplicate. Collapse repeated single-value flags to last-wins (so the
    #    variant override survives); no-op for sglang.
    resolved_server_args = dedup_vllm_server_args(
        resolved_server_args,
        bench.get("framework"),
    )
    # 6. JSON-valued flags (--speculative-config / --compilation-config /
    #    --hf-overrides ...): Magpie expands $EXTRA_VLLM_ARGS UNQUOTED, so a JSON
    #    value with the conventional separator spaces ('{"k": v}') is word-split
    #    by the shell and the server dies at boot. Compact each JSON blob to be
    #    space-free (string-internal spaces preserved) so it survives as one
    #    shell word — otherwise spec-decode / compilation-config explore variants
    #    can never be evaluated. No-op for sglang and for arg strings with no
    #    JSON.
    resolved_server_args = compact_json_server_args(
        resolved_server_args, bench.get("framework"),
    )
    if resolved_server_args:
        envs[framework_env] = resolved_server_args
    # ── Client trust-remote-code (model-agnostic) ─────────────────────────
    # The MI300X bench scripts (vllm_mi300x.sh / sglang_mi300x.sh) always
    # launch the SERVER with --trust-remote-code, so a custom-tokenizer model's
    # measurement CLIENT must load the same remote code to tokenize prompts —
    # otherwise transformers raises ValueError mid-warmup and the variant fails
    # (seen on Kimi-K2 / Qwen3.6 / any custom-code model). Mirror it onto every
    # client-trust env so custom-code models work WITHOUT per-model special-
    # casing. This is the single choke point every bench path (baseline /
    # profile / sweep / explore / framework_pr / conc_sweep) funnels through.
    # setdefault never overrides an operator's deliberate opt-out (e.g.
    # extra_envs={"BENCH_TRUST_REMOTE_CODE": "0"}).
    for _trust_key in (
        "MAGPIE_TRUST_REMOTE_CODE",  # Magpie sglang remote-direct client
        "BENCH_TRUST_REMOTE_CODE",  # GEAK bench_e2e.sh inferencex client
        "HF_HUB_TRUST_REMOTE_CODE",  # transformers / HF hub tokenizer auto-load
    ):
        envs.setdefault(_trust_key, "1")
    if _model_requires_remote_code(model_path or bench.get("model")):
        _remote_code_existing = str(envs.get(framework_env, "")).strip()
        if "trust-remote-code" not in _remote_code_existing:
            from ._grid_runner import merge_server_args

            envs[framework_env] = (
                merge_server_args(_remote_code_existing, "--trust-remote-code")
                if _remote_code_existing
                else "--trust-remote-code"
            )
    # Server-side trust-remote-code for custom-code models (Qwen3.6 MoE): the
    # checkpoint ships a custom text-generation implementation behind a config
    # that advertises vision_config, so the SERVER must also load remote code
    # or it refuses the arch at boot. Scoped to this exact daily-candidate
    # family so other models' server args are untouched; the client side is
    # already covered model-agnostically above. Merge (never overwrite) so an
    # operator pin survives, and skip when --trust-remote-code is already set.
    if "qwen3.6-35b-a3b" in _model_basename or "qwen3-6-35b-a3b" in _model_basename:
        _trust_existing = str(envs.get(framework_env, "")).strip()
        if "trust-remote-code" not in _trust_existing:
            from ._grid_runner import merge_server_args

            envs[framework_env] = (
                merge_server_args(_trust_existing, "--trust-remote-code")
                if _trust_existing
                else "--trust-remote-code"
            )
    # Accuracy eval (GSM8K) is ON by default; env / extra_envs may override.
    # Disabling it removes the per-variant accuracy gate, so accuracy-
    # destroying changes can pass on throughput alone — warn loudly, never
    # block. Resolved after extra_envs merging so the source is honored.
    if "RUN_EVAL" not in envs:
        env_run_eval = os.environ.get("RUN_EVAL")
        envs["RUN_EVAL"] = env_run_eval if env_run_eval is not None else "true"
    if str(envs.get("RUN_EVAL", "")).strip().lower() in _RUN_EVAL_FALSE_VALUES:
        global _RUN_EVAL_DISABLED_WARN_EMITTED
        if not _RUN_EVAL_DISABLED_WARN_EMITTED:
            log.warning(
                "RUN_EVAL is disabled: no per-variant accuracy gate, so "
                "accuracy regressions will not be caught. Set RUN_EVAL=true "
                "to restore the gate. This warning fires once per process."
            )
            _RUN_EVAL_DISABLED_WARN_EMITTED = True
    # KernelForge fp8 block-scale CK backend switch: when the coordinator
    # promoted an fp8-blockscale gemm_tuning KEEP it injects
    # SGLANG_FP8_BLOCKSCALE_CK_MAX_M into the serving envs. That env only takes
    # effect on a KernelForge-patched sglang fp8_utils.py (M-aware CK routing);
    # the unpatched tree ignores it. Ensure the patch here, strictly scoped to
    # sglang + an active CK optimization (env present) — there is no point
    # patching otherwise. Fail-soft: a failed patch just leaves the env a no-op,
    # so the serving run still proceeds (never hard-fail). Honors the
    # HYPERLOOM_ENABLE_PATCH kill switch like the TraceLens hook above.
    _fw = str(bench.get("framework") or "").lower()
    if (
        _tracelens_patch_enabled()
        and "sglang" in _fw
        and "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" in envs
    ):
        if not ensure_sglang_patched_for_ck_blockscale():
            log.warning(
                "CK fp8 block-scale patch could not be applied; "
                "SGLANG_FP8_BLOCKSCALE_CK_MAX_M will no-op on the unpatched "
                "sglang fp8_utils.py (serving run continues unaffected)."
            )
    # sglang FP8 per-channel/per-token CK fast path: a dense FP8 checkpoint
    # with per-channel weight + per-token (dynamic) activation falls into the
    # slow unfused _apply_fallback_scaled_mm in sglang's apply_fp8_linear
    # unless SGLANG_USE_AITER_FP8_PER_TOKEN=1 flips use_per_token_if_dynamic on
    # and routes the GEMM to aiter's CK gemm_a8w8_bpreshuffle. Inject it from
    # Hyperloom, strictly scoped to sglang + fp8 + gfx942 + that exact quant
    # scheme so per-tensor and block-scale FP8 are never touched. setdefault so
    # an operator-set value (YAML / extra_envs) always wins.
    from ...cli import _resolve_amd_gpu_type

    _model_for_quant = str(model_path or os.environ.get("MODEL_PATH", ""))
    if (
        "sglang" in _fw
        and str(bench.get("precision") or "").strip().lower() == "fp8"
        and _resolve_amd_gpu_type(gpu_type or bench.get("runner_type"))
        in _GFX942_GPU_TYPES
        and _fp8_is_per_channel_per_token(_model_for_quant)
    ):
        envs.setdefault("SGLANG_USE_AITER_FP8_PER_TOKEN", "1")
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = output_dir / out_name
    with materialized.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return materialized


__all__ = [
    "default_baseline_config",
    "materialize_config_with_envs",
]
