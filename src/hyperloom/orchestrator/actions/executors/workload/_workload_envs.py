# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared workload-env materialization (single source of truth).

This module is the single source of truth for rendering a point YAML with the
user's actual process-env workload contract, so the baseline and grid-runner
paths render identical YAML:

* :func:`materialize_config_with_envs` — write a per-run YAML honoring process
  env (+ optional caller overrides).
* :func:`default_baseline_config` — pick the shipped YAML by ``$FRAMEWORK``.

Used by ``baseline.py`` (materializes once, surfaces the path) and the
``explore`` / ``sweep`` grid runs (fall back to materializing on their own).

:func:`~._server_argv.seal_server_argv` settles the server argument string at
the end of :func:`materialize_config_with_envs`; nothing may write it after.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import yaml

from hyperloom.common.benchmark_config import BenchmarkPoint, read_benchmark_point
from hyperloom.common.outside_facts import ENV_EVAL_LIMIT, ENV_EVAL_TASKS, ENV_EVAL_TOKENIZED_REQUESTS
from hyperloom.common.agentx_workload import is_agentx_client_key
from hyperloom.common.visible_devices import mask_tokens
from hyperloom.common.session_files import shared_options
from hyperloom_common.coerce import to_str_list
from hyperloom_common.io import atomic_write_text
from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path
from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from hyperloom.common.perf_metric import DATASET_PATH_ENV, agentx_enabled
from hyperloom.common.env_safety import (
    BENCHMARK_SECRET_ENV_NAMES,
    BLOCKED_EXTERNAL_ENV_NAMES,
    filter_untrusted_env_mapping,
    is_allowed_variant_env_key,
)
from hyperloom.common.workload_defaults import (
    DEFAULT_CONC,
    DEFAULT_ISL,
    DEFAULT_OSL,
)
from hyperloom.inference_optimizer.session.paths import asset_root, session_dir
from hyperloom.inference_optimizer.framework_paths import ENV_FLYDSL_EXTRA_SOURCE_DIRS, flydsl_extra_source_dirs
from hyperloom.inference_optimizer.grid_server_args import (
    WorkloadTokenWindow,
    dedup_vllm_server_args,
    inject_sglang_attention_backend,
    inject_sglang_context_length,
    inject_sglang_watchdog_timeout,
)
from hyperloom.inference_optimizer.grid_server_args import merge_server_args
from hyperloom.inference_optimizer.grid_server_args import remove_moe_runner_backend_arg
from hyperloom.inference_optimizer.grid_server_args import validate_server_args_shell_safe
from inference_suite.argv import compact_json_server_args, remove_server_args
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.framework_registry import server_args_env_name
from hyperloom.orchestrator.state.base_launch import ServerBase, session_cards, session_server_base
from ..gpu._gpu_pin import pin_rocr_visible_devices
from ..serving._server_argv import add_server_arg_unless_pinned, seal_server_argv
from .._profile_setup import _AtomTracelensCaps, _profiler_bound_holds, _profiler_flag_value, apply_atom_profile_caps
from .._profile_setup import fit_capture_to_speculative
from ._agentx_switch import agentx_env_for_conc, apply_agentx_switch
from ._framework_runtime import (
    _apply_vllm_source_runtime,
    _operator_extra_env,
    _sync_repo_aliases,
    _vllm_image_source,
    apply_scriptable_runtime_defaults,
)
from hyperloom.inference_optimizer.model_config_utils import (
    _fp8_is_per_channel_per_token,
    load_model_config_dict,
    _sparse_kv_block_size,
)

log = logging.getLogger(__name__)

_XDIT_EXTRA_ARGS_ENV = framework_registry.extra_args_env("xdit")

# The aiter CK gemm_a8w8_bpreshuffle kernel ships for gfx942 / CDNA3 only; gfx950
# has no such kernel. The gate is the arch, so the board list is read off the
# board table rather than typed out beside it.
_GFX942_GPU_TYPES = frozenset(product for product, board in AMD_GPU_BOARDS.items() if board.gfx == "gfx942")


# SGLang graph-capture profiling flag shipped by the profile YAML, and the
# eager flag that makes it a no-op. Literals rather than an import from
# ``baseline``: that module imports this one, so the dependency only goes one
# way.
_SGLANG_PROFILE_CUDA_GRAPH_FLAG = "--enable-profile-cuda-graph"
_SGLANG_DISABLE_CUDA_GRAPH_FLAG = "--disable-cuda-graph"

# Quality-reference env names, in resolution order. Every scriptable workload
# needs this gate, so the contract is the framework-neutral ``HYPERLOOM_`` pair.
# The ``XDIT_`` pair predates ``--framework custom`` and is still both read and
# written because operator bench scripts live outside this repo and cannot be
# renamed in lockstep; drop it once those scripts have moved over.
QUALITY_REF_ENVS = ("HYPERLOOM_QUALITY_REF", "XDIT_QUALITY_REF")
_QUALITY_REF_WRITE_ENVS = ("HYPERLOOM_QUALITY_REF_WRITE", "XDIT_QUALITY_REF_WRITE")


def _first_env(names: tuple[str, ...]) -> str:
    """Return the first non-empty stripped value among ``names`` in ``os.environ``.

    Args:
        names (tuple[str, ...]): Env var names in resolution order.

    Returns:
        str: The first non-empty value, or ``""`` when none is set.
    """
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _model_requires_remote_code(model_path: str | None) -> bool:
    """Return whether benchmark server/client must trust custom HF code.

    Fires for any local config that advertises a custom AutoTokenizer (e.g.
    Kimi K2.6, ``model_type=kimi_k25``), so custom-code tokenizers do not need
    per-model special cases.
    """
    model = str(model_path or "").strip()
    if not model:
        return False
    data = load_model_config_dict(model)
    basename = Path(model).name.lower()
    if data is None:
        # Fallback for mounted model dirs whose config is unreadable.
        return "kimi-k2" in basename or "kimi_k2" in basename
    model_type = str(data.get("model_type") or "").lower()
    archs = {str(a).lower() for a in data.get("architectures") or []}
    if model_type == "kimi_k25" or "kimik25forconditionalgeneration" in archs:
        return True
    auto_map = data.get("auto_map")
    return isinstance(auto_map, dict) and bool(auto_map.get("AutoTokenizer"))


#: Tokenizer modes the benchmark client actually implements a loader for.
#:
#: A tokenizer mode is a loader backend, not a model type: naming one the client
#: does not implement makes it reject the flag, which fails exactly the way the
#: unnamed tokenizer did. So "transformers cannot map this model_type" is
#: necessary but not sufficient -- the mode has to be one the client knows.
#: ``deepseek_v4`` is routed by InferenceX's ``benchmark_serving.py`` to vLLM's
#: own loader; other custom-code checkpoints (``kimi_k25``, say) are served by
#: the trust-remote-code path above and must NOT be named here.
_CLIENT_TOKENIZER_MODES: frozenset[str] = frozenset({"deepseek_v4"})


def _client_tokenizer_mode(model_path: str | None) -> str:
    """Return the tokenizer mode the benchmark client must be told, or ``""``.

    The generic bench client resolves a checkpoint through HF ``AutoConfig``. A
    model whose ``model_type`` this transformers build does not know dies there
    with ``KeyError: '<model_type>'`` before issuing a request, so no throughput
    result is written and the round is graded a boot failure with the server up
    and serving. InferenceX's client already routes ``--tokenizer-mode`` to
    vLLM's own loader, which does know it; this names the mode to pass.

    Model-agnostic on purpose: the question asked is "can HF resolve this
    model_type", not "is this DeepSeek-V4". A model HF understands returns ``""``
    and the client argv is unchanged.
    """
    model = str(model_path or "").strip()
    if not model:
        return ""
    data = load_model_config_dict(model)
    if data is None:
        return ""
    model_type = str(data.get("model_type") or "").strip().lower()
    if model_type not in _CLIENT_TOKENIZER_MODES:
        return ""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError:
        # No transformers, no HF resolution to reason about; naming a mode here
        # would be a guess.
        return ""
    if model_type in CONFIG_MAPPING:
        return ""
    return model_type


class FrameworkScriptMismatchError(ValueError):
    """Raised when benchmark_script targets a different framework than the run.

    Subclasses ValueError so callers can catch it specifically and turn it
    into a structured action failure instead of an uncaught exception.
    """


def _coerce_workload_int_env(env_key: str, raw: str) -> int:
    """Coerce workload env values, accepting ``CONC`` comma ladders."""
    text = str(raw or "").strip()
    if env_key == "CONC" and "," in text:
        values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
        if not values or any(v <= 0 for v in values):
            raise ValueError(f"{env_key}={raw!r} must contain positive integers")
        return values[0]
    value = int(text)
    if value <= 0:
        raise ValueError(f"{env_key}={raw!r} must be positive")
    return value


def default_baseline_config() -> Path:
    """Resolve the shipped point YAML for ``$FRAMEWORK``, or for the default framework when it is unset.

    Resolved at call time so ``$INFERENCE_OPTIMIZER_ASSET_ROOT`` is honoured.

    Returns:
        Path: The shipped point YAML config path for the resolved framework.
    """
    from hyperloom.common.benchmark_config import benchmark_config_path

    source = benchmark_config_path()
    if source is not None:
        return source
    fw = os.environ.get("FRAMEWORK") or framework_registry.DEFAULT_FRAMEWORK
    return asset_root() / "assets" / "configs" / framework_registry.shipped_config_name("baseline", fw)


def _finalize_framework_server_args(bench: dict[str, Any], window: _RequestWindow) -> None:
    """Apply the final framework server-arg guard pipeline in place
    (context-length/watchdog/attention/dedup/compact/shell-safe); order is fixed.

    1. --context-length cap: sglang sizes max_total_tokens off the model's
       max_position_embeddings, so a huge native window balloons the aiter
       workspace past GPU memory. Cap to ISL+OSL+headroom, clamped to the
       native window AND to the run's MAX_MODEL_LEN.
    2. MI300X cold-compile guard: raise sglang's scheduler watchdog so the
       first-request aiter JIT compile survives (the 300s default fires
       SIGQUIT mid-warmup on a cold aiter cache).

    Steps 1-3 are sglang-scoped; steps 4 and 5 are vLLM/atom-scoped. The
    inline comments below carry the per-step rationale."""
    envs = bench["envs"]
    framework_env = server_args_env_name(bench.get("framework"))
    resolved_server_args = str(envs.get(framework_env, "")).strip()
    resolved_server_args = inject_sglang_context_length(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
        WorkloadTokenWindow(window.isl, window.osl, envs.get("MAX_MODEL_LEN") or os.environ.get("MAX_MODEL_LEN")),
    )
    resolved_server_args = inject_sglang_watchdog_timeout(
        resolved_server_args,
        bench.get("framework"),
    )
    # 3. Dual-chunk attention backend: Qwen 1M models need
    #    dual_chunk_flash_attn; inject it unless --attention-backend is pinned.
    resolved_server_args = inject_sglang_attention_backend(
        resolved_server_args,
        bench.get("framework"),
        bench.get("model"),
    )
    # 4. vLLM/atom argparse dedup: collapse repeated single-value flags to
    #    last-wins (vLLM crashes EngineCoreProc on a duplicate); no-op for
    #    sglang.
    resolved_server_args = dedup_vllm_server_args(
        resolved_server_args,
        bench.get("framework"),
    )
    # 5. JSON-valued flags (--speculative-config / --compilation-config /
    #    --hf-overrides ...): $EXTRA_VLLM_ARGS travels unquoted, so
    #    compact each JSON blob to be space-free so it survives as one shell
    #    word. No-op for arg strings with no JSON.
    resolved_server_args = compact_json_server_args(resolved_server_args)
    resolved_server_args = validate_server_args_shell_safe(resolved_server_args)
    if resolved_server_args:
        envs[framework_env] = resolved_server_args


class _RequestWindow(NamedTuple):
    isl: int
    osl: int
    profiling: bool
    serves_requests: bool
    #: ``(sentinel, flag)`` pairs the re-assertion restores when a later step drops one;
    #: a dropped ``max_iterations`` is an OOM, not a cosmetic problem.
    vllm_profiler_flags: list[tuple[str, str]]
    #: The serialization-safe cap, above which (or at zero, which vLLM reads as no
    #: limit) a surviving ``max_iterations`` is rejected rather than trusted.
    vllm_profiler_cap: int
    atom_caps: _AtomTracelensCaps | None
    xdit_profiler_args: tuple[str, ...] = ()
    #: The copy of the framework TraceLens's patch was laid in (``_server_patcher``), which the server must import.
    patched_copy: Path | None = None


def _pin_benchmark_target(bench: dict[str, Any], target: BenchmarkTarget) -> None:
    if target.model_path:
        bench["model"] = str(target.model_path)
    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision
    if target.benchmark_script:
        bench["benchmark_script"] = str(target.benchmark_script)
    effective_inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if effective_inferencex_path:
        # Persist the resolved InferenceX checkout so the round's checkout
        # matches Hyperloom's patch target.
        bench["inferencex_path"] = effective_inferencex_path


def _refuse_framework_script_mismatch(bench: dict[str, Any]) -> None:
    # Only trip when the script carries a DIFFERENT known framework's prefix, so
    # custom/non-prefixed scripts are not falsely rejected.
    _script = str(bench.get("benchmark_script") or "").lower()
    _fw = str(bench.get("framework") or "").lower()
    from hyperloom.inference_optimizer import framework_registry

    _known_fw = framework_registry.names()
    if _script and _fw in _known_fw:
        _other = [k for k in _known_fw if k != _fw and _script.startswith(f"{k}_")]
        if _other:
            raise FrameworkScriptMismatchError(
                f"framework/script mismatch: framework={_fw!r} but "
                f"benchmark_script={_script!r} targets {_other[0]!r}; refusing "
                f"to boot server (would launch the wrong framework's entrypoint)"
            )


def _pin_workload_shape_envs(envs: dict[str, Any], *, in_session: bool) -> None:
    for env_key in ("CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "EP", "PORT"):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = _coerce_workload_int_env(env_key, val)
    # RANDOM_RANGE_RATIO is a float feeding the steady-state formulas below; do
    # NOT coerce to int.
    r_env = os.environ.get("RANDOM_RANGE_RATIO", "").strip()
    if r_env:
        envs["RANDOM_RANGE_RATIO"] = float(r_env)
    dataset = os.environ.get(DATASET_PATH_ENV, "").strip()
    if dataset:
        envs[DATASET_PATH_ENV] = dataset
    tp_from_yaml = envs.get("TP")
    if tp_from_yaml:
        resolved_tp = int(tp_from_yaml)
    else:
        # Derive TP from the pinned GPU list when the YAML doesn't set TP.
        resolved_tp = (
            len(mask_tokens(envs.get("ROCR_VISIBLE_DEVICES")) or mask_tokens(os.environ.get("ROCR_VISIBLE_DEVICES")))
            or 1
        )
    envs["TP"] = resolved_tp
    pin_rocr_visible_devices(envs, cards=session_cards(session_dir(), resolved_tp) if in_session else resolved_tp)


def _apply_xdit_profile(envs: dict[str, Any]) -> tuple[str, ...]:
    """Set xDiT profiler flags and write patch status; return the flags for restore."""
    from .._server_patcher import ensure_xdit_patched_for_tracelens
    from .._profile_setup import _tracelens_patch_enabled, write_tracelens_patch_status

    xdit_patch_attempted = _tracelens_patch_enabled()
    xdit_patched = xdit_patch_attempted and ensure_xdit_patched_for_tracelens()
    needed = ["--profile_wait 1"]
    if xdit_patched:
        needed += ["--profile_capture_phase", "--profile_with_stack"]
    existing = str(envs.get(_XDIT_EXTRA_ARGS_ENV, "")).strip()
    missing = [f for f in needed if f not in existing]
    if missing:
        envs[_XDIT_EXTRA_ARGS_ENV] = f"{existing} {' '.join(missing)}".strip()
    write_tracelens_patch_status(envs, attempted=xdit_patch_attempted, patched=xdit_patched, fw="xdit")
    return tuple(needed)


def num_prompts_per_conc(isl: int, osl: int) -> int:
    """Requests a benchmark round sends per unit of concurrency: fewer waves the longer each sequence runs."""
    seq_cost = isl + osl
    if seq_cost <= 1024:
        return 10
    if seq_cost <= 4096:
        return 5
    return 3 if seq_cost <= 16384 else 2


def _shape_request_window(bench: dict[str, Any]) -> _RequestWindow:
    from hyperloom.inference_optimizer import framework_registry

    envs = bench["envs"]
    # Last-resort fallbacks, resolved from the CLI workload defaults so the
    # recipe and the workload spec published beside it cannot disagree about what
    # "unset" means; normally the CLI has already projected the resolved values
    # into these envs before materialization.
    isl_val = int(envs.get("ISL") or DEFAULT_ISL)
    osl_val = int(envs.get("OSL") or DEFAULT_OSL)
    conc_val = int(envs.get("CONC") or DEFAULT_CONC)
    # Steady-state window for profiling configs (detected by YAML
    # ``benchmark.envs.PROFILE`` or ``profiler.torch_profiler.enabled``, not the process env).
    # The captured-step count is capped at
    # a serialization-safe budget; the profile OSL is resolved (and lowered if
    # needed) so the steady-state floor fits that cap:
    #   max_iters    = HYPERLOOM_PROFILE_MAX_STEPS_CAP (default 128)
    #   steady_floor = ceil(OSL * (1 + R) / (2 * CONC)), must be <= max_iters
    #   delay_iters  = OSL * (R + 1) * 3 - max_iters / 2 iterations
    is_profile = str(envs.get("PROFILE", "")).strip() == "1" or (
        bench.get("profiler", {}).get("torch_profiler", {}).get("enabled") is True
    )
    # Scriptable image frameworks (xDiT diffusion) have no LLM decode steady-state
    # window, so the OSL/steady-floor math below is a serving concept they never consume.
    serves_requests = not framework_registry.is_scriptable(bench.get("framework"))
    _is_xdit_profile = is_profile and framework_registry.extra_args_env(bench.get("framework")) == _XDIT_EXTRA_ARGS_ENV
    xdit_saved_args = _apply_xdit_profile(envs) if _is_xdit_profile else ()
    profile_num_prompts: int | None = None
    atom_caps: _AtomTracelensCaps | None = None
    vllm_flags: list[tuple[str, str]] = []
    vllm_cap = 0
    patched_copy: Path | None = None
    if is_profile and serves_requests:
        from .._profile_setup import prepare_profile

        checkout = _vllm_image_source(bench, envs)
        # The copy TraceLens's patch is laid in goes ahead of that checkout (``materialize_config_with_envs``).
        outcome = prepare_profile(bench, envs, vllm_checkout=Path(checkout) if checkout else None)
        osl_val = int(envs["OSL"])
        profile_num_prompts = outcome.num_prompts
        atom_caps = outcome.atom_caps
        vllm_flags = outcome.vllm_flags
        vllm_cap = outcome.vllm_cap
        patched_copy = outcome.patched_copy
    if serves_requests:
        # NUM_PROMPTS / NUM_WARMUPS are serving-request concepts; xDiT drives its
        # own iteration count, so leave them untouched.
        if profile_num_prompts is not None:
            # Profile mode: force-override NUM_PROMPTS to reach the steady-state
            # window.
            envs["NUM_PROMPTS"] = profile_num_prompts
        elif "NUM_PROMPTS" not in envs:
            envs["NUM_PROMPTS"] = conc_val * num_prompts_per_conc(isl_val, osl_val)
        if "NUM_WARMUPS" not in envs:
            envs["NUM_WARMUPS"] = min(conc_val, 8)
    return _RequestWindow(
        isl_val, osl_val, is_profile, serves_requests, vllm_flags, vllm_cap, atom_caps, xdit_saved_args, patched_copy
    )


def _seed_base_launch(bench: dict[str, Any], framework_env: str, base: ServerBase) -> None:
    """Put ``base`` below the YAML and any per-task args (leftmost, so later merges win). A base flag the YAML already
    names is not seeded again, so a materialized config re-materialized carries the base once."""
    envs = bench["envs"]
    if base.args:
        from hyperloom.inference_optimizer.grid_server_args import base_args_not_restated, compose_server_args

        yaml_args = str(envs.get(framework_env, ""))
        envs[framework_env] = compose_server_args(
            base_extra_args=base_args_not_restated(base.args, yaml_args),
            variant_extra_args=yaml_args,
            args_mode="replace",
        )
    for name, value in base.env.items():
        envs.setdefault(name, value)  # never clobber YAML/CLI envs


def _merge_caller_overrides(
    bench: dict[str, Any], framework_env: str, server_args: str, replace_args: bool, extra_envs: dict[str, Any] | None
) -> None:
    envs = bench["envs"]
    if server_args:
        # Merge into (not overwrite) the framework env so the profile path's
        # graph-capture flags aren't dropped.
        existing = str(envs.get(framework_env, "")).strip()
        if replace_args:
            envs[framework_env] = server_args
        else:
            if framework_env == "EXTRA_SGLANG_ARGS" and "--moe-runner-backend" in str(server_args):
                # The caller's backend replaces an inherited one rather than
                # relying on duplicate last-wins flags.
                existing = remove_moe_runner_backend_arg(existing)
            envs[framework_env] = merge_server_args(existing, server_args)
    # Only ``benchmark.envs`` reaches the round, so ``--extra-env`` lands there for
    # every framework: vLLM Ray workers see MTP pins only this way.
    combined_extra: dict[str, Any] = dict(_operator_extra_env())
    from hyperloom.common.agentx_workload import is_agentx_client_script

    if is_agentx_client_script(str(bench.get("benchmark_script") or "")):
        # The AgentX switch already settled these from the same pins; the raw
        # pin would undo its CONC scaling of the warmup grace.
        agentx_env = agentx_env_for_conc()
        combined_extra = {k: v for k, v in combined_extra.items() if not is_agentx_client_key(k, agentx_env)}
    if extra_envs:
        combined_extra.update(extra_envs)
    safe_extra_envs, dropped_extra_envs = filter_untrusted_env_mapping(
        combined_extra, allow_predicate=is_allowed_variant_env_key
    )
    for _dk in dropped_extra_envs:
        log.warning("Dropping extra_envs key %s before benchmark materialization", _dk)
    for key, value in safe_extra_envs.items():
        envs[str(key)] = str(value)
    # aiter logs a line for every tuned-GEMM table lookup it MISSES
    # unconditionally, but the corresponding HIT line only when this is set. Two
    # things depend on having it on:
    #   * GEMM tuning takes its shape list from the misses -- shapes derived from
    #     config.json instead cover 0.4% of what the runtime actually asks for;
    #   * apply verification counts hits to decide whether a tuned artifact was
    #     ever read. Without hit lines, "0 hits" and "hit logging was off" are
    #     indistinguishable, and treating the latter as the former would REVERT
    #     every arm.
    # A scan of 60 production server logs found the flag set in none of them, so
    # this is not a hypothetical gap. setdefault keeps an operator override.
    envs.setdefault("AITER_LOG_TUNED_CONFIG", "1")
    prefer_dir = "CUSTOM_DIR" in safe_extra_envs and "CUSTOM_REPO_PATH" not in safe_extra_envs
    _sync_repo_aliases(bench, envs, framework="custom", prefer_dir=prefer_dir)


def _settle_merged_server_args(bench: dict[str, Any], framework_env: str) -> None:
    envs = bench["envs"]
    # Final dedup after reference/server_args merges: collapse repeated
    # vLLM/atom single-value flags to last-wins so recipe/variant values
    # override earlier ones (no-op for sglang).
    if _final_args := str(envs.get(framework_env, "")).strip():
        envs[framework_env] = dedup_vllm_server_args(_final_args, bench.get("framework"))
    # An eager profile cannot capture a CUDA graph, so the profile YAML's
    # ``--enable-profile-cuda-graph`` is dead weight the server still sets up
    # for. The two flags reach the env from different layers -- the YAML
    # template vs the eager fallback folded into ``extra_server_args`` -- and
    # only meet here, after every merge above, so this is the first point that
    # can see the contradiction.
    if framework_env == "EXTRA_SGLANG_ARGS":
        _sg_tokens = str(envs.get(framework_env, "")).split()
        if _SGLANG_DISABLE_CUDA_GRAPH_FLAG in _sg_tokens and _SGLANG_PROFILE_CUDA_GRAPH_FLAG in _sg_tokens:
            envs[framework_env] = " ".join(t for t in _sg_tokens if t != _SGLANG_PROFILE_CUDA_GRAPH_FLAG)
            log.info(
                "Dropping %s: %s is set, so there is no graph capture to profile.",
                _SGLANG_PROFILE_CUDA_GRAPH_FLAG,
                _SGLANG_DISABLE_CUDA_GRAPH_FLAG,
            )


def _publish_scriptable_wiring(bench: dict[str, Any], window: _RequestWindow, establish_quality_ref: bool) -> None:
    envs = bench["envs"]
    # Only ``benchmark.envs`` reaches the round's processes, so
    # re-inject the quality reference here (the single scriptable choke point)
    # or the shipped empty YAML default silently skips the gate. What the
    # reference holds is the workload's business — an xDiT image, an operator's
    # own artifact — so the default filename is a convention, not a contract.
    # When the operator configures nothing, derive a stable reference path
    # under ``session_dir()`` (pinned via
    # ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``) so baseline + variants
    # resolve the same file.
    if not window.serves_requests:
        _qref = _first_env(QUALITY_REF_ENVS)
        if not _qref:
            _qref = str(session_dir() / "storage" / "quality_ref" / "baseline.png")
        if window.profiling:
            _ref_compare, _ref_write = "", ""
        elif establish_quality_ref:
            _ref_compare, _ref_write = "", (_first_env(_QUALITY_REF_WRITE_ENVS) or _qref)
        else:
            _ref_compare, _ref_write = _qref, ""
        for _name in QUALITY_REF_ENVS:
            envs[_name] = _ref_compare
        for _name in _QUALITY_REF_WRITE_ENVS:
            envs[_name] = _ref_write
    # These three are read by the xDiT runner and by nothing else, so they are
    # keyed on the framework rather than on scriptability: an operator-supplied
    # ``custom`` workload must not have its baseline altered by settings it
    # never declared, least of all an attention backend.
    if framework_registry.extra_args_env(bench.get("framework")) == _XDIT_EXTRA_ARGS_ENV:
        _xdit_model_name = os.environ.get("XDIT_MODEL_NAME", "").strip()
        if _xdit_model_name:
            envs["XDIT_MODEL_NAME"] = _xdit_model_name
        # If set, the baked hyperloom_local_aliases map each registered name to
        # a local snapshot dir rooted at $XDIT_MODEL_ROOT/<slug>. Leave unset in
        # public/default deployments so the operator chooses the model cache
        # location explicitly.
        _xdit_model_root = os.environ.get("XDIT_MODEL_ROOT", "").strip()
        if _xdit_model_root:
            envs["XDIT_MODEL_ROOT"] = _xdit_model_root
        # For the baseline only, force the operator-pinned backend (default
        # 'aiter', the MI300X-verified path) so an invalid agent override cannot
        # poison the reference measurement. Explore/sweep variants keep their
        # freedom to try alternative backends.
        if establish_quality_ref:
            envs["XDIT_ATTENTION_BACKEND"] = os.environ.get("XDIT_ATTENTION_BACKEND", "").strip() or "aiter"


def _pin_model_launch_workarounds(bench: dict[str, Any], model_path: str | None) -> str:
    envs = bench["envs"]
    # A handful of flagship models SIGABRT during CUDA-graph capture on the
    # sglang ROCm image because their DEFAULT fused kernels are buggy on
    # gfx942. Inject the verified per-model work-around unless the caller
    # already pinned it (setdefault/merge, never overwrite). Matched on the
    # model basename.
    _model_basename = Path(str(model_path or os.environ.get("MODEL_PATH", ""))).name.lower()
    if "kimi-k2" in _model_basename:
        # Kimi K2.x at tp8 takes sglang's ROCm fused-decode-MLA path, whose RoPE
        # kernel aborts during CUDA-graph capture. Disabling the fused decode
        # pipeline keeps tp8 + the clean aiter MLA path.
        envs.setdefault("SGLANG_ROCM_FUSED_DECODE_MLA", "0")
    if "mimo-v2" in _model_basename and not framework_registry.loads_model_file(bench.get("framework")):
        # MiMo-V2.x's DEFAULT aiter attention backend SIGABRTs during CUDA-graph
        # capture on gfx942. Pin the triton attention backend. sglang accepts
        # lowercase `triton`; vLLM only knows TRITON_ATTN.
        _mimo_is_vllm = "vllm" in str(bench.get("framework") or "").lower()
        _mimo_attn_backend = "TRITON_ATTN" if _mimo_is_vllm else "triton"
        add_server_arg_unless_pinned(
            envs,
            bench.get("framework"),
            f"--attention-backend {_mimo_attn_backend}",
            pinned_by=("--attention-backend", "--attention_backend"),
        )
        # vLLM registers this checkpoint under MiMoV2FlashForCausalLM but the HF
        # config declares MiMoV2ForCausalLM, which the pod-local vLLM build
        # rejects at boot. Remap the arch via --hf-overrides. vLLM-only; JSON
        # kept space-free so it survives the unquoted splice.
        if _mimo_is_vllm:
            add_server_arg_unless_pinned(
                envs,
                bench.get("framework"),
                '--hf-overrides {"architectures":["MiMoV2FlashForCausalLM"]}',
                pinned_by=("--hf-overrides", "--hf_overrides"),
            )
    # Sparse-attention KV-cache block size (config-derived, model-agnostic).
    # Models like MiniMax-M3 (MSA) place the main K/V and the indexer side-cache
    # in one KV-cache group whose sparse backends only accept the model's
    # sparse_attention_config.sparse_block_size (e.g. 128). vLLM's default
    # --block-size 16 (and the generic launcher's default when EXTRA_VLLM_ARGS
    # is empty) has no common block size with it, so KV-cache init aborts with
    # "No common block size for 16" -- baseline, roofline, and every
    # explore variant crash at startup. Read the required size from the
    # model config and pin --block-size at this shared choke point so it rides
    # EXTRA_VLLM_ARGS on every path (the roofline path in particular seeds from
    # the current-best delta and would otherwise drop the baseline's block size
    # and fall back to the default). vLLM-only: --block-size is a vLLM flag
    # (sglang rejects it; its sparse page size is set differently). Merge (never
    # overwrite) and skip when a --block-size is already pinned so an explicit
    # operator/explore choice wins; the later dedup_vllm_server_args collapses
    # any duplicate last-wins anyway. Config unreadable (e.g. an uncached
    # hub-id) -> None -> no injection, prior behaviour preserved.
    if "vllm" in str(bench.get("framework") or "").lower():
        if _sparse_bs := _sparse_kv_block_size(str(model_path or bench.get("model") or "")):
            add_server_arg_unless_pinned(
                envs,
                bench.get("framework"),
                f"--block-size {_sparse_bs}",
                pinned_by=("--block-size", "--block_size"),
            )
    return _model_basename


def _pin_client_model_access(bench: dict[str, Any], model_path: str | None, model_basename: str) -> None:
    envs = bench["envs"]
    # The MI300X bench scripts always launch the SERVER with
    # --trust-remote-code, so a custom-tokenizer model's CLIENT must load the
    # same remote code or transformers raises mid-warmup. Mirror it onto every
    # client-trust env. setdefault never overrides an operator opt-out.
    for _trust_key in (
        "BENCH_TRUST_REMOTE_CODE",  # the benchmark client
        "HF_HUB_TRUST_REMOTE_CODE",  # transformers / HF hub tokenizer auto-load
    ):
        envs.setdefault(_trust_key, "1")
    # Trusting remote code is not enough when transformers cannot map the
    # model_type at all: the client has to be told which loader to use.
    _model = model_path or bench.get("model")
    if _client_tok_mode := _client_tokenizer_mode(_model):
        envs.setdefault("HYPERLOOM_CLIENT_TOKENIZER_MODE", _client_tok_mode)
    # Qwen3.6 MoE declares an arch the SERVER must also load remote code for, or it
    # refuses the arch at boot.
    _custom_code = "qwen3.6-35b-a3b" in model_basename or "qwen3-6-35b-a3b" in model_basename
    _hf_loader = not framework_registry.loads_model_file(bench.get("framework"))
    if _hf_loader and (_custom_code or _model_requires_remote_code(_model)):
        add_server_arg_unless_pinned(
            envs,
            bench.get("framework"),
            "--trust-remote-code",
            pinned_by=("--trust-remote-code", "--trust_remote_code"),
        )


def _pin_accuracy_eval_envs(envs: dict[str, Any]) -> None:
    # Accuracy eval (GSM8K) is ON unless the round passes ``extra_envs["RUN_EVAL"]`` or the workload is AgentX.
    if "RUN_EVAL" not in envs:
        envs["RUN_EVAL"] = "true"
    # Persist eval task/limit into the YAML: the measurement surface hashes them and the round reads them from there.
    for _eval_key in (ENV_EVAL_TASKS, ENV_EVAL_LIMIT):
        _eval_value = os.environ.get(_eval_key, "").strip()
        if _eval_value and _eval_key not in envs:
            envs[_eval_key] = _eval_value
    # PD (router-fronted): lm_eval's default token-id prompts are rejected by the
    # sglang_router's /v1/completions (HTTP 422, StringOrArray), collapsing the
    # accuracy eval. Force string prompts so the gate works. Explicit env /
    # extra_envs win; only disaggregated runs are touched (aggregated hits the
    # sglang server directly, which accepts token-id prompts).
    _eval_tok_env = os.environ.get(ENV_EVAL_TOKENIZED_REQUESTS, "").strip()
    if not _eval_tok_env:
        from .._multi_node_env import resolve_kb_topology

        if str(resolve_kb_topology().pd_mode or "").lower() == "disaggregated":
            _eval_tok_env = "false"
    if _eval_tok_env and ENV_EVAL_TOKENIZED_REQUESTS not in envs:
        envs[ENV_EVAL_TOKENIZED_REQUESTS] = _eval_tok_env


def _pin_runtime_build_envs(
    bench: dict[str, Any], model_path: str | None, gpu_type: str | None, flydsl_source_dirs: bool
) -> None:
    envs = bench["envs"]
    # FlyDSL folds only same-directory helpers into its JIT cache key, so a patched
    # helper one directory over is served from a stale binary. Naming the roots
    # folds their sources into the key. Only the run that applied such a patch has
    # that hazard; setdefault so an operator-set value (YAML / extra_envs) wins.
    if flydsl_source_dirs and (_flydsl_dirs := flydsl_extra_source_dirs()):
        envs.setdefault(ENV_FLYDSL_EXTRA_SOURCE_DIRS, _flydsl_dirs)
    # sglang FP8 per-channel/per-token CK fast path: a dense FP8 checkpoint
    # with per-channel weight + per-token (dynamic) activation falls into the
    # slow unfused _apply_fallback_scaled_mm in sglang's apply_fp8_linear
    # unless SGLANG_USE_AITER_FP8_PER_TOKEN=1 flips use_per_token_if_dynamic on
    # and routes the GEMM to aiter's CK gemm_a8w8_bpreshuffle. Inject it from
    # Hyperloom, strictly scoped to sglang + fp8 + gfx942 + that exact quant
    # scheme so per-tensor and block-scale FP8 are never touched. setdefault so
    # an operator-set value (YAML / extra_envs) always wins.
    from hyperloom.inference_optimizer.gpu_types import _resolve_amd_gpu_type

    if (
        "sglang" in str(bench.get("framework") or "").lower()
        and str(bench.get("precision") or "").strip().lower() == "fp8"
        and _resolve_amd_gpu_type(gpu_type) in _GFX942_GPU_TYPES
        and _fp8_is_per_channel_per_token(str(model_path or os.environ.get("MODEL_PATH", "")))
    ):
        envs.setdefault("SGLANG_USE_AITER_FP8_PER_TOKEN", "1")


def _apply_caller_removals(envs: dict[str, Any], framework_env: str, remove_args: Any, unset_envs: Any) -> None:
    remove_list = to_str_list(remove_args)
    unset_list = to_str_list(unset_envs)
    if remove_list:
        envs[framework_env] = remove_server_args(envs.get(framework_env, ""), remove_list)
    for key in unset_list:
        # Unsetting a pin retargets the benchmark rather than toggling a knob.
        if str(key).strip().upper() in BLOCKED_EXTERNAL_ENV_NAMES:
            log.warning("Refusing to unset pinned benchmark env %s", key)
            continue
        envs.pop(str(key), None)


def _restate_profile_capture(
    envs: dict[str, Any], framework_env: str, window: _RequestWindow, replace_args: bool
) -> None:
    if window.vllm_profiler_flags and framework_env == "EXTRA_VLLM_ARGS":
        # The profile path's iteration bounds have to be the LAST word on this env,
        # because three separate steps above can drop them: a candidate carrying
        # ``args_mode="replace"`` (which writeback sets as soon as a KEEP needs
        # ``remove_args``) overwrites the whole flag string, ``extra_envs`` can
        # override it outright, and ``remove_args`` itself strips flags by name.
        # vLLM reads a missing ``max_iterations`` as "profile until stop_profile",
        # and the worker then accumulates every profiler event in host RAM at
        # ~60 MiB/s until the cgroup OOM-killer takes it out mid-roofline. No
        # exploration result is worth that, so the bounds win over all three.
        profile_args = str(envs.get(framework_env, "")).strip()
        restored = [
            flag
            for name, flag in window.vllm_profiler_flags
            if not _profiler_bound_holds(
                name,
                _profiler_flag_value(profile_args, name),
                cap=window.vllm_profiler_cap,
            )
        ]
        if restored:
            log.warning(
                "profile server args lost torch-profiler flags %r (replace_args=%s); "
                "restoring them so the profiler cannot run unbounded.",
                restored,
                bool(replace_args),
            )
            # The seal applies the sink-side guard to whatever is left here.
            envs[framework_env] = merge_server_args(profile_args, " ".join(restored))
    if window.xdit_profiler_args and framework_env == _XDIT_EXTRA_ARGS_ENV:
        current = str(envs.get(framework_env, "")).strip()
        missing = [f for f in window.xdit_profiler_args if f not in current]
        if missing:
            log.warning(
                "profile server args lost xDiT profiler flags %r (replace_args=%s); restoring.",
                missing,
                bool(replace_args),
            )
            envs[framework_env] = merge_server_args(current, " ".join(missing))
    # After the restore, so a vLLM delay it re-added is divided like any other.
    if window.profiling and window.serves_requests:
        fit_capture_to_speculative(envs, framework_env)
    # After extra_envs / replace_args / remove_args so TraceLens knobs survive the
    # same last-wins path as vLLM profiler bounds, and before the seal.
    apply_atom_profile_caps(envs, window.atom_caps)


def _drop_benchmark_credentials(envs: dict[str, Any]) -> None:
    # The rendered YAML is persisted, so credentials must not reach it.
    filtered_envs, dropped_credentials = filter_untrusted_env_mapping(
        envs,
        allow_predicate=lambda key: key not in BENCHMARK_SECRET_ENV_NAMES,
    )
    if dropped_credentials:
        log.warning(
            "Dropping control-plane credentials from benchmark envs: %s", ", ".join(sorted(dropped_credentials))
        )
        envs.clear()
        envs.update(filtered_envs)


@dataclass(frozen=True)
class LaunchLayer:
    """One layer of server-arg and env changes composed over a config's launch."""

    extra_server_args: str = ""
    extra_envs: dict[str, Any] | None = None
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = None
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = None
    args_mode: str = "append"


@dataclass(frozen=True)
class BenchmarkTarget:
    """The model, GPU type and benchmark script a rendered config is pinned to."""

    model_path: str | None = None
    gpu_type: str | None = None
    benchmark_script: str | None = None


@dataclass(frozen=True)
class MaterializeOptions:
    """Per-run choices for a materialized config beyond its launch layer and target.

    ``server_base`` is the frozen base a launch is seeded with; ``None`` reads the current session's, and a
    renderer outside a session (the multibranch toolbox) names its own and reads no session state, whose directory is
    then the workspace root and its ``state.json`` another run's: its point runs on that base, on its TP's cards.
    """

    out_name: str = "baseline_config.with_envs.yaml"
    establish_quality_ref: bool = False
    flydsl_source_dirs: bool = False
    agentx_mode: bool | None = None
    grading: Mapping[str, Any] | None = None
    server_base: ServerBase | None = None


def prepare_grid_benchmark_config(
    config_path: Path,
    output_dir: Path,
    *,
    launch: LaunchLayer = LaunchLayer(),
    target: BenchmarkTarget = BenchmarkTarget(),
    options: MaterializeOptions = MaterializeOptions(),
) -> Path:
    """Keep the verified native baseline intact until the grid composes its candidate.

    A native workload YAML that was never materialized (the pre-baseline fallback) is
    materialized first, so the operator ``--server-args`` reach every candidate.
    """
    point = read_benchmark_point(config_path)
    if point.agentx:
        if point.agentx_launch.composed:
            return config_path
        return materialize_benchmark_config(
            config_path, output_dir, launch=LaunchLayer(args_mode=launch.args_mode), target=target, options=options
        )
    return materialize_config_with_envs(config_path, output_dir, launch=launch, target=target, options=options)


def materialize_benchmark_config(
    config_path: Path,
    output_dir: Path,
    *,
    launch: LaunchLayer = LaunchLayer(),
    target: BenchmarkTarget = BenchmarkTarget(),
    options: MaterializeOptions = MaterializeOptions(),
) -> Path:
    """Resolve the workload owner before applying server candidate changes."""
    import yaml

    from hyperloom_common.coerce import to_str_list
    from ..agentx_point import (
        LaunchChanges,
        apply_launch_changes,
        materialize_agentx,
        seed_session_base,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not BenchmarkPoint.from_document(config, source=str(config_path)).agentx:
        return materialize_config_with_envs(config_path, output_dir, launch=launch, target=target, options=options)
    replacing = launch.args_mode == "replace"
    inherited = "" if replacing else os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "")
    seed_session_base(
        config["benchmark"], session_server_base() if options.server_base is None else options.server_base
    )
    apply_launch_changes(config["benchmark"], LaunchChanges(args=inherited), accepted=True)
    from hyperloom_common.env import is_truthy

    unset_envs = to_str_list(launch.unset_envs)
    server_env, dropped = filter_untrusted_env_mapping(
        {
            key: value
            for key, value in _operator_extra_env().items()
            if key not in {"MODEL_PATH", "FRAMEWORK", "PRECISION", "TP", "EP", "CONC"}
            and key not in unset_envs
            and not key.startswith(("HYPERLOOM_", "INFERENCE_OPTIMIZER_"))
        },
        allow_predicate=is_allowed_variant_env_key,
    )
    for key in dropped:
        log.warning("Dropping extra_envs key %s before AgentX materialization", key)
    server_env.update(launch.extra_envs or {})
    if is_truthy(server_env.pop("RUN_EVAL", False)):
        raise ValueError("AgentX uses the replay's validity; RUN_EVAL cannot enable GSM8K for this workload")
    changes = LaunchChanges(
        launch.extra_server_args,
        server_env,
        to_str_list(launch.remove_args),
        unset_envs,
        replacing,
    )
    apply_launch_changes(config["benchmark"], changes, accepted=True)
    return materialize_agentx(
        config,
        output_dir / options.out_name,
        model_path=target.model_path,
    )


def materialize_config_with_envs(
    config_path: Path,
    output_dir: Path,
    *,
    launch: LaunchLayer = LaunchLayer(),
    target: BenchmarkTarget = BenchmarkTarget(),
    options: MaterializeOptions = MaterializeOptions(),
) -> Path:
    """Render a per-run point YAML with caller-provided overrides.

    The process env wins over the YAML defaults, except a point's own ``ROCR_VISIBLE_DEVICES``
    (see :func:`pin_rocr_visible_devices`), and the body is the order the phases run
    in: what the run launches, the request shape it serves, the server arguments composed
    from every source, the per-framework and per-model envs, and the caller's own removals.

    Raises:
        FrameworkScriptMismatchError: If ``target.benchmark_script`` targets a
            different known framework than the run's framework.
        ValueError: If the point's ROCR mask names a card outside the process's.
    """
    server_args = (launch.extra_server_args or "").strip()
    operator_server_args = os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip()
    replace_args = str(launch.args_mode or "append").strip().lower() == "replace"
    if operator_server_args and not replace_args:
        server_args = merge_server_args(operator_server_args, server_args)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bench = cfg.setdefault("benchmark", {})
    _pin_benchmark_target(bench, target)
    agentx_run = agentx_enabled() if options.agentx_mode is None else options.agentx_mode
    envs = bench["envs"] = {k: v for k, v in (bench.get("envs") or {}).items() if agentx_run or k != "RUN_EVAL"}
    runtime_path = os.environ.get("PATH", "").strip()
    if "PATH" in envs and runtime_path:
        envs["PATH"] = runtime_path
    apply_scriptable_runtime_defaults(
        bench, envs, gpu_type=target.gpu_type, explicit_benchmark_script=bool(target.benchmark_script)
    )
    apply_agentx_switch(bench, target.model_path, active=agentx_run, grading=options.grading)
    _refuse_framework_script_mismatch(bench)
    _pin_workload_shape_envs(envs, in_session=options.server_base is None)
    framework_env = server_args_env_name(bench.get("framework"))
    window = _shape_request_window(bench)
    _seed_base_launch(
        bench, framework_env, session_server_base() if options.server_base is None else options.server_base
    )
    _merge_caller_overrides(bench, framework_env, server_args, replace_args, launch.extra_envs)
    _settle_merged_server_args(bench, framework_env)
    _publish_scriptable_wiring(bench, window, options.establish_quality_ref)
    model_basename = _pin_model_launch_workarounds(bench, target.model_path)
    _finalize_framework_server_args(bench, window)
    _pin_client_model_access(bench, target.model_path, model_basename)
    _pin_accuracy_eval_envs(envs)
    _pin_runtime_build_envs(bench, target.model_path, target.gpu_type, options.flydsl_source_dirs)
    _apply_caller_removals(envs, framework_env, launch.remove_args, launch.unset_envs)
    _restate_profile_capture(envs, framework_env, window, replace_args)
    _apply_vllm_source_runtime(bench, envs)
    if window.patched_copy is not None:
        # Last, as a vLLM source checkout put first on the path above would shadow the copy.
        prepend_launch_path(envs, "PYTHONPATH", str(window.patched_copy))
    _drop_benchmark_credentials(envs)
    seal_server_argv(envs, bench.get("framework"))
    materialized = output_dir / options.out_name
    atomic_write_text(materialized, yaml.safe_dump(cfg, sort_keys=False), options=shared_options(cfg))
    return materialized


__all__ = [
    "default_baseline_config",
    "materialize_config_with_envs",
]
