# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL + SAFE_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex

# shutil / subprocess are re-exported (not directly used in this module
# anymore after the .preflight split) because tests patch the stdlib module
# singletons via ``cli.shutil.which`` / ``cli.subprocess.run`` /
# ``"hyperloom.inference_optimizer.cli.subprocess.run"``; that attribute path
# must keep resolving through this package because stdlib module patches are
# global and only work here if the ``cli.<module>`` attribute itself exists.
import shutil  # noqa: F401 - re-exported for callers/tests
import subprocess  # noqa: F401 - re-exported for callers/tests
import sys
import time
from pathlib import Path
from typing import Any

from .executors import (  # noqa: F401 - re-exported for callers/tests
    _NOOP_KINDS_KERNEL_ONLY,
    _REAL_EXECUTORS_FULL,
    _build_specialist_executor,
    _noop_prep,
    _register_executors,
)
from .kb import (  # noqa: F401 - re-exported for callers/tests
    _bootstrap_cortex_kb,
    _bootstrap_knowledge_plane,
    _build_recipe_kb_dispatcher,
    _resolve_local_kb_root,
)
from .backends import (  # noqa: F401 - re-exported for callers/tests
    _MULTI_NODE_WORKLOAD_UID_ENV_KEYS,
    _build_backends,
    _build_proposal_scorer,
    _build_robustness_options,
    _robustness_server_configured,
)
from .model_gate import (  # noqa: F401 - re-exported for callers/tests
    _AMD_GPU_TYPES,
    _AMD_UNSUPPORTED_ARCHITECTURES,
    _AMD_UNSUPPORTED_MODEL_TYPES,
    _AMD_UNSUPPORTED_QUANT_ALGOS,
    _AMD_UNSUPPORTED_QUANT_METHODS,
    _CONTEXT_HEADROOM_DEFAULT,
    _CONTEXT_HEADROOM_ENV,
    _GEMMA2_ARCHITECTURES,
    _GFX_TO_RUNNER,
    _MAX_MODEL_LEN_HEADROOM,
    _MAXPOS_CONFIG_KEYS,
    _NESTED_ONLY_UNRECOGNIZED_MODEL_TYPES,
    _PHI3_ROPE_TYPES,
    _ROPE_CONFIG_KEYS,
    _SAFETENSORS_HEADER_LIMIT,
    _SUPPORTED_ARCH_MARKERS,
    _SUPPORTED_MODEL_TYPES,
    _TEXT_COERCIBLE_MODEL_TYPES,
    _TEXT_DECODER_CONFIG_KEYS,
    _TOKENIZER_ARTIFACT_FILES,
    _UNRECOGNIZED_ARCHITECTURES,
    _UNRECOGNIZED_MODEL_TYPES,
    _UNREGISTERED_CUSTOM_CONFIG_TYPES,
    _UNSUPPORTED_ARCHITECTURES,
    _UNSUPPORTED_CONFIG_KEYS,
    _UNSUPPORTED_MODEL_TYPES,
    _VERDICT_TEXT_COERCIBLE,
    _VERDICT_VISION_ONLY,
    _VOCAB_WEIGHT_NAMES,
    _arch_is_supported_text_generation,
    _autodetect_gpu_type,
    _config_architectures,
    _config_declares_text_decoder,
    _context_headroom_tokens,
    _detect_amd_unsupported_quant,
    _detect_gemma2_missing_hidden_act,
    _detect_incompatible_model_config,
    _detect_missing_tokenizer_files,
    _detect_phi3_rope_scaling_incompatible,
    _detect_unrecognized_architecture,
    _detect_unsupported_model,
    _detect_vocab_weight_shape_mismatch,
    _gpu_runner_type,
    _load_model_arch,
    _load_model_config_dict,
    _load_model_config_tags,
    _load_model_max_position_embeddings,
    _model_has_dual_chunk_attention,
    _model_is_moe,
    _preflight_context_window,
    _preflight_model_config_compat,
    _preflight_unsupported_model_arch,
    _read_safetensors_header,
    _resolve_amd_gpu_type,
    _resolve_gpu_type,
    _resolve_max_model_len,
)
from ..model_config_utils import (  # noqa: F401 - re-exported for callers/tests
    _model_is_gemma2,
    summarize_model_config,
)
from .bootstrap import (  # noqa: F401 - re-exported for callers/tests
    _default_target_summary,
    _parse_conc_sweep_concs,
    _print_final_summary,
    _print_kernel_opt_summary_line,
    _print_session_skeleton,
    _read_failure_summary,
    _reconcile_crash_count,
    _resolve_reference_recipe,
    _resolve_session_dir_for_summary,
    _seed_shared_state,
    _snapshot_system_prompts,
    resolve_model_display_name,
)
from hyperloom.orchestrator.actions.executors._aiter_jit import (
    AITER_LOCK_STALE_MINUTES,
    clean_stale_aiter_locks as _clean_stale_aiter_locks_impl,
)

# Cohesive clusters live in sibling modules; re-exported here so the module
# namespace + monkeypatch surface is intact.
from .credentials import (
    _CLAUDE_PREFERRED_MODEL as _CLAUDE_PREFERRED_MODEL,
    _CLAUDE_FALLBACK_MODEL as _CLAUDE_FALLBACK_MODEL,
    _CLAUDE_ALLOWED_MODELS as _CLAUDE_ALLOWED_MODELS,
    _CATALOG_RETRY_DELAYS_SEC as _CATALOG_RETRY_DELAYS_SEC,
    _CRITIC_AGENT_ROOT_ENV as _CRITIC_AGENT_ROOT_ENV,
    _resolve_critic_agent_root as _resolve_critic_agent_root,
    _validate_critic_agent_runtime as _validate_critic_agent_runtime,
    _ROBUSTNESS_AGENT_ROOT_ENV as _ROBUSTNESS_AGENT_ROOT_ENV,
    _resolve_robustness_agent_root as _resolve_robustness_agent_root,
    _validate_robustness_agent_runtime as _validate_robustness_agent_runtime,
    _GEAK_BASE_URL_RE as _GEAK_BASE_URL_RE,
    _is_stale_proxy_url as _is_stale_proxy_url,
    _sync_geak_config_base_url as _sync_geak_config_base_url,
    _derive_anthropic_base_url as _derive_anthropic_base_url,
    _resolve_llm_endpoints as _resolve_llm_endpoints,
    _reset_claude_config_to_upstream as _reset_claude_config_to_upstream,
    _validate_credentials as _validate_credentials,
)
from .multi_node import (
    _gc_old_profile_traces as _gc_old_profile_traces,
    _resolve_mn_backend as _resolve_mn_backend,
    _provision_multi_node_dynamo_stack as _provision_multi_node_dynamo_stack,
    _provision_multi_node_rayjob_stack as _provision_multi_node_rayjob_stack,
    _replay_kernel_patches_for_multi_node as _replay_kernel_patches_for_multi_node,
)
from .quantization import (
    _quantization_enabled_via_env as _quantization_enabled_via_env,
    _run_quantization_prelude as _run_quantization_prelude,
)
from .recover import (
    _session_recovery_status as _session_recovery_status,
    _run_recover_session as _run_recover_session,
)


# Backward-compat re-exports: these helpers were extracted into cli_executors /
# cli_kb / cli_backends / cli_model_gate / model_config_utils / cli_bootstrap
# during the CLI decomposition, but callers and tests still import them from
# ``hyperloom.inference_optimizer.cli`` (e.g. _grid_runner, test_reference_script). Listing
# them in ``__all__`` keeps the re-export intentional (and silences the unused-
# import lint for a re-export-only binding).
__all__ = [
    # from .cli_executors
    "_NOOP_KINDS_KERNEL_ONLY", "_REAL_EXECUTORS_FULL", "_build_specialist_executor",
    "_noop_prep", "_register_executors",
    # from .cli_kb
    "_bootstrap_cortex_kb", "_bootstrap_knowledge_plane", "_build_recipe_kb_dispatcher",
    "_resolve_local_kb_root",
    # from .cli_backends
    "_MULTI_NODE_WORKLOAD_UID_ENV_KEYS", "_build_backends", "_build_proposal_scorer",
    "_build_robustness_options", "_robustness_server_configured",
    # from .cli_model_gate
    "_AMD_GPU_TYPES", "_AMD_UNSUPPORTED_ARCHITECTURES", "_AMD_UNSUPPORTED_MODEL_TYPES",
    "_AMD_UNSUPPORTED_QUANT_ALGOS", "_AMD_UNSUPPORTED_QUANT_METHODS",
    "_CONTEXT_HEADROOM_DEFAULT", "_CONTEXT_HEADROOM_ENV", "_GEMMA2_ARCHITECTURES",
    "_GFX_TO_RUNNER", "_MAX_MODEL_LEN_HEADROOM", "_MAXPOS_CONFIG_KEYS",
    "_NESTED_ONLY_UNRECOGNIZED_MODEL_TYPES", "_PHI3_ROPE_TYPES", "_ROPE_CONFIG_KEYS",
    "_SAFETENSORS_HEADER_LIMIT", "_SUPPORTED_ARCH_MARKERS", "_SUPPORTED_MODEL_TYPES",
    "_TEXT_COERCIBLE_MODEL_TYPES", "_TEXT_DECODER_CONFIG_KEYS", "_TOKENIZER_ARTIFACT_FILES",
    "_UNRECOGNIZED_ARCHITECTURES", "_UNRECOGNIZED_MODEL_TYPES",
    "_UNREGISTERED_CUSTOM_CONFIG_TYPES", "_UNSUPPORTED_ARCHITECTURES",
    "_UNSUPPORTED_CONFIG_KEYS", "_UNSUPPORTED_MODEL_TYPES", "_VERDICT_TEXT_COERCIBLE",
    "_VERDICT_VISION_ONLY", "_VOCAB_WEIGHT_NAMES", "_arch_is_supported_text_generation",
    "_autodetect_gpu_type", "_config_architectures", "_config_declares_text_decoder",
    "_context_headroom_tokens", "_detect_amd_unsupported_quant",
    "_detect_gemma2_missing_hidden_act", "_detect_incompatible_model_config",
    "_detect_missing_tokenizer_files", "_detect_phi3_rope_scaling_incompatible",
    "_detect_unrecognized_architecture", "_detect_unsupported_model",
    "_detect_vocab_weight_shape_mismatch", "_gpu_runner_type", "_load_model_arch",
    "_load_model_config_dict", "_load_model_config_tags",
    "_load_model_max_position_embeddings", "_model_has_dual_chunk_attention",
    "_model_is_moe", "_preflight_context_window", "_preflight_model_config_compat",
    "_preflight_unsupported_model_arch", "_read_safetensors_header",
    "_resolve_amd_gpu_type", "_resolve_gpu_type", "_resolve_max_model_len",
    # from .model_config_utils
    "_model_is_gemma2",
    # from .cli_bootstrap
    "_default_target_summary", "_parse_conc_sweep_concs", "_print_final_summary",
    "_print_kernel_opt_summary_line", "_print_session_skeleton", "_read_failure_summary",
    "_reconcile_crash_count", "_resolve_reference_recipe", "_resolve_session_dir_for_summary",
    "_seed_shared_state", "_snapshot_system_prompts", "resolve_model_display_name",
]
from .. import framework_registry
from ..session.manifest import load_manifest, write_manifest
from hyperloom.orchestrator.actions.registry import ActionRegistry
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.framework.paths import resolve_source_file_allowlist
from hyperloom.orchestrator.state.objective import Objective, build_objective
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)
from ..session.lock import SessionAlreadyRunning, SessionLock
from ..session.paths import (
    asset_system_prompts_dir,
    make_session_dir,
)


log = logging.getLogger("hyperloom.inference_optimizer.cli")

# _RetiredFlag / _build_parser (+ its purely computational helpers) live in
# .parser; re-exported here so the module namespace is unchanged for
# callers/tests.
from .parser import (  # noqa: F401 - re-exported for callers/tests
    _RetiredFlag as _RetiredFlag,
    _build_parser as _build_parser,
    _default_gpu_specialist_capacity as _default_gpu_specialist_capacity,
    _default_research_lane_capacity as _default_research_lane_capacity,
    _parse_conc_env_default as _parse_conc_env_default,
    _parse_conc_sweep_default as _parse_conc_sweep_default,
    _parse_conc_values as _parse_conc_values,
    _positive_int_arg as _positive_int_arg,
)
# The _preflight cluster (env-hygiene / SDK install / TraceLens-CLI gate /
# diagnostics) lives in .preflight; re-exported here so the module namespace +
# monkeypatch surface is intact. See
# preflight.py's module docstring for why _load_dotenv_fallback /
# _load_kernel_agent_env_fallback / _clone_inferencex use lazy
# package-qualified lookups internally instead of bare-name calls.
from .preflight import (  # noqa: F401 - re-exported for callers/tests
    _check_gpu_visibility as _check_gpu_visibility,
    _check_node_claude_cli as _check_node_claude_cli,
    _check_shm_disk as _check_shm_disk,
    _check_tracelens_cli as _check_tracelens_cli,
    _check_tracelens_root_exists as _check_tracelens_root_exists,
    _clone_inferencex as _clone_inferencex,
    _emit_preflight_diagnostics as _emit_preflight_diagnostics,
    _ensure_python_sdks as _ensure_python_sdks,
    _inferencex_checkout_ok as _inferencex_checkout_ok,
    _is_placeholder_tracelens_path as _is_placeholder_tracelens_path,
    _load_dotenv_fallback as _load_dotenv_fallback,
    _load_kernel_agent_env_fallback as _load_kernel_agent_env_fallback,
    _preflight as _preflight,
    _print_cortex_kb_queue_status as _print_cortex_kb_queue_status,
    _run_ir3_preflight as _run_ir3_preflight,
    _tracelens_required_at_preflight as _tracelens_required_at_preflight,
    _unset_hip_visible_devices as _unset_hip_visible_devices,
    _TRACELENS_REQUIRED_CLIS as _TRACELENS_REQUIRED_CLIS,
)


def _orchestration_rules_fragment_path() -> Path:
    """Path to the rules-only ``orchestration.md`` fragment consumed by ``prompt_builder``.

    Returns:
        Path: The path to the bundled ``orchestration.md`` fragment.
    """
    return asset_system_prompts_dir() / "orchestration.md"


def _normalise_framework_name(value: str | None) -> str:
    """Normalize a framework string for equality checks."""
    return str(value or "").strip().lower().replace("_", "-")


def _enforce_expected_framework(
    framework: str,
    *,
    expected: str | None = None,
) -> None:
    """Fail fast when a launcher-pinned expected framework is violated.

    Long-running launches often pass through generated shell scripts. A stale
    script that mutates ``$FRAMEWORK`` can otherwise silently run a different
    backend than the operator requested. ``EXPECTED_FRAMEWORK`` is the compact
    launcher-facing guard; ``INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK`` is the
    namespaced equivalent for platform integrations.
    """
    actual = _normalise_framework_name(framework)
    expected_raw = (
        expected
        if expected is not None
        else (
            os.environ.get("INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK", "")
            or os.environ.get("EXPECTED_FRAMEWORK", "")
        )
    )
    wanted = _normalise_framework_name(expected_raw)
    if not wanted:
        return
    if wanted != actual:
        print(
            "ERROR: framework mismatch: "
            f"EXPECTED_FRAMEWORK={wanted!r} but resolved framework={actual!r}. "
            "Refusing to launch because this would run a different backend "
            "than the operator requested.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _objective_summary_for_prompt(objective: Objective) -> tuple[str, float | str | None]:
    """Summarise an objective into the ``(kind, value)`` pair the prompt expects.

    Inspects the objective for the first recognised target attribute
    (``target_gain_pct`` → float, ``target_tput_per_gpu`` → float,
    ``baseline_dir`` → str) and pairs it with the objective's ``kind()``.

    Args:
        objective (Objective): The run objective to summarise.

    Returns:
        tuple[str, float | str | None]: ``(kind, value)`` where ``value`` is the
        objective's numeric / string target, or ``None`` when none is present.
    """
    kind = objective.kind()
    value: float | str | None = None
    if hasattr(objective, "target_gain_pct"):
        value = float(getattr(objective, "target_gain_pct"))
    elif hasattr(objective, "target_tput_per_gpu"):
        value = float(getattr(objective, "target_tput_per_gpu"))
    elif hasattr(objective, "baseline_dir"):
        value = str(getattr(objective, "baseline_dir"))
    return kind, value


def _build_orchestration_prompt(
    *,
    no_kernel: bool,
    framework: str,
    objective: Objective,
    max_minutes: int,
    no_explore: bool = False,
    no_framework_agent: bool = False,
    action_registry: ActionRegistry | None = None,
) -> str:
    """Compose the Orchestration system prompt from typed inputs (``--orch-prompt`` overrides).

    Args:
        no_kernel (bool): When ``True`` the kernel actions are disabled.
        framework (str): The serving framework name (e.g. ``sglang``).
        objective (Objective): The run objective summarised into the prompt.
        max_minutes (int): The wall-clock budget in minutes.
        no_explore (bool): When ``True`` the EXPLORE phase is disabled.
        no_framework_agent (bool): When ``True`` the FRAMEWORK_AGENT phase is disabled.
        action_registry (ActionRegistry | None): The action registry to use;
            a fresh loaded registry is built when ``None``.

    Returns:
        str: The composed Orchestration system prompt.
    """
    registry = action_registry or ActionRegistry().load()
    enabled = default_enabled_actions(no_kernel=no_kernel, no_explore=no_explore)
    kind, value = _objective_summary_for_prompt(objective)
    return build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=enabled,
        framework=framework,
        kernel_enabled=not no_kernel,
        explore_enabled=not no_explore,
        framework_agent_phase_enabled=not no_framework_agent,
        objective_kind=kind,
        objective_value=value,
        max_minutes=int(max_minutes),
        rules_fragment_path=_orchestration_rules_fragment_path(),
        framework_source_roots=resolve_source_file_allowlist(),
    )


def _load_critic_prompt() -> str:
    """Return the Critic system prompt sourced from ``system_prompts/critic.md``.

    Returns:
        str: The contents of ``critic.md``.
    """
    return (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")


_DEFAULT_KERNEL_PROMPT = (
    "You are the Kernel-agent — responder-only. You receive `request`\n"
    "events from Orchestration in your inbox.\n\n"
    "For every un-answered request, emit ONE `response` intent in reply.\n"
    "Schema:\n"
    "  intent_type: response\n"
    "  payload: {\n"
    "    in_reply_to: <request msg_id>,\n"
    "    kind:        '<request.kind>_done',\n"
    "    status:      'ok' | 'failed' | 'needs_review',\n"
    "    result:      { /* whatever the request asked for */ }\n"
    "  }\n\n"
    "Native-only rule: run_optimization must refuse runtime-generated\n"
    "torch.compile/Inductor/Triton cache kernels. Only reusable framework\n"
    "sources under stable repos (aiter/sglang/vllm source trees) are valid\n"
    "kernel-opt targets; otherwise return status='failed' with a clear reason.\n\n"
    "SESSION_DIR contract: every path you emit in result.* must be either\n"
    "verbatim from the request payload, prefixed by SESSION_DIR (injected\n"
    "per tick), or under one of `/sgl-workspace/aiter/`, `/sgl-workspace/\n"
    "sglang/`, `/sgl-workspace/vllm/` (the framework source allowlists).\n"
    "PolicyGate rejects responses whose path fields escape this set.\n\n"
    "If your inbox has no requests, emit one send_message{topic='heartbeat',\n"
    "body_md='ok'}. You may NOT propose, delegate, or initiate REQUESTs."
)



# Per-attempt read timeout for the gateway /models catalog probe. The AMD
# gateway is documented-flaky; on slow/borderline days a healthy /models call
# can take ~5.5s, straddling this cutoff and causing spurious "gateway catalog
# unreachable" launch refusals even though the gateway is up. Allow an operator
# override via env (default unchanged at 5.0s) for slow-gateway windows.
try:
    _CATALOG_REQUEST_TIMEOUT_SEC = float(
        os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_TIMEOUT_SEC", "5.0")
        or "5.0"
    )
except (TypeError, ValueError):
    _CATALOG_REQUEST_TIMEOUT_SEC = 5.0










def _apply_atom_auto_tighten(args: argparse.Namespace) -> list[str]:
    """Validate atom-specific CLI knobs: sole job is the ``--nodes>=2`` fail-fast guard (IR-8).

    No auto-tightening is applied; kernel/framework/profile all work on atom. Multi-node TP wiring
    is unimplemented so ``--nodes>=2`` exits 2. Returns the list of auto-disabled flags (always empty).

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``nodes``).

    Returns:
        list[str]: The auto-disabled flags (always empty).

    Raises:
        SystemExit: With code 2 when ``--nodes >= 2`` (unsupported on atom).
    """
    auto_disabled: list[str] = []
    if int(getattr(args, "nodes", 1) or 1) >= 2:
        print(
            "ERROR: --framework atom does not support multi-node "
            "(--nodes >= 2). atom multi-node TP wiring is deferred; "
            "drop to --nodes 1 or pick --framework sglang/vllm.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(
        "  framework=atom: no auto-disable applied (kernel-agent + "
        "framework-agent + profile / roofline / TraceLens all wired "
        "for atom); --nodes>=2 guard active — see SKILL.md IR-8"
    )
    return auto_disabled


# Forward-looking alias for _apply_atom_auto_tighten (old name kept for tests/SKILL refs; same callable).
_assert_atom_single_node = _apply_atom_auto_tighten




def _emit_launch_info(
    *,
    pid: int,
    session_dir: Path,
    session_id: str,
    run_log: str,
    gpu_type: str,
    framework: str,
    model: str,
    launch_info_file: str | None,
) -> dict[str, Any]:
    """Print the machine-readable HYPERLOOM_LAUNCH stdout line; optionally JSON-dump to ``launch_info_file``.

    Returns the launch_info dict for callers/tests.

    Args:
        pid (int): The launched process id.
        session_dir (Path): The session root directory.
        session_id (str): The session identifier.
        run_log (str): The run-log path string.
        gpu_type (str): The resolved GPU type.
        framework (str): The serving framework name.
        model (str): The model name / path.
        launch_info_file (str | None): Optional path to JSON-dump the launch
            info; skipped when ``None``.

    Returns:
        dict[str, Any]: The launch-info dict that was printed (and optionally
            written).
    """
    launch_info: dict[str, Any] = {
        "event": "launch",
        "pid": pid,
        "session_dir": str(session_dir),
        "session_id": session_id,
        "run_log": run_log,
        "manifest": str(session_dir / "manifest.json"),
        "gpu_type": gpu_type,
        "framework": framework,
        "model": model,
    }
    kv_body = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in launch_info.items())
    print(f"HYPERLOOM_LAUNCH {kv_body}")
    if launch_info_file:
        path = Path(launch_info_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(launch_info, indent=2))
        print(f"Launch info file: {path}")
    return launch_info


# Exit code for "another optimizer already owns this session" (issue #592).
# Distinct from the generic config/usage failures (``2``) so the robustness
# monitor can tell a refused duplicate launch from a real misconfiguration.
SESSION_BUSY_EXIT_CODE = 3


def _acquire_session_lock_or_exit(session_dir: Path) -> SessionLock:
    """Take the single-optimizer session lock or exit ``SESSION_BUSY_EXIT_CODE``.

    Guards both fresh ``optimize`` and ``--resume`` against a second optimizer
    attaching to the same ``session_dir`` (issue #592). When a live optimizer
    already owns the session this refuses to run *before* any ``state.json`` /
    lease mutation, so a misfiring robustness monitor can never corrupt the
    shared session.

    Args:
        session_dir (Path): The resolved session root directory.

    Returns:
        SessionLock: The acquired lock; the caller must keep it referenced for
            the optimizer's lifetime.
    """
    lock = SessionLock(session_dir)
    try:
        lock.acquire()
    except SessionAlreadyRunning as exc:
        print(
            f"ERROR: {exc}. Refusing to start a second optimizer on the same "
            f"session (would corrupt shared leases / state.json). If the owner "
            f"is truly dead, wait for the OS to drop the lock or remove "
            f"{lock.path} before retrying.",
            file=sys.stderr,
        )
        sys.exit(SESSION_BUSY_EXIT_CODE)
    return lock


def _clean_stale_aiter_locks(
    aiter_jit_dir: Path | None = None,
    stale_minutes: int = AITER_LOCK_STALE_MINUTES,
) -> dict[str, Any]:
    """Startup sweep of stale aiter JIT locks; delegates to the shared helper.

    Thin shim preserving the historical signature / stats-dict contract for the
    startup call site and tests. The canonical implementation (plus the
    liveness-gated per-cold-start variant) lives in
    ``orchestrator/action_executors/_aiter_jit.py``.
    """
    return _clean_stale_aiter_locks_impl(
        aiter_jit_dir, stale_minutes=stale_minutes,
    )




# AMD/ROCm runner types (gfx9). dual_chunk_flash_attn (sm90+) is unsupported
# here, and some upstream archs (DSA) are not adapted to AMD yet.




def _resume_safe_flag(
    args: argparse.Namespace,
    arg_name: str,
    manifest: dict | None,
    manifest_key: str,
    *,
    default: bool,
    invert: bool = False,
) -> bool:
    """Resolve a boolean CLI flag with resume-safe manifest fallback: explicit arg → manifest → default.

    ``invert=True`` handles the ``--no-*`` pattern (args.no_X True == disable; manifest stores positive form).
    Lets robustness_monitor.sh resume preserve original intent without re-passing the flag.

    Args:
        args (argparse.Namespace): The parsed CLI namespace.
        arg_name (str): The attribute name to read from ``args``.
        manifest (dict | None): The resume manifest, or ``None``.
        manifest_key (str): The manifest key holding the persisted value.
        default (bool): The fallback value when neither arg nor manifest
            supplies one.
        invert (bool): When ``True`` apply the ``--no-*`` inversion to the
            explicit arg.

    Returns:
        bool: The resolved boolean flag value.
    """
    raw_arg = getattr(args, arg_name, None)
    if isinstance(raw_arg, bool) and raw_arg:
        return (not raw_arg) if invert else raw_arg
    if manifest is not None and manifest_key in manifest:
        stored = manifest.get(manifest_key)
        if isinstance(stored, bool):
            return stored
    return default


def _resume_safe_numeric(
    args: argparse.Namespace,
    arg_name: str,
    manifest: dict | None,
    manifest_key: str,
    *,
    default: float,
) -> float:
    """Float-valued analog of :func:`_resume_safe_flag`: explicit non-default arg → manifest → default.

    Args:
        args (argparse.Namespace): The parsed CLI namespace.
        arg_name (str): The attribute name to read from ``args``.
        manifest (dict | None): The resume manifest, or ``None``.
        manifest_key (str): The manifest key holding the persisted value.
        default (float): The fallback value when neither source supplies one.

    Returns:
        float: The resolved numeric value.
    """
    raw_arg = getattr(args, arg_name, None)
    if raw_arg is not None:
        try:
            v = float(raw_arg)
        except (TypeError, ValueError):
            v = None
        if v is not None and v != default:
            return v
    if manifest is not None and manifest_key in manifest:
        try:
            return float(manifest.get(manifest_key) or default)
        except (TypeError, ValueError):
            pass
    return default






















# kernel-agent env vars that MUST resolve to an existing checkout. Unlike
# ordinary vars (env-wins no-clobber), a stale/invalid inherited value for
# these is CORRECTED from the installer-written env file so trace_analyze does
# not fall back to an empty pod-local dir (issue #722). Scoped to TRACELENS_ROOT
# only: MAGPIE_PATH is deliberately excluded because a merely-existing dir that
# is not a Magpie checkout would flip the downstream preflight into the
# explicit-override branch and hard-fail auto-clone.






















def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | None:
    """Probe ``<base_url>/models`` with retry (gateway flakes); return set of model ids or None.

    TLS verification is on by default; ``INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1`` skips it (warns).

    Args:
        base_url (str): The gateway base URL; ``""`` returns ``None``.
        api_key (str): Optional bearer key sent in the ``Authorization``
            header.

    Returns:
        set[str] | None: The set of model ids from ``<base_url>/models``, or
            ``None`` when the probe is skipped or exhausts its retries.
    """

    if not base_url:
        return None

    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        # _ensure_python_sdks should have installed httpx; return None so the caller decides.
        print(
            "Preflight: WARNING — httpx not importable, skipping catalog "
            "probe. _ensure_python_sdks should have installed it."
        )
        return None

    insecure = os.environ.get(
        "INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE",
        "",
    ).strip().lower() in ("1", "true", "yes")
    if insecure:
        print(
            "Preflight: WARNING — INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE=1 "
            "is set; catalog probe will skip TLS verification while sending "
            "an Authorization: Bearer header. Use only against trusted internal "
            "gateways with self-signed certs."
        )
        try:
            import urllib3  # type: ignore[import-not-found]

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass

    probe_url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    delays = (0.0, *_CATALOG_RETRY_DELAYS_SEC)
    last_err: str = ""
    for i, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = httpx.get(
                probe_url,
                headers=headers,
                timeout=_CATALOG_REQUEST_TIMEOUT_SEC,
                verify=not insecure,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} failed: {last_err}")
            continue
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {(resp.text or '')[:200]}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} got {last_err}")
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            last_err = f"JSON decode: {exc}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} returned non-JSON: {last_err}")
            continue
        ids = {m["id"] for m in data.get("data") or [] if isinstance(m, dict) and isinstance(m.get("id"), str)}
        if not ids:
            last_err = "empty data[]"
            continue
        return ids

    print(f"Preflight: catalog probe exhausted {len(delays)} attempts ({last_err}); cannot validate model availability")
    return None


def _validate_and_resolve_claude_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str] | None,
) -> set[str] | None:
    """Hard-gate Claude model selection (must be in _CLAUDE_ALLOWED_MODELS); mutates ``args.claude_model``.

    Probes the gateway catalog (retries); falls back 4-7→4-6 with a WARN, else sys.exit(2). Returns the
    catalog id set on success (reused by the codex smoke-test).

    Args:
        args (argparse.Namespace): The parsed CLI namespace; ``claude_model``
            may be mutated to the fallback model.
        resolved_urls (tuple[str, str] | None): Optional
            ``(anthropic_base_url, openai_base_url)`` used to resolve the probe
            base URL when env is unset.

    Returns:
        set[str] | None: The gateway catalog id set on success.

    Raises:
        SystemExit: With code 2 when the model is disallowed, the catalog is
            unreachable, or no acceptable model is present.
    """
    chosen = (args.claude_model or "").strip()
    # #340: non-AMD deployments (Vultr / TensorWave / self-hosted gateways)
    # may not serve the AMD-blessed opus-4-7/4-6 ids. The static allowlist is
    # an AMD-network safety default; an operator can opt out via
    # INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1, after which the gateway
    # catalog probe below is the sole gate (a typo still fails because the id
    # won't be in the catalog). Default behavior is unchanged.
    allow_custom = os.environ.get(
        "INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL",
        "",
    ).strip().lower() in ("1", "true", "yes", "on")
    if not allow_custom and chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} is not allowed. "
            f"Orchestration model must be one of {list(_CLAUDE_ALLOWED_MODELS)} "
            f"(preferred: {_CLAUDE_PREFERRED_MODEL}, "
            f"fallback: {_CLAUDE_FALLBACK_MODEL}). Refusing to start. "
            f"For a non-AMD gateway, set "
            f"INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1 to use a custom "
            f"orchestration model validated against your gateway catalog.",
            file=sys.stderr,
        )
        sys.exit(2)
    if allow_custom and not chosen:
        print(
            "ERROR: --claude-model is empty but "
            "INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1; pass an explicit "
            "orchestration model id. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Catalog probe GETs <base>/models. The orchestration model is a Claude
    # model, so probe the Anthropic side first; if that side has no reachable
    # catalog (e.g. native api.anthropic.com without a LiteLLM /models route)
    # fall back to the OpenAI side. INFERENCE_OPTIMIZER_CATALOG_PROBE_URL
    # overrides the host outright (single probe, no fallback).
    catalog_ids = None
    # True when a dedicated Anthropic-side endpoint is configured (split entry).
    # A Claude catalog can only come from that side; if it has no /models route
    # (e.g. native api.anthropic.com or DeepSeek's Anthropic API) we treat the
    # model as "unverifiable" and pass, rather than borrowing the OpenAI (gpt-*)
    # catalog to reject a Claude id.
    anthropic_side_configured = False
    override_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if override_url:
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            or os.environ.get("SAFE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        catalog_ids = _probe_llm_catalog(base_url=override_url, api_key=api_key)
    else:
        anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not anthropic_url and resolved_urls is not None:
            anthropic_url = resolved_urls[0]
        if not openai_url and resolved_urls is not None:
            openai_url = resolved_urls[1]
        anthropic_key = (
            os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            or os.environ.get("SAFE_API_KEY", "")
        )
        openai_key = (
            os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("SAFE_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        )
        # The orchestration model is a Claude model, so its catalog must come
        # from the Anthropic side only. In a split deploy the OpenAI catalog
        # (gpt-*) would never carry a Claude id and must not gate it. Fall back
        # to the OpenAI side ONLY for a single-gateway deploy where both sides
        # resolve to the same endpoint.
        candidates: list[tuple[str, str]] = []
        if anthropic_url:
            anthropic_side_configured = anthropic_url != openai_url
            candidates.append((anthropic_url, anthropic_key))
        if openai_url and openai_url == anthropic_url:
            # single gateway: same URL serves both; OpenAI key is a valid retry
            candidates.append((openai_url, openai_key))
        elif openai_url and not anthropic_url:
            # pure single-gateway with only OPENAI_BASE_URL configured
            candidates.append((openai_url, openai_key))
        seen_urls: set[str] = set()
        for cand_url, cand_key in candidates:
            if not cand_url or cand_url in seen_urls:
                continue
            seen_urls.add(cand_url)
            catalog_ids = _probe_llm_catalog(base_url=cand_url, api_key=cand_key)
            if catalog_ids is not None:
                break

    if catalog_ids is None:
        # Split entry: the Anthropic side has no /models route (native
        # Anthropic API, DeepSeek Anthropic API, etc.). The Claude model can't
        # be verified there and the OpenAI catalog must not gate it, so treat it
        # as unverifiable and proceed rather than refusing to start.
        if anthropic_side_configured:
            print(
                f"Preflight: WARNING — Anthropic-side catalog has no /models route; "
                f"cannot verify --claude-model={chosen!r}. Proceeding (split entry)."
            )
            return None
        # #340: a custom-model deploy may sit behind a gateway without a
        # /models route; don't hard-block — warn and trust the operator id.
        if allow_custom:
            print(
                f"Preflight: WARNING — gateway catalog unreachable; cannot verify "
                f"--claude-model={chosen!r}. Proceeding under "
                f"INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1 (trusting the operator id)."
            )
            return None
        print(
            "ERROR: gateway catalog unreachable after retries; cannot "
            "verify Claude model availability. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    if chosen in catalog_ids:
        print(f"Preflight: Claude model {chosen!r} confirmed in gateway catalog")
        return catalog_ids

    # #340: under the custom-model opt-out the AMD opus-4-6 fallback is
    # meaningless (a non-AMD catalog won't carry it); fail clearly on a
    # catalog miss so the operator fixes the id rather than silently running a
    # model their gateway doesn't serve.
    if allow_custom:
        print(
            f"ERROR: --claude-model={chosen!r} not present in gateway catalog "
            f"(INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1; catalog has "
            f"{sorted(catalog_ids)[:20]}). Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    if _CLAUDE_FALLBACK_MODEL in catalog_ids:
        print(f"Preflight: WARNING — {chosen!r} not in gateway catalog; falling back to {_CLAUDE_FALLBACK_MODEL!r}")
        args.claude_model = _CLAUDE_FALLBACK_MODEL
        return catalog_ids

    print(
        f"ERROR: neither {_CLAUDE_PREFERRED_MODEL!r} nor "
        f"{_CLAUDE_FALLBACK_MODEL!r} present in gateway catalog "
        f"(catalog has {sorted(m for m in catalog_ids if m.startswith('claude-'))}). "
        f"Refusing to start.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _smoke_test_codex_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str] | None,
) -> None:
    """WARN-only catalog check for ``--codex-model`` (no hard gate); flags typos before Coordinator starts.

    Probes the OpenAI-side catalog independently of the Claude check: in a
    split-entrypoint deploy the Claude catalog lives on the Anthropic gateway
    and would not list ``gpt-*``, so reusing it would always false-warn.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``codex_model`` / ``critic_backend`` / ``kernel_codex`` /
            ``no_kernel``).
        resolved_urls (tuple[str, str] | None): ``(anthropic_url, openai_url)``
            from preflight; the OpenAI side is probed for the Codex catalog.
    """
    # Codex is needed by the Kernel-agent (kernel-codex on) and the critic-agent review path.
    critic_uses_codex = args.critic_backend == "agent"
    needs_codex = critic_uses_codex or (args.kernel_codex and not getattr(args, "no_kernel", False))
    if not needs_codex:
        return

    openai_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if not openai_url:
        openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not openai_url and resolved_urls is not None:
        openai_url = resolved_urls[1]
    openai_key = (
        os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("SAFE_API_KEY", "")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    )
    catalog_ids = _probe_llm_catalog(base_url=openai_url, api_key=openai_key)
    if catalog_ids is None:
        # WARN-only path: don't block startup just because the OpenAI catalog
        # is unreachable (the Claude gate already validated reachability).
        print(
            "Preflight: WARNING — OpenAI-side catalog unreachable; skipping "
            "--codex-model verification (CodexBackend may fail at first turn)."
        )
        return

    chosen = (args.codex_model or "").strip()
    if chosen in catalog_ids:
        print(f"Preflight: Codex model {chosen!r} confirmed in gateway catalog")
        return
    print(
        f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog "
        f"({sorted(m for m in catalog_ids if m.startswith('gpt-'))}); "
        f"CodexBackend will fail at first turn. Pass --codex-model with a "
        f"value in the catalog or use --critic-mock / --kernel-claude to "
        f"avoid the Codex path entirely."
    )


# InferenceX clone defaults — kept in sync with
# src/hyperloom/inference_optimizer/assets/install.sh (INFERENCEX_REPO / INFERENCEX_REF).








# Default critic backend ("agent" since Step D); override via env or --critic-mock/--critic-agent.
DEFAULT_CRITIC_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND",
    "agent",
)
_VALID_CRITIC_BACKENDS = ("mock", "agent")


def _resolve_critic_choice(args: argparse.Namespace) -> str:
    """Resolve the active critic backend choice (arg → DEFAULT_CRITIC_BACKEND); hard-fails on invalid.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``critic_backend``).

    Returns:
        str: The resolved critic backend (one of ``_VALID_CRITIC_BACKENDS``).

    Raises:
        SystemExit: With code 2 when the chosen backend is invalid.
    """
    chosen = args.critic_backend
    if chosen is None:
        chosen = DEFAULT_CRITIC_BACKEND
    if chosen not in _VALID_CRITIC_BACKENDS:
        print(
            f"ERROR: critic backend {chosen!r} not in {_VALID_CRITIC_BACKENDS!r} "
            f"(set by --critic-mock / --critic-agent or "
            f"INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    return chosen


# Default robustness backend ("agent"); force heartbeat-only mock via --robustness-mock or env.
DEFAULT_ROBUSTNESS_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND",
    "agent",
)
_VALID_ROBUSTNESS_BACKENDS = ("mock", "agent")


def _resolve_robustness_choice(args: argparse.Namespace) -> str:
    """Resolve the active robustness backend choice (arg → DEFAULT_ROBUSTNESS_BACKEND); hard-fails on invalid.

    Multi-node policy: on ``nodes>=2`` the agent's LocalProbe targets sandbox-local resources that live in
    separate pods (HIGH false positives). Keep ``agent`` only when a robustness-server is configured; else
    auto-downgrade to ``mock`` (explicit --robustness-agent gets a WARN).

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``robustness_backend`` / ``nodes`` and server config).

    Returns:
        str: The resolved robustness backend (one of
            ``_VALID_ROBUSTNESS_BACKENDS``).

    Raises:
        SystemExit: With code 2 when the chosen backend is invalid.
    """
    chosen = getattr(args, "robustness_backend", None)
    explicit = chosen is not None
    if chosen is None:
        chosen = DEFAULT_ROBUSTNESS_BACKEND
    if chosen not in _VALID_ROBUSTNESS_BACKENDS:
        print(
            f"ERROR: robustness backend {chosen!r} not in "
            f"{_VALID_ROBUSTNESS_BACKENDS!r} (set by --robustness-mock / "
            f"--robustness-agent or "
            f"INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND)",
            file=sys.stderr,
        )
        sys.exit(2)
    nodes = int(getattr(args, "nodes", 1) or 1)
    if nodes >= 2 and chosen == "agent" and not _robustness_server_configured(args):
        if explicit:
            print(
                f"WARN: --robustness-agent selected but nodes={nodes} and "
                f"no robustness-server configured — the agent's LocalProbe "
                f"family targets sandbox-local resources (ray, inference "
                f"server, GPU, ...) that all live in separate pods on "
                f"multi-node and surface as HIGH false positives. "
                f"Auto-downgrading to --robustness-mock; configure "
                f"--robustness-server-url / ROBUSTNESS_SERVER_URL to keep "
                f"the agent backend, or pass --robustness-mock explicitly "
                f"to suppress this warning. See "
                f"src/hyperloom/inference_optimizer/multi_node/SKILL.md "
                f"(Robustness limitation in multi-node mode).",
                file=sys.stderr,
            )
        chosen = "mock"
    return chosen


def _reset_state_file(session_dir: Path) -> None:
    """Back up ``state.json`` to ``state.json.preReset.<unix_ts>`` and start fresh (Cortex KB untouched).

    Args:
        session_dir (Path): The session root directory holding ``state.json``.
    """
    state_path = session_dir / "state.json"
    if not state_path.exists():
        return
    import time as _time

    ts = int(_time.time())
    backup_path = session_dir / f"state.json.preReset.{ts}"
    try:
        state_path.replace(backup_path)
    except OSError as exc:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "v0.8 §3.10 --reset-state: could not move %s → %s: %s",
            state_path,
            backup_path,
            exc,
        )
        return
    import logging as _logging

    _logging.getLogger(__name__).info(
        "v0.8 §3.10 --reset-state: backed up state.json to %s; session starts blank.",
        backup_path.name,
    )
















def _argv_has_option(argv: list[str], option: str) -> bool:
    """Report whether ``argv`` explicitly carries a given option.

    Matches both the bare flag (``--tp``) and the ``=``-joined form
    (``--tp=8``).

    Args:
        argv (list[str]): The argument vector to scan.
        option (str): The long-option flag to look for (e.g. ``"--tp"``).

    Returns:
        bool: ``True`` when the option appears in ``argv``, else ``False``.
    """
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def _resolve_run_max_model_len(args: argparse.Namespace) -> tuple[int, str]:
    """Resolve run-wide MAX_MODEL_LEN with explicit operator values winning."""
    if getattr(args, "max_model_len", None):
        return int(args.max_model_len), "--max-model-len"
    max_model_len_env = os.environ.get("MAX_MODEL_LEN", "").strip()
    if max_model_len_env:
        try:
            return _positive_int_arg(max_model_len_env), "$MAX_MODEL_LEN"
        except argparse.ArgumentTypeError as exc:
            print(
                f"ERROR: MAX_MODEL_LEN={max_model_len_env!r} is invalid: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2)
    return (
        _resolve_max_model_len(
            args.isl,
            args.osl,
            str(args.model or ""),
        ),
        "auto",
    )


# Phases that still sit upstream of EXPLORE, so a resume may retroactively
# honour --no-explore. Includes the legacy "FRAMEWORK" name for sessions
# persisted before the FRAMEWORK -> FRAMEWORK_AGENT rename (commit 33ac6ccc).
_PRE_EXPLORE_PHASES: frozenset[str] = frozenset({"", "PRELUDE", "FRAMEWORK", "FRAMEWORK_AGENT"})


def _resume_can_disable_explore(cur_phase: str) -> bool:
    """Whether ``--no-explore`` may still disable EXPLORE for a resumed session.

    Args:
        cur_phase (str): The persisted ``state.phase``; case/whitespace-insensitive.

    Returns:
        bool: ``True`` when the phase is upstream of EXPLORE (EXPLORE not yet
        entered), so the flag can be honoured retroactively.
    """
    return (cur_phase or "").strip().upper() in _PRE_EXPLORE_PHASES


def _build_phase_budget_pct(args: argparse.Namespace) -> dict[str, float]:
    """Map ``--*-pct`` CLI flags to a ``phase -> pct`` override dict.

    Keys MUST be the canonical phase names from
    :mod:`hyperloom.orchestrator.phases.machine_state`; otherwise
    :func:`normalize_budget_pct` silently drops the entry and the phase falls
    back to its library default.
    """
    from hyperloom.orchestrator.phases.machine_state import (
        PHASE_CLOSE,
        PHASE_EXPLORE,
        PHASE_FRAMEWORK_AGENT,
        PHASE_KERNEL_AGENT,
        PHASE_PRELUDE,
        PHASE_SWEEP,
    )

    phase_budget_pct: dict[str, float] = {}
    for cli_field, phase_name in (
        ("phase_budget_prelude_pct", PHASE_PRELUDE),
        ("phase_budget_framework_pct", PHASE_FRAMEWORK_AGENT),
        ("phase_budget_explore_pct", PHASE_EXPLORE),
        ("phase_budget_kernel_pct", PHASE_KERNEL_AGENT),
        ("phase_budget_sweep_pct", PHASE_SWEEP),
        ("phase_budget_close_pct", PHASE_CLOSE),
    ):
        val = getattr(args, cli_field, None)
        if val is not None:
            phase_budget_pct[phase_name] = float(val)
    return phase_budget_pct


def _export_workload_envs_for_optimize(
    args: argparse.Namespace,
    *,
    nodes_resolved: int,
    tp_resolved: int,
    ep_resolved: int,
    argv: list[str] | None = None,
) -> None:
    """Mirror explicit workload CLI flags (--tp/--conc/--ep) into env so executors' Magpie YAMLs honor them.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``conc``).
        nodes_resolved (int): The resolved node count; ``>= 2`` forces export
            of all three knobs regardless of whether they were passed.
        tp_resolved (int): The resolved tensor-parallel size to export as ``TP``.
        ep_resolved (int): The resolved expert-parallel size to export as ``EP``.
        argv (list[str] | None): The argument vector to inspect for explicit
            flags; defaults to ``sys.argv[1:]`` when ``None``.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if nodes_resolved >= 2 or _argv_has_option(argv, "--tp"):
        os.environ["TP"] = str(tp_resolved)
    if nodes_resolved >= 2 or _argv_has_option(argv, "--conc"):
        os.environ["CONC"] = str(max(1, int(getattr(args, "conc", 8) or 8)))
    if nodes_resolved >= 2 or _argv_has_option(argv, "--ep"):
        os.environ["EP"] = str(ep_resolved)


async def _run_optimize(args: argparse.Namespace) -> int:
    """Run the ``optimize`` subcommand end to end.

    Resolves topology arguments (nodes, TP/EP, GPUs per node), runs
    preflight, and drives the optimization session to completion.

    Args:
        args: Parsed CLI arguments for the ``optimize`` subcommand.

    Returns:
        Process exit code (``0`` on success).
    """
    # Surface --nodes (CLI flag wins) before _preflight runs.
    nodes_resolved = max(1, int(args.nodes))
    tp_resolved = max(1, int(getattr(args, "tp", 1) or 1))
    ep_resolved = max(1, int(getattr(args, "ep", 1) or 1))
    # Resolve gpus_per_node with the same CLI > env > 8 chain _provision_multi_node_rayjob_stack uses.
    gpn_attr = getattr(args, "rayjob_gpus_per_node", None)
    if gpn_attr is not None:
        gpus_per_node_resolved = int(gpn_attr)
    else:
        try:
            gpus_per_node_resolved = int(
                os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8,
            )
        except ValueError:
            gpus_per_node_resolved = 8
    total_gpus = nodes_resolved * gpus_per_node_resolved

    # Topology sanity gates — multi-node only (nodes>=2); fail fast vs a cryptic launcher crash mid-cold-start.
    if nodes_resolved >= 2:
        # Gate 1: total cluster GPUs (nodes*gpus_per_node) must hold the model's TP shards.
        if total_gpus < tp_resolved:
            print(
                f"ERROR: TP={tp_resolved} exceeds total GPU count "
                f"({nodes_resolved} nodes * {gpus_per_node_resolved} "
                f"gpus_per_node = {total_gpus}). Either lower --tp, raise "
                "--nodes, or use a larger --rayjob-gpus-per-node pod "
                "template.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Gate 2: EP cannot exceed TP (can't place more expert shards than ranks); fail before bootstrap.
        if ep_resolved > tp_resolved:
            print(
                f"ERROR: EP={ep_resolved} > TP={tp_resolved}. Expert-parallel "
                "size must be <= tensor-parallel size. Either lower --ep or "
                "raise --tp.",
                file=sys.stderr,
            )
            sys.exit(2)

    os.environ["INFERENCE_OPTIMIZER_NODES"] = str(nodes_resolved)
    operator_server_args = str(getattr(args, "server_args", "") or "").strip()
    if operator_server_args:
        os.environ["INFERENCE_OPTIMIZER_SERVER_ARGS"] = operator_server_args
    # Re-export $TP/$CONC/$EP when explicitly supplied (always for multi-node); skip defaults in single-node.
    _export_workload_envs_for_optimize(
        args,
        nodes_resolved=nodes_resolved,
        tp_resolved=tp_resolved,
        ep_resolved=ep_resolved,
    )
    # User-declared grid skip list; re-export so subprocess executors inherit it (empty clears stale values).
    skip_variants_resolved = (getattr(args, "skip_variants", "") or "").strip()
    os.environ["SKIP_VARIANTS"] = skip_variants_resolved
    # Surface PD_* knobs for executors; empty means "resolve from state.json", pd_mode always exported.
    pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
    if pd_mode == "disaggregated" and nodes_resolved < 2:
        # PD disaggregation needs >=2 nodes (separate prefill + decode pods); fail at parse time.
        print(
            f"ERROR: --pd-mode disaggregated requires --nodes >= 2 "
            f"(got --nodes {nodes_resolved}). PD splits the cluster "
            "into prefill + decode groups; a single pod cannot host "
            "both. Either drop --pd-mode (defaults to colocated) or "
            "raise --nodes.",
            file=sys.stderr,
        )
        sys.exit(2)
    os.environ["PD_MODE"] = pd_mode
    if pd_mode == "disaggregated":
        for cli_attr, env_key in (
            ("pd_prefill_nodes", "PD_PREFILL_NODES"),
            ("pd_decode_nodes", "PD_DECODE_NODES"),
            ("pd_prefill_tp", "PD_PREFILL_TP"),
            ("pd_decode_tp", "PD_DECODE_TP"),
        ):
            v = int(getattr(args, cli_attr, 0) or 0)
            if v > 0:
                os.environ[env_key] = str(v)
        for cli_attr, env_key in (
            ("pd_transfer_backend", "PD_TRANSFER_BACKEND"),
            ("pd_ib_device", "PD_IB_DEVICE"),
        ):
            v = (getattr(args, cli_attr, "") or "").strip()
            if v:
                os.environ[env_key] = v

    # Stale aiter JIT lock sweep: killed runs leave locks that block subsequent starts (locks <5min preserved).
    aiter_sweep = _clean_stale_aiter_locks()
    if aiter_sweep["dir"] and aiter_sweep["deleted"]:
        print(
            f"Stale aiter locks cleared: "
            f"dir={aiter_sweep['dir']} "
            f"deleted={aiter_sweep['deleted']} "
            f"skipped_fresh={aiter_sweep['skipped_fresh']} "
            f"errors={aiter_sweep['errors']}"
        )

    resolved_urls = _preflight(args)

    # Hard-gate Claude model before any session work (mutates args.claude_model on fallback; sys.exit(2) on failure).
    _validate_and_resolve_claude_model(args, resolved_urls)
    # Codex smoke probes the OpenAI side independently (split entrypoints).
    _smoke_test_codex_model(args, resolved_urls)

    # `--resume-from <path>` implies `--resume` (operator convenience).
    if args.resume_from and not args.resume:
        args.resume = True

    if args.resume:
        # Resume mode: USER_DATA_PATH stays at workspace level; pick the per-session subdir via --resume-from
        # or auto-pick the latest under <model>/<ts>/ (legacy flat layout fallback). Pin
        # INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR so paths/subprocesses resolve consistently.
        from ..session.paths import (
            ENV_CURRENT_SESSION_DIR,
            find_latest_per_session_dir,
            workspace_root,
        )

        ws = workspace_root()
        if args.resume_from:
            session_dir = Path(args.resume_from).expanduser().resolve()
            try:
                session_dir.relative_to(ws.resolve())
            except ValueError:
                print(
                    f"ERROR: --resume-from {session_dir!r} is not under "
                    f"$USER_DATA_PATH={ws}. Move USER_DATA_PATH to the "
                    f"workspace root (the parent of the per-session subdirs) "
                    f"and pass the per-session subdir via --resume-from.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not session_dir.is_dir():
                print(
                    f"ERROR: --resume-from {session_dir!r} does not exist.",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            picked = find_latest_per_session_dir()
            if picked is not None:
                session_dir = picked
                print("  --resume: auto-picked latest per-session subdir")
            else:
                # Legacy flat layout — workspace_root itself is the session_dir.
                session_dir = ws
                print(
                    f"  --resume: no per-session subdir found under "
                    f"{ws}/<model>/<ts>/; falling back to flat layout "
                    f"({ws})"
                )
        # Pin before Coordinator/SharedState load so paths/subprocesses inherit the resolved location.
        os.environ[ENV_CURRENT_SESSION_DIR] = str(session_dir)
        # Ensure per-session skeleton exists (idempotent mkdir -p).
        for sub in __import__(
            "hyperloom.inference_optimizer.session.paths", fromlist=["_SESSION_SKELETON"]
        )._SESSION_SKELETON:
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        # Single-optimizer guard (issue #592): take the session lock before any
        # state.json / lease access so a misfiring monitor cannot attach a
        # second optimizer to this session. Held for the whole run.
        session_lock = _acquire_session_lock_or_exit(session_dir)

        try:
            manifest = load_manifest(session_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: --resume failed: {exc}", file=sys.stderr)
            sys.exit(2)
        if not (session_dir / "state.json").exists():
            print(
                f"ERROR: --resume failed: {session_dir}/state.json missing "
                f"(manifest exists but Coordinator never wrote SharedState)",
                file=sys.stderr,
            )
            sys.exit(2)
        state = SharedState.load_or_init(session_dir)
        prior_stop = state.stop_reason
        print(f"Resuming session: {session_dir}")
        print(f"  manifest.session_id    : {manifest.get('session_id')}")
        print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
        print(f"  prior cumul_gain      : {state.cumulative_gain:.2f}%")
        print(
            f"  prior current_best    : "
            f"{(state.current_best or {}).get('action')}/"
            f"{(state.current_best or {}).get('tput')}"
        )
        print(f"  prior stop_reason     : {prior_stop or '(none)'}")

        # Re-export session-level env from persisted state so a fresh-shell resume doesn't fall back to YAML defaults.
        if state.model_path:
            os.environ["MODEL_PATH"] = state.model_path
            print(f"  re-exported MODEL_PATH: {state.model_path}")
            # Backfill model_info for sessions created before the field existed
            # (or whose config was unreadable at launch); fail-soft to {}.
            if not state.model_info:
                state.model_info = summarize_model_config(state.model_path)
                if state.model_info:
                    state.save(session_dir)
                    print("  backfilled model_info (from config.json)")
        if state.framework:
            _enforce_expected_framework(state.framework)
            os.environ["FRAMEWORK"] = state.framework
            print(f"  re-exported FRAMEWORK : {state.framework}")
        if state.gpu_type:
            runner_gpu_type = _gpu_runner_type(state.gpu_type)
            os.environ["TARGET_GPU_TYPE"] = state.gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"  re-exported GPU_TYPE  : {state.gpu_type}")
            if runner_gpu_type != state.gpu_type:
                print(f"  Magpie runner GPU_TYPE: {runner_gpu_type}")
        # Re-export workload metadata from SharedState so resume sees the same workload contract (not YAML defaults).
        for state_attr, env_name in (
            ("tp", "TP"),
            # ``ep`` mirrors EP so single-node vLLM MoE resume still injects
            # --enable-expert-parallel (#569); lost EP would silently drop it.
            ("ep", "EP"),
            ("conc", "CONC"),
            ("isl", "ISL"),
            ("osl", "OSL"),
            ("max_model_len", "MAX_MODEL_LEN"),
        ):
            val = getattr(state, state_attr, 0) or 0
            if val:
                os.environ[env_name] = str(val)
                print(f"  re-exported {env_name:<14s}: {val}")
        # Profile-scoped OSL: an explicit --profile-osl on this resume wins;
        # otherwise re-export the value persisted from the original run so the
        # profile phase doesn't silently revert to its default (and re-trigger
        # the oversized-trace / EngineCore-timeout failure this guards against).
        _resume_profile_osl = getattr(args, "profile_osl", None) or getattr(state, "profile_osl", 0)
        if _resume_profile_osl:
            os.environ["PROFILE_OSL"] = str(int(_resume_profile_osl))
            state.profile_osl = int(_resume_profile_osl)
            print(f"  re-exported PROFILE_OSL   : {int(_resume_profile_osl)}")
        if state.precision:
            os.environ["PRECISION"] = state.precision
            print(f"  re-exported PRECISION     : {state.precision}")
        if getattr(state, "framework_version", ""):
            os.environ["FRAMEWORK_VERSION"] = state.framework_version
            print(f"  re-exported FRAMEWORK_VERSION: {state.framework_version}")
        # Honour persisted kernel_enabled on resume; CLI --no-kernel can still override.
        if not state.kernel_enabled:
            args.no_kernel = True
            print("  Kernel-agent          : DISABLED (persisted from original run)")
        # Same persistence contract for the FRAMEWORK_AGENT phase toggle.
        if not bool(getattr(state, "framework_agent_phase_enabled", True)):
            args.no_framework_agent = True
            print("  framework phase       : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_framework_agent", False)):
            # Inverse: honour --no-framework-agent on resume only before FRAMEWORK is entered.
            cur_phase = (getattr(state, "phase", "") or "").strip().upper()
            if cur_phase in ("", "PRELUDE"):
                state.framework_agent_phase_enabled = False
                # Persist immediately; the later conditional save only runs on prior stop_reason/crash.
                state.save(session_dir)
                print("  framework phase       : DISABLING for resume (--no-framework-agent + phase=PRELUDE)")
            else:
                print(
                    f"  framework phase       : WARN --no-framework-agent ignored; "
                    f"session is already in phase={cur_phase!r} "
                    f"(cannot retroactively skip)"
                )
        # Same persistence contract for the EXPLORE phase toggle.
        if not bool(getattr(state, "explore_enabled", True)):
            args.no_explore = True
            print("  explore phase         : DISABLED (persisted from original run)")
        elif bool(getattr(args, "no_explore", False)):
            # Honour --no-explore on resume only before EXPLORE is entered.
            # Phases preceding EXPLORE are PRELUDE and FRAMEWORK_AGENT (the
            # latter renamed from the legacy "FRAMEWORK" in commit 33ac6ccc,
            # which persisted sessions may still carry).
            cur_phase = (getattr(state, "phase", "") or "").strip().upper()
            if _resume_can_disable_explore(cur_phase):
                state.explore_enabled = False
                print(f"  explore phase         : DISABLING for resume (--no-explore + phase={cur_phase or 'PRELUDE'})")
            else:
                print(
                    f"  explore phase         : WARN --no-explore ignored; "
                    f"session is already in phase={cur_phase!r} "
                    f"(cannot retroactively skip)"
                )

        # CRITICAL: clear leftover stop_reason or Orchestration heartbeats forever thinking work is done.
        prior_crash = state.crash_count

        # target_reached is an intentional terminal state;
        # require --force-resume to push past it. Other reasons (time_exhausted, max_ticks, crash) auto-clear.
        force_resume = bool(getattr(args, "force_resume", False))
        gated_terminal = {"target_reached"}
        if prior_stop in gated_terminal and not force_resume:
            print(
                f"\nERROR: --resume blocked by terminal stop_reason="
                f"{prior_stop!r}.\n"
                f"\n"
                f"  SKILL.md (Run-time signals): {prior_stop!r} is a "
                f"deliberate terminal state.\n"
                f"  The optimizer will not auto-resume past it because "
                f"the prior run\n"
                f"  declared exhaustion — picking up where it left off "
                f"only repeats\n"
                f"  the same exhaustion verdict.\n"
                f"\n"
                f"  Override paths:\n"
                f"  1. Pass ``--force-resume`` if you have changed the "
                f"workload /\n"
                f"     search space / model / strategy and want to "
                f"continue regardless.\n"
                f"  2. Start a fresh session (different "
                f"$USER_DATA_PATH) for a clean run.\n"
                f"\n"
                f"  Reports for the prior run live under "
                f"{session_dir}/reports/.\n",
                file=sys.stderr,
            )
            sys.exit(2)

        if prior_stop or prior_crash >= 3:
            state.stop_reason = ""
            state.closing_phase = False
            state.closing_started_unix = 0.0
            state.closing_report_task_id = ""
            # Reset persisted crash_count so a fresh resume isn't immediately tripped into "emergency".
            state.crash_count = 0
            # Reset start_ts to now so resume budget isn't seen as already-over-budget by the LLM.
            from datetime import datetime, timezone

            state.start_ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            state.save(session_dir)
            override_note = " (--force-resume override)" if force_resume and prior_stop in gated_terminal else ""
            print(f"  → cleared stop_reason and reset crash_count (was {prior_crash}) for fresh resume{override_note}")
            print(f"  → reset start_ts to {state.start_ts} (resume budget)")
        # Re-bootstrap the Cortex KB client (recreates client + reruns T0 warm-start); resume=True is banner-only.
        cortex_client = _bootstrap_cortex_kb(
            args,
            session_dir=session_dir,
            manifest=manifest,
            resume=True,
        )
        # KnowledgePlane facade (fail-soft degrades when PR Monitor/Cortex unreachable); None only when --degraded-kb.
        knowledge_plane = (
            None
            if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )
        # No resume backfill needed for roofline (PR #321: roofline_snapshots restored by SharedState.from_dict).
    else:
        # Resolve model path: --model > $MODEL_PATH; fail fast rather than silently use the YAML hardcoded model.
        if not args.model:
            args.model = os.environ.get("MODEL_PATH") or ""
        if not args.model:
            print(
                "ERROR: model is required. Pass --model <path> or set "
                "MODEL_PATH env (or use --resume to continue an existing "
                "session at the canonical session_dir).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Re-export so subprocess executors inject the resolved model into the Magpie YAML, not its hardcoded model.
        os.environ["MODEL_PATH"] = str(args.model)

        # Quantization prelude (one-shot, before any session/baseline work):
        # if --quantize was passed, quantize the source model now and rewrite
        # args.model to the exported quantized model. No-op otherwise.
        await _run_quantization_prelude(args)

        # Resolve framework: --framework > $FRAMEWORK > "sglang" (session-wide; no framework mixing).
        framework = (
            (args.framework or os.environ.get("FRAMEWORK", "")).strip().lower()
            or framework_registry.DEFAULT_FRAMEWORK
        )
        if not framework_registry.is_supported(framework):
            print(
                f"ERROR: --framework must be one of "
                f"{', '.join(framework_registry.names())} "
                f"(got {framework!r}); set $FRAMEWORK accordingly or pass "
                "--framework",
                file=sys.stderr,
            )
            sys.exit(2)
        _enforce_expected_framework(framework)
        os.environ["FRAMEWORK"] = framework
        print(f"Framework       : {framework}")

        # B3: --framework atom auto-tightens incompatible phases (see _apply_atom_auto_tighten).
        if framework == "atom":
            _apply_atom_auto_tighten(args)

        # Resolve real target GPU: probe > --gpu-type hint; probe wins to catch wrong-host typos that corrupt KB.
        user_specified = (args.gpu_type or os.environ.get("GPU_TYPE", "")).strip().lower()
        probed = _autodetect_gpu_type() or ""
        gpu_type, gpu_warnings = _resolve_gpu_type(
            user_specified=user_specified,
            probed=probed,
        )
        for line in gpu_warnings:
            print(line, file=sys.stderr)
        if probed and not user_specified:
            print(f"GPU type        : {gpu_type} (auto-detected)")
        runner_gpu_type = _gpu_runner_type(gpu_type)
        if gpu_type and runner_gpu_type != gpu_type:
            print(
                f"WARN: {gpu_type} uses {runner_gpu_type} as Magpie "
                f"runner_type (same gfx942/CDNA3 arch; Magpie has no "
                f"sglang_{gpu_type}.sh / vllm_{gpu_type}.sh yet)",
                file=sys.stderr,
            )
        args.gpu_type = gpu_type or None
        if runner_gpu_type:
            os.environ["TARGET_GPU_TYPE"] = gpu_type
            os.environ["GPU_TYPE"] = runner_gpu_type
            print(f"GPU type        : {gpu_type}")
            print(f"Magpie runner   : {runner_gpu_type} (will inject runner_type into Magpie YAML)")
        else:
            os.environ.pop("TARGET_GPU_TYPE", None)
            os.environ.pop("GPU_TYPE", None)
            args.gpu_type = None
            print("GPU type        : <unset> (Magpie will auto-detect)")

        # MAX_MODEL_LEN is operator-overridable. Auto resolution only runs when
        # neither --max-model-len nor $MAX_MODEL_LEN was supplied.
        max_model_len, max_model_len_source = _resolve_run_max_model_len(args)
        args.max_model_len = max_model_len
        os.environ["MAX_MODEL_LEN"] = str(max_model_len)
        os.environ["ISL"] = str(args.isl)
        os.environ["OSL"] = str(args.osl)
        # Profile-scoped OSL (issue #571): exported only when explicitly set, so
        # the profile/roofline materializer can decouple its OSL from the served
        # workload. Unset leaves the profile phase on the global --osl.
        if getattr(args, "profile_osl", None) is not None:
            os.environ["PROFILE_OSL"] = str(args.profile_osl)
        os.environ["PRECISION"] = args.precision
        # Mirror resolved framework_version into env (explicit > auto-detect > unset; see _resolve_framework_version).
        _fw_version_for_env = (getattr(args, "framework_version", None) or "").strip() or (
            os.environ.get("FRAMEWORK_VERSION", "") or ""
        ).strip()
        if not _fw_version_for_env:
            from ..recipe_snapshot_constants import (
                DEFAULT_FRAMEWORK_VERSION_SLUG,
                detect_framework_version,
            )

            _detected = detect_framework_version(
                (getattr(args, "framework", None) or "").strip() or os.environ.get("FRAMEWORK", "")
            )
            if _detected and _detected != DEFAULT_FRAMEWORK_VERSION_SLUG:
                _fw_version_for_env = _detected
        if _fw_version_for_env:
            os.environ["FRAMEWORK_VERSION"] = _fw_version_for_env
        print(
            f"Workload        : ISL={args.isl} OSL={args.osl} "
            f"MAX_MODEL_LEN={max_model_len} ({max_model_len_source}) "
            f"PRECISION={args.precision} "
            f"FRAMEWORK_VERSION={_fw_version_for_env or '<unset>'}"
        )

        # session_dir defaults to <workspace_root>/<model>/<UTC ts>/ (INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat for legacy).
        # Use the resolved identity so a quantized run is named after the source
        # model (e.g. "<model>-quantized") instead of the generic export-dir
        # basename "quantized".
        session_dir = make_session_dir(model_name=resolve_model_display_name(args))
        # Single-optimizer guard (issue #592): a fresh per-session dir is
        # normally uncontended, but take the lock here too so the contract
        # ("one optimizer owns a session") holds uniformly and the owner pid is
        # published for the robustness monitor.
        session_lock = _acquire_session_lock_or_exit(session_dir)
        manifest = write_manifest(session_dir, args=args)
        # One-shot Langfuse startup marker: ties the WekaFS user dir + session
        # dir to code_revision/dependency commits the moment the session is
        # created, so a run that aborts in pre-flight or is killed before a
        # breakdown still leaves a correlatable trace. Best-effort, never fatal.
        try:
            from hyperloom.orchestrator.trace.langfuse_emitter import record_session_start

            record_session_start(session_dir)
        except Exception:  # noqa: BLE001 — startup marker must never break launch
            log.debug("langfuse record_session_start failed (non-fatal)", exc_info=True)
        print(f"Session dir     : {session_dir}")
        print(f"Session id      : {manifest['session_id']}  (manifest label only)")
        _print_session_skeleton(session_dir)

        # Machine-readable launch info: stable point for launcher scripts to harvest pid/session_dir/run_log.
        _emit_launch_info(
            pid=os.getpid(),
            session_dir=session_dir,
            session_id=str(manifest["session_id"]),
            run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
            gpu_type=gpu_type or "",
            framework=args.framework or "",
            model=str(args.model) if args.model else "",
            launch_info_file=getattr(args, "launch_info_file", None),
        )
        _seed_shared_state(
            session_dir,
            args,
            session_id=manifest["session_id"],
        )
        # Unsupported-model preflight: reject multimodal/vision configs (runs after seed, before heavy bring-up).
        if _preflight_unsupported_model_arch(args, session_dir):
            sys.exit(2)
        # Model-config compatibility preflight: reject statically-broken
        # configs before the heavy server bring-up.
        if _preflight_model_config_compat(args, session_dir):
            sys.exit(2)
        # Context-window preflight: reject when ISL+OSL+headroom exceeds max_position_embeddings (no stretch by policy).
        if _preflight_context_window(args, session_dir):
            sys.exit(2)
        # Cortex KB T0 anchor (after seed for recipe_canonical_id, before Coordinator); fails fast unless --degraded-kb.
        cortex_client = _bootstrap_cortex_kb(
            args,
            session_dir=session_dir,
            manifest=manifest,
            resume=False,
        )
        # KnowledgePlane facade for specialists (fail-soft both sides; always non-None for dispatch).
        knowledge_plane = (
            None
            if not getattr(args, "cortex_enabled", True)
            else _bootstrap_knowledge_plane(
                args,
                cortex_client=cortex_client,
                session_dir=session_dir,
            )
        )

    from ..multi_node.state_paths import bind_state_file_to_session

    bind_state_file_to_session(session_dir)
    if nodes_resolved >= 2:
        await asyncio.to_thread(_provision_multi_node_rayjob_stack, args)

    objective = build_objective(
        {
            "MAX_HOURS": str(args.max_hours),
            "TARGET_GAIN_PCT": str(args.target_gain) if args.target_gain else "",
            "TARGET_TPUT_PER_GPU": str(args.target_tput) if args.target_tput else "",
            "TARGET_DIR": args.target_baseline_dir or "",
        }
    )
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
    no_kernel = getattr(args, "no_kernel", False)
    no_explore = getattr(args, "no_explore", False)
    no_framework_agent = bool(getattr(args, "no_framework_agent", False))
    # Unconditional phase-toggle banner lines (mirror the kernel banner so all
    # three --no-xxx flags surface their ENABLED/DISABLED state at startup).
    if no_explore:
        print(
            "Explore phase   : DISABLED (--no-explore); "
            f"{'baseline -> SWEEP' if no_kernel else 'baseline -> KERNEL -> SWEEP'}"
        )
    else:
        print("Explore phase   : ENABLED")
    if no_framework_agent:
        print("Framework-agent phase : DISABLED (--no-framework-agent)")
    else:
        print("Framework-agent phase : ENABLED")
    if no_explore and no_kernel:
        print(
            "WARNING: --no-explore and --no-kernel are both set; the run "
            "collapses to baseline -> SWEEP over an empty optimization_stack "
            "(no EXPLORE param search, no KERNEL rewrites). SWEEP only "
            "re-validates the baseline recipe. Continuing as requested.",
            file=sys.stderr,
        )
    if bool(getattr(args, "research_scout", True)):
        print(
            "Research scout  : ENABLED at PRELUDE (re-dispatch every "
            f"{max(1, int(getattr(args, 'research_scout_interval', 3) or 3))} "
            "explore rounds)"
        )
    else:
        print("Research scout  : DISABLED (--no-research-scout)")
    if bool(getattr(args, "target_advisory", True)):
        print("Target advisory : ENABLED (External target gap injected into prompts; advisory-only)")
    else:
        print("Target advisory : DISABLED (--no-target-advisory)")
    if bool(getattr(args, "recipe_sediment", True)):
        print("Recipe sediment : ENABLED (KEEP/REVERT provenance written to persistent recipe)")
    else:
        print("Recipe sediment : DISABLED (--no-recipe-sediment)")
    if bool(getattr(args, "allow_empty_kernel_shape", False)):
        os.environ["HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE"] = "1"
        print("Kernel shape    : empty-shape dispatch ALLOWED (--allow-empty-kernel-shape)")
    else:
        os.environ.pop("HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE", None)
        print("Kernel shape    : non-empty trace shape REQUIRED for kernel-opt dispatch")

    # Resolve critic backend + runtime root before _build_backends; abort rc=2 if --critic-agent runtime unreachable.
    critic_choice = _resolve_critic_choice(args)
    critic_agent_root: Path | None = None
    critic_kb_mode = os.environ.get("CRITIC_KB_CLIENT_MODE", "inmemory").lower()
    if critic_kb_mode not in ("inmemory", "live"):
        print(
            f"ERROR: CRITIC_KB_CLIENT_MODE={critic_kb_mode!r} not in {{'inmemory','live'}}",
            file=sys.stderr,
        )
        sys.exit(2)
    if critic_choice == "agent":
        critic_agent_root = _resolve_critic_agent_root()
        if critic_agent_root is None:
            print(
                f"ERROR: --critic-agent selected but critic-agent runtime not "
                f"found.\n"
                f"  Set ${_CRITIC_AGENT_ROOT_ENV} to the directory containing "
                f"runtime/cli.py, or check the "
                f"src/hyperloom/agents/critic/ install.\n"
                f"  Bypass with --critic-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_critic_agent_runtime(critic_agent_root)
        if critic_kb_mode == "live" and not os.environ.get("KB_BASE_URL"):
            print(
                "ERROR: CRITIC_KB_CLIENT_MODE=live but KB_BASE_URL is not "
                "set. Either export KB_BASE_URL or unset "
                "CRITIC_KB_CLIENT_MODE to fall back to inmemory.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Default WORKSPACE_PATH for critic-agent runtime: SKILL static-asset root (repo root), not artefact dir.
        os.environ.setdefault("WORKSPACE_PATH", str(Path(__file__).resolve().parents[4]))

    # Resolve robustness backend choice + runtime root, mirroring critic.
    robustness_choice = _resolve_robustness_choice(args)
    robustness_agent_root: Path | None = None
    robustness_options = _build_robustness_options(args)
    if robustness_choice == "agent":
        robustness_agent_root = _resolve_robustness_agent_root()
        if robustness_agent_root is None:
            print(
                f"ERROR: --robustness-agent selected but robustness-agent "
                f"runtime not found.\n"
                f"  Set ${_ROBUSTNESS_AGENT_ROOT_ENV} to the directory "
                f"containing src/robustness_agent/runtime/cli.py, or install "
                f"robustness-agent at $REPO_ROOT/robustness-agent/.\n"
                f"  Bypass with --robustness-mock.",
                file=sys.stderr,
            )
            sys.exit(2)
        _validate_robustness_agent_runtime(robustness_agent_root)

    backends = _build_backends(
        claude_model=args.claude_model,
        codex_model=args.codex_model,
        kernel_codex=args.kernel_codex,
        critic_choice=critic_choice,
        session_dir=session_dir,
        critic_agent_root=critic_agent_root,
        critic_kb_mode=critic_kb_mode,
        cortex_kb_url=(getattr(args, "cortex_kb_url", None) or "").strip() or None,
        robustness_choice=robustness_choice,
        robustness_agent_root=robustness_agent_root,
        robustness_options=robustness_options,
        no_kernel=no_kernel,
    )
    # Expose active session_dir to in-process executors via the canonical
    # pin env var (read by paths.session_dir() -> report.py, sweep.py, etc.).
    # Note: make_session_dir() already sets this, but we reinforce it here
    # for --resume paths where make_session_dir may not have been called.
    # DO NOT overwrite USER_DATA_PATH — it must remain the workspace root
    # so concurrent sessions, install.sh, and setup_env.sh resolution work
    # correctly on shared filesystems (WekaFS).
    os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] = str(session_dir)
    # Production: enable strict PolicyGate path-containment (escaping intents land as policy_denied).
    os.environ["INFERENCE_OPTIMIZER_STRICT_PATHS"] = "1"
    # PolicyGate R1 phase_incompatible enforcement for production runs (env affects cli boot path only).
    if getattr(args, "strict_phase", True):
        os.environ["INFERENCE_OPTIMIZER_STRICT_PHASE"] = "1"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_STRICT_PHASE", None)
    # Propagate --legacy-action-scores so SharedState.from_dict handles drop/warn uniformly (default drop).
    legacy_mode = (
        str(
            getattr(args, "legacy_action_scores", "drop") or "drop",
        )
        .strip()
        .lower()
    )
    if legacy_mode == "warn":
        os.environ["INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES"] = "warn"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", None)
    # Propagate --migration-mode: SharedState.from_dict treats fact-layer discrepancy as fatal (strict) or WARN (lenient).
    migration_mode = (
        str(
            getattr(args, "migration_mode", "strict") or "strict",
        )
        .strip()
        .lower()
    )
    if migration_mode == "lenient":
        os.environ["INFERENCE_OPTIMIZER_MIGRATION_MODE"] = "lenient"
    else:
        os.environ.pop("INFERENCE_OPTIMIZER_MIGRATION_MODE", None)
    # --reset-state backs up state.json and starts blank, before Coordinator is constructed.
    if getattr(args, "reset_state", False):
        _reset_state_file(session_dir)
    # Propagate --breakdown-include-transcripts (inline / path-only choice) to end-of-session breakdown.
    transcripts_flag = (
        str(
            getattr(args, "breakdown_include_transcripts", "false") or "false",
        )
        .strip()
        .lower()
    )
    if transcripts_flag == "true":
        os.environ["INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS"] = "1"
    else:
        os.environ.pop(
            "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS",
            None,
        )

    # Build phase budget pct dict from CLI flags; absent values fall back to Coordinator library defaults.
    phase_budget_pct = _build_phase_budget_pct(args)

    # When kernel is disabled, strip it from the role registry (no tick / no backend expectation).
    role_registry = None
    if no_kernel:
        from hyperloom.orchestrator.roles.agent_role import default_role_registry

        role_registry = {k: v for k, v in default_role_registry().items() if k != "kernel_agent"}

    coordinator = Coordinator(
        session_dir,
        backends=backends,
        role_registry=role_registry,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        model_class=(getattr(args, "model_class", None) or os.environ.get("MODEL_CLASS") or ""),
        cortex_kb=cortex_client,
        phase_budget_pct=phase_budget_pct or None,
        # KnowledgePlane facade (None when --degraded-kb).
        knowledge_plane=knowledge_plane,
        # Advisory multi-model specialist-proposal scorer. ``None`` when
        # --no-proposal-scoring or an empty model list; otherwise scores
        # each proposal_set and surfaces the results to Orchestration as
        # one reference among many (never gates anything). ``session_dir``
        # is forwarded so the scorer can append its per-model token usage
        # to the full-trace ledger (component=proposal_scorer).
        proposal_scorer=_build_proposal_scorer(args, session_dir),
        # Warm-recipe replay controls. Default ON, fires when
        # warm_start_recipe.confidence >= min_confidence and the
        # measured gain reproduces at least min_reproduce_pct of the
        # recipe's historical claim. Manifest is the persistent
        # authority across restarts (resume-safe).
        warm_replay_enabled=_resume_safe_flag(
            args,
            "no_warm_replay",
            manifest,
            "warm_replay_enabled",
            default=True,
            invert=True,
        ),
        warm_replay_min_confidence=_resume_safe_numeric(
            args,
            "warm_replay_min_confidence",
            manifest,
            "warm_replay_min_confidence",
            default=0.7,
        ),
        warm_replay_min_reproduce_pct=_resume_safe_numeric(
            args,
            "warm_replay_min_reproduce_pct",
            manifest,
            "warm_replay_min_reproduce_pct",
            default=0.8,
        ),
    )
    framework_for_prompt = os.environ.get("FRAMEWORK", "").strip().lower() or "sglang"
    max_minutes_for_prompt = int(round(float(args.max_hours) * 60))
    prompts: dict[str, str] = {
        "orchestration": args.orch_prompt
        or _build_orchestration_prompt(
            no_kernel=no_kernel,
            no_explore=no_explore,
            no_framework_agent=bool(getattr(args, "no_framework_agent", False)),
            framework=framework_for_prompt,
            objective=objective,
            max_minutes=max_minutes_for_prompt,
        ),
        "critic": args.critic_prompt or _load_critic_prompt(),
    }
    if not no_kernel:
        prompts["kernel_agent"] = args.kernel_prompt or _DEFAULT_KERNEL_PROMPT
    coordinator.system_prompt_overrides = prompts
    # ``fa phase-discover`` timeout override (falsy -> DEFAULT_FA_PHASE_TIMEOUT_SEC 180s).
    try:
        coordinator.framework_agent_discover_timeout_sec = float(
            getattr(args, "framework_agent_discover_timeout_sec", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        coordinator.framework_agent_discover_timeout_sec = 0.0
    # Build specialist executor only when research_lane capacity > 0 (0 degrades to LLM-direct grid).
    specialist_capacity = int(getattr(args, "research_lane_capacity", 1) or 0)
    specialist_executor: "Any" = None
    if specialist_capacity > 0:
        specialist_executor = _build_specialist_executor(
            args,
            session_dir=session_dir,
            knowledge_plane=knowledge_plane,
        )
    _register_executors(
        coordinator,
        no_kernel=no_kernel,
        compare_against_gpu=getattr(args, "compare_against_gpu", None),
        session_dir=session_dir,
        specialist_executor=specialist_executor,
    )
    # Persist effective system prompts for resume / drift inspection.
    _snapshot_system_prompts(session_dir, prompts=prompts)

    kernel_str = "DISABLED" if no_kernel else (f"{'Codex' if args.kernel_codex else 'Claude'}")
    if critic_choice == "mock":
        critic_str = "mock"
    else:  # "agent"
        critic_str = f"critic-agent(kb={critic_kb_mode}, codex={args.codex_model}, root={critic_agent_root})"
    if robustness_choice == "mock":
        robustness_str = "mock"
    else:
        robustness_str = f"robustness-agent(root={robustness_agent_root})"
        if robustness_options:
            kvs = ",".join(f"{k}={v!r}" for k, v in sorted(robustness_options.items()))
            robustness_str += f"[{kvs}]"
    print(
        f"Backends        : "
        f"orchestration=Claude({args.claude_model}), "
        f"kernel={kernel_str}, "
        f"critic={critic_str}, "
        f"robustness={robustness_str}"
    )
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} (budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()

    if not (getattr(args, "compare_against_gpu", None) or "").strip():
        print(
            "[target_analysis] no --compare-against-gpu set; will write a "
            "marker JSON at $SESSION_DIR/target_analysis/target_baseline.json "
            "(reason=no_target_gpu_configured) — set --compare-against-gpu "
            "to fetch real InferenceX reference data.",
            file=sys.stderr,
        )

    try:
        stop_reason = await coordinator.run(
            objective=objective,
            max_minutes=args.max_hours * 60.0,
            tick_interval_sec=args.tick_interval_sec,
            max_ticks=args.max_ticks,
            install_signal_handlers=True,
            closing_grace_sec=args.closing_grace_sec,
        )
    finally:
        await coordinator.stop()
        # Drop the single-optimizer session lock once the
        # coordinator has released its leases. The OS would drop it on process
        # exit anyway; this just frees it promptly for an intentional resume.
        session_lock.release()
        # Crash-safe reports/final.json. Runs unconditionally and
        # FIRST so a consumable machine-readable summary always exists even
        # when the CLOSE sequencer never ran (time_exhausted / external
        # SIGTERM) or its report task failed. Idempotent: a no-op when the
        # full ReportExecutor already wrote final.json. coordinator.run's own
        # finally has persisted state.json (with stop_reason) before we get
        # here, so the fields are current.
        try:
            from ..breakdown import write_minimal_final_json

            final_json = write_minimal_final_json(session_dir)
            print(f"Final summary     : {final_json}")
        except Exception:  # noqa: BLE001 — safety net must never mask stop_reason
            log.exception("crash-safe final.json write failed (non-fatal)")
        # End-of-session safety net: always materialize session_breakdown.json (best-effort; never mask stop_reason).
        # Skip when the CLOSE sequencer already wrote it (close_sequence_done is locked in CORE_STATE_FIELDS).
        sequencer_done = getattr(
            coordinator.shared_state,
            "close_sequence_done",
            False,
        )
        if sequencer_done:
            print(
                "Session breakdown : (already written by CLOSE phase sequencer; skipping cli.finally safety-net write)"
            )
            # The sequencer already flushed Langfuse + packaged at step 2.5/2.6.
            # Re-run flush idempotently as a safety net (only re-writes the
            # receipt if it already ran), so a step-2.5 failure still gets a
            # final receipt spliced in.
            try:
                from hyperloom.orchestrator.trace.langfuse_emitter import (
                    flush_session,
                    record_session_breakdown,
                )

                flush_session(session_dir)
                from ..breakdown import patch_breakdown_langfuse

                patch_breakdown_langfuse(session_dir)
                record_session_breakdown(session_dir)
            except Exception:  # noqa: BLE001
                log.debug("langfuse flush_session (post-sequencer) failed", exc_info=True)
        else:
            try:
                from ..breakdown import write_breakdown_json

                breakdown_path = write_breakdown_json(session_dir)
                print(f"Session breakdown : {breakdown_path}")
            except Exception:  # noqa: BLE001
                log.exception("session_breakdown finalize failed (non-fatal)")
            # Safety-net reports/final.md write (no-op when the sequencer's final.md already exists).
            try:
                from ..breakdown import write_minimal_final_report

                final_md = write_minimal_final_report(session_dir)
                print(f"Final report      : {final_md}")
            except Exception:  # noqa: BLE001
                log.exception("emergency final report write failed (non-fatal)")
            # Live Langfuse push (opt-in, default off): reconcile + flush,
            # then splice the post-flush receipt (final counts) into the
            # session_breakdown.json langfuse section (written above with only
            # the pre-flush in-process counts). MUST run BEFORE the artifact
            # package below, so the bundled SBD carries counts_final=true and
            # the bundle includes the final langfuse_receipt.json. No-op unless
            # HYPERLOOM_LANGFUSE_ENABLE + LANGFUSE_* are set; idempotent if the
            # CLOSE sequencer already flushed (re-writes the receipt only).
            try:
                from hyperloom.orchestrator.trace.langfuse_emitter import (
                    flush_session,
                    record_session_breakdown,
                )

                flush_session(session_dir)
                from ..breakdown import patch_breakdown_langfuse

                patch_breakdown_langfuse(session_dir)
                record_session_breakdown(session_dir)
            except Exception:  # noqa: BLE001
                log.debug("langfuse flush_session failed (non-fatal)", exc_info=True)

        # Safety-net artifact package -> /workspace. The CLOSE phase
        # sequencer normally packages at step 2.6, but the wall-clock
        # deadline path (_enter_closing_phase) and crash paths leave
        # close_sequence_done False and never run the sequencer, so the
        # bundle would be missing without this. Best-effort: failures
        # must not mask stop_reason. Runs after the SBD/final.md +
        # Langfuse flush above so the freshest products are bundled.
        try:
            from ..breakdown import package_session_artifacts

            pkg_path = package_session_artifacts(
                session_dir,
                session_id=str(
                    getattr(coordinator.shared_state, "session_id", "") or "",
                ),
            )
            if pkg_path is not None:
                print(f"Artifact package  : {pkg_path}")
        except Exception:  # noqa: BLE001
            log.exception("session artifact package failed (non-fatal)")

    _reconcile_crash_count(coordinator.shared_state, session_dir)
    # NOTE: conc_sweep is now a SWEEP-phase action auto-enqueued by the Coordinator, not a post-hook here.

    _print_final_summary(coordinator.shared_state, stop_reason, session_dir)
    return (
        0
        if stop_reason
        in (
            "target_reached",
            "global_converged",
            "time_exhausted",
            "max_ticks",
        )
        else 1
    )








def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments and dispatch the requested subcommand.

    Configures logging from the ``-v`` count, resolves any ``--*-prompt`` flag
    that points at a file (reading its contents in place), and runs the
    ``optimize`` subcommand via :func:`asyncio.run`. Prints help and returns a
    non-zero code for unknown commands.

    Args:
        argv (list[str] | None): Argument vector to parse; defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        int: The process exit code (``optimize`` result, or ``2`` for no/unknown
        command).
    """
    # Force line-buffering so output piped through `tee` (or any
    # non-TTY sink) flushes every line immediately instead of
    # block-buffering ~8 KB.  Without this the top-level log appears
    # frozen for the entire duration of a Magpie subprocess (~30-60 min)
    # because no new print() calls happen while communicate() blocks.
    # See #468.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = _build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if args.command == "optimize":
        # Resolve any --*-prompt that point at a file.
        for attr in ("orch_prompt", "critic_prompt", "kernel_prompt"):
            v = getattr(args, attr)
            if v and Path(v).exists():
                setattr(args, attr, Path(v).read_text(encoding="utf-8"))
        return asyncio.run(_run_optimize(args))
    if args.command == "recover-session":
        return _run_recover_session(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
