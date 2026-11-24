# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run()."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom_common.llm import llm_config
from hyperloom_common.env import EnvValueError, env_bool
from hyperloom.common import agent_isolation
from hyperloom.common.gpu_partition import PartitionModel
from .lineup_banners import print_lineup_banners
from .kb import (
    _bootstrap_recipe_kb,
    _bootstrap_knowledge_plane,
)
from .backends import (
    CriticSetup,
    ModelRouting,
    _build_backends,
    _orchestration_banner,
    _resolve_orchestration_backend,
)
from .session_gates import _check_codex_sandbox, _preflight_agentx_backend
from .model_gate import (
    _finish_model_gate,
    _preflight_context_window,
    _preflight_model_config_compat,
    _preflight_unsupported_model_arch,
    _start_model_gate,
)
from ..gpu_types import (
    _autodetect_gpu_type,
    _resolve_amd_gpu_type,
    _resolve_gpu_type,
)
from .benchmark_config import bind_benchmark, configure_benchmark
from .bootstrap import (
    _emit_launch_info,
    _install_executors,
    _print_final_summary,
    _print_session_skeleton,
    _reconcile_crash_count,
    _seed_shared_state,
    _snapshot_system_prompts,
    constraint_budget_resume_conflict,
    fresh_measurement_fields,
    latency_budget_scope_error,
    LaunchInfo,
    measurement_state_is_stale,
    open_plug_host,
    resolve_framework_version,
    resolve_launch_gpu_power,
    resolve_model_display_name,
    resolve_plug_manifest,
    resolve_session_base,
    resume_gpu_power_settings,
    running_plug_host,
    SettledLaunch,
    warn_without_target_gpu,
)
from hyperloom.orchestrator.actions.executors._aiter_jit import clean_stale_aiter_locks
from hyperloom.common.perf_metric import agentx_enabled as _agentx_enabled

from .credentials import (
    _CRITIC_AGENT_ROOT_ENV as _CRITIC_AGENT_ROOT_ENV,
    _resolve_agent_root as _resolve_agent_root,
    _validate_agent_runtime as _validate_agent_runtime,
)
from .multi_node import (
    _prepare_multi_node_state as _prepare_multi_node_state,
    _resolve_mn_backend,
)
from .model_catalog import (
    _resolve_critic_choice,
    _resolve_models_for_run,
)
from .launch_shape import (
    _export_operator_launch_shape,
    _launch_pins,
    _resolve_precision,
    _resolve_run_max_model_len,
    _resolve_run_topology,
)
from .quantization import (
    _run_quantization_prelude as _run_quantization_prelude,
)
from .recover import (
    _run_recover_session as _run_recover_session,
)


__all__ = ["main"]
from ..session.manifest import write_manifest
from ..protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator, CoordinatorSettings, RunOptions
from hyperloom.orchestrator.collaborator import OrchestrationPrompt as _OrchestrationPrompt
from hyperloom.inference_optimizer.framework_paths import resolve_framework_tree, resolve_kernel_search_roots
from hyperloom.orchestrator.state.objective import AnyObjective, Objective, build_objective
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.prompts.prompt_builder import (
    TRANSPORT_TOOLS,
    build_orchestration_prompt,
    default_enabled_actions,
    PromptAssets,
    PromptCycle,
    SessionContext,
)
from hyperloom.inference_optimizer.breakdown.stop_reasons import SUPERVISOR_RESTART_REASON

from ..session.lock import SessionAlreadyRunning, SessionLock, SessionLockPathError
from ..session.paths import ENV_USER_DATA_PATH, asset_system_prompts_dir, make_session_dir


log = logging.getLogger("hyperloom.inference_optimizer.cli")

from .gpu_power_apply import gpu_power_restore_guard
from .stack_dump import dump_thread_stacks_on_sigusr1
from .custom_specialist import (
    CUSTOM_SPECIALIST_ONLY_ENV,
    _custom_specialist_conflict as _custom_specialist_conflict,
    _load_custom_specialist_args as _load_custom_specialist_args,
)
from .parser import (
    _build_parser as _build_parser,
)
from .install_event import _mark_pending_install_event_failed, _persist_install_event
from .preflight import (
    _check_gfx_arch_resolvable,
    _preflight as _preflight,
)
from .session_resume import (
    _OperatorLaunchShape,
    _open_next_leg,
    _open_resumed_session,
    _reexport_session_environment,
    _restore_phase_toggles,
)
from .session_setup import (
    _acquire_session_lock_or_exit,
    _apply_atom_auto_tighten,
    _apply_default_budget,
    _apply_operator_supplied_paths,
    _enforce_expected_framework,
    _export_partition_shape,
    _publish_gpu_power_settings,
    _registered_framework_or_exit,
    _require_custom_entrypoint,
    _restore_partition_shape_from_state,
    _SessionStart,
    PartitionRequest,
)


def _orchestration_rules_fragment_path() -> Path:
    """Path to the rules-only ``orchestration.md`` fragment consumed by ``prompt_builder``."""
    return asset_system_prompts_dir() / "orchestration.md"


def _objective_summary_for_prompt(objective: Objective) -> tuple[str, float | str | None]:
    """Summarise an objective into the ``(kind, value)`` pair the prompt expects."""
    kind = objective.kind()
    if isinstance(objective, AnyObjective):
        return kind, objective.describe()
    value: float | str | None = None
    if hasattr(objective, "target_gain_pct"):
        value = float(getattr(objective, "target_gain_pct"))
    elif hasattr(objective, "target_tput_per_gpu"):
        value = float(getattr(objective, "target_tput_per_gpu"))
    elif hasattr(objective, "target_within_pct"):
        value = float(getattr(objective, "target_within_pct"))
    elif hasattr(objective, "baseline_dir"):
        value = str(getattr(objective, "baseline_dir"))
    return kind, value


@dataclass(frozen=True)
class OrchestrationPromptContext:
    """What the Orchestration prompt says about the session itself; fixed for the run while cycles and phases move."""

    no_kernel: bool
    framework: str
    objective: Objective
    max_minutes: int
    no_framework_agent: bool = False
    transport: str = TRANSPORT_TOOLS
    benchmark_mode: str = ""
    agentx_corpus_shape: Mapping[str, Any] | None = None
    agentx_grading: Mapping[str, Any] | None = None
    agentx_backend: str = ""


def _build_orchestration_prompt(
    context: OrchestrationPromptContext,
    *,
    macro_cycle: int = 0,
    cycle_directive: str = "",
    cycle_strategy: Mapping[str, Any] | None = None,
    phase: str = "",
) -> str:
    """Compose the Orchestration system prompt from typed inputs (``--orch-prompt`` overrides)."""
    no_kernel, no_framework_agent, framework = context.no_kernel, context.no_framework_agent, context.framework
    enabled = default_enabled_actions(no_kernel=no_kernel, no_optimize=no_framework_agent)
    kind, value = _objective_summary_for_prompt(context.objective)
    return build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=enabled,
        session=SessionContext(
            framework=framework,
            framework_agent_phase_enabled=not no_framework_agent,
            objective_kind=kind,
            objective_value=value,
            max_minutes=int(context.max_minutes),
            framework_source_roots=resolve_kernel_search_roots(),
            session_framework_tree=resolve_framework_tree(framework),
            transport=context.transport,
            benchmark_mode=context.benchmark_mode,
            agentx_corpus_shape=context.agentx_corpus_shape,
            agentx_grading=context.agentx_grading,
            agentx_backend=context.agentx_backend,
        ),
        cycle=PromptCycle(
            macro_cycle=int(macro_cycle), cycle_directive=cycle_directive, cycle_strategy=cycle_strategy, phase=phase
        ),
        assets=PromptAssets(rules_fragment_path=_orchestration_rules_fragment_path()),
    )


def _load_critic_prompt() -> str:
    """Return the Critic system prompt sourced from ``orchestrator/prompts/critic.md``."""
    return (asset_system_prompts_dir() / "critic.md").read_text(encoding="utf-8")


def _reset_state_file(session_dir: Path, state: SharedState) -> None:
    """Back up ``state.json`` to ``state.json.preReset.<unix_ts>`` and start fresh (Recipe KB untouched).

    The fresh state keeps ``framework`` and the measurement identity of *state*: the session is single-framework and
    keeps its measurement path, nothing after this point re-derives either, and a later resume checks the latter.
    """
    state_path = session_dir / "state.json"
    if not state_path.exists():
        return
    backup_path = session_dir / f"state.json.preReset.{int(time.time())}"
    try:
        state_path.replace(backup_path)
    except OSError as exc:
        log.warning("--reset-state: could not move %s → %s: %s", state_path, backup_path, exc)
        return
    SharedState(framework=state.framework, **fresh_measurement_fields(state)).save(session_dir)
    log.info(
        "--reset-state: backed up state.json to %s; session starts blank except for framework=%s.",
        backup_path.name,
        state.framework,
    )


def _build_phase_budget_pct(args: argparse.Namespace) -> dict[str, float]:
    """Map ``--*-pct`` CLI flags to a ``phase -> pct`` override dict.

    ENABLEMENT has no flag: nothing enforces a per-phase cap for it.
    """
    from hyperloom.orchestrator.phases.machine_state import (
        PHASE_CLOSE,
        PHASE_FRAMEWORK_AGENT,
        PHASE_KERNEL_AGENT,
        PHASE_PRELUDE,
        PHASE_SWEEP,
    )

    phase_budget_pct: dict[str, float] = {}
    for cli_field, phase_name in (
        ("phase_budget_prelude_pct", PHASE_PRELUDE),
        ("phase_budget_framework_pct", PHASE_FRAMEWORK_AGENT),
        ("phase_budget_kernel_pct", PHASE_KERNEL_AGENT),
        ("phase_budget_sweep_pct", PHASE_SWEEP),
        ("phase_budget_close_pct", PHASE_CLOSE),
    ):
        val = getattr(args, cli_field)
        if val is not None:
            phase_budget_pct[phase_name] = float(val)
    return phase_budget_pct


def _exit_code_for_stop_reason(stop_reason: str | None, baseline_tput: float) -> int:
    """Map a terminal ``stop_reason`` to a process exit code (0 success, 1 failure).

    Reads the same classifier the breakdown grades outcomes against. A second
    copy here would decide CI's verdict on a vocabulary that had drifted from
    the one the report was written from.
    """
    from hyperloom.inference_optimizer.breakdown.stop_reasons import outcome_status

    return 0 if outcome_status(str(stop_reason or ""), baseline_tput) == "completed" else 1


def _write_cli_terminal_artifacts(session_dir: Path, state: SharedState, stop_reason: str | None) -> None:
    """Write the session's terminal artifacts in their established order.

    A watchdog restart ends a leg, not the session, so it produces none of them
    -- including the close-out package, which speaks for a finished run.
    """
    if stop_reason == SUPERVISOR_RESTART_REASON:
        return
    if not state.close_sequence_done:
        from ..breakdown.recorder.close_out import record_close_safety_net

        record_close_safety_net(session_dir)
    try:
        from ..breakdown import write_minimal_final_json

        final_json = write_minimal_final_json(session_dir)
        print(f"Final summary     : {final_json}")
    except (OSError, ValueError):
        log.exception("crash-safe final.json write failed (non-fatal)")
    if state.close_sequence_done:
        print("Session breakdown : (already written by CLOSE phase sequencer; skipping cli.finally safety-net write)")
    else:
        try:
            from ..breakdown import write_breakdown_json

            breakdown_path = write_breakdown_json(session_dir)
            print(f"Session breakdown : {breakdown_path}")
        except (OSError, ValueError, TypeError):
            log.exception("session_breakdown finalize failed (non-fatal)")
        try:
            from ..breakdown import write_minimal_final_report

            final_md = write_minimal_final_report(session_dir)
            print(f"Final report      : {final_md}")
        except (OSError, ValueError, TypeError):
            log.exception("emergency final report write failed (non-fatal)")
    try:
        from hyperloom.inference_optimizer.trace.langfuse_emitter import flush_session, record_session_breakdown

        flush_session(session_dir)
        from ..breakdown import patch_breakdown_langfuse

        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
    except (OSError, ValueError):
        log.debug("langfuse flush_session failed", exc_info=True)
    # Safety net for paths that leave close_sequence_done False and never run
    # the sequencer; ordered after langfuse so the package carries its patch.
    try:
        from ..breakdown import package_session_artifacts

        pkg_path = package_session_artifacts(session_dir, session_id=str(getattr(state, "session_id", "") or ""))
        if pkg_path is not None:
            print(f"Artifact package  : {pkg_path}")
    except (OSError, ValueError):
        log.exception("session artifact package failed (non-fatal)")


def _new_preflight_failure_session_dir(
    args: argparse.Namespace,
    *,
    failed_attempt: bool = False,
) -> Path:
    """Create a standalone session for a failed preflight attempt."""
    model_name = resolve_model_display_name(args)
    if not model_name:
        model_name = Path(os.environ.get("MODEL_PATH", "")).name
    if not model_name:
        resume_from = str(getattr(args, "resume_from", "") or "").strip()
        if resume_from:
            model_name = Path(resume_from).expanduser().parent.name
    if failed_attempt:
        model_name = f"{model_name or 'preflight'}-failed-attempt"
    return make_session_dir(model_name=model_name or "preflight-failure")


def _persist_preflight_failure_artifacts(
    args: argparse.Namespace,
    exc: Exception,
) -> Path | None:
    """Best-effort materialize the failed install event and final SBD."""
    _mark_pending_install_event_failed(args, exc)
    session_dir = _new_preflight_failure_session_dir(
        args,
        failed_attempt=bool(str(getattr(args, "resume_from", "") or "").strip()),
    )

    session_lock = SessionLock(session_dir)
    try:
        session_lock.acquire()
    except (SessionAlreadyRunning, SessionLockPathError, OSError):
        session_lock.release()
        log.warning("failed to lock SBD V6 preflight failure session", exc_info=True)
        return None

    try:
        if not (session_dir / "manifest.json").is_file():
            try:
                manifest_args = argparse.Namespace(**vars(args))
                if not getattr(manifest_args, "model", None):
                    manifest_args.model = os.environ.get("MODEL_PATH", "")
                write_manifest(session_dir, args=manifest_args)
            except (OSError, ValueError, TypeError) as write_exc:
                log.warning("failed to write manifest for SBD V6 preflight failure", exc_info=True)
                from ..session.sbd_v6 import record_write_warning

                record_write_warning(session_dir, component="preflight_failure.manifest", exc=write_exc)

        _persist_install_event(args, session_dir)
        from ..breakdown.recorder.close_out import record_close_safety_net
        from ..breakdown.recorder import record_stage_reached

        record_close_safety_net(session_dir)
        record_stage_reached(session_dir, "install")
        from ..breakdown import write_breakdown_json

        write_breakdown_json(session_dir)
    finally:
        session_lock.release()
    print(f"Preflight failure artifacts: {session_dir}", file=sys.stderr)
    return session_dir


def _start_experience_kb() -> None:
    """Bring the workspace's Experience KB service to serving, then check its Experiences can be collected."""
    from hyperloom_kb import ConfigurationError, LocalServiceError, RemoteClientError
    from hyperloom_kb.collect import MappingError

    from ..experience_collect import validate_config as validate_experience_collection
    from ..experience_kb_service import ensure_service

    # The run must not depend on the Experience KB: reads come back empty and writes are spooled or skipped.
    try:
        ensure_service()
    except (LocalServiceError, RemoteClientError) as exc:
        log.warning("Experience KB service is not serving (%s); Experience writes are spooled until it is", exc)
    try:
        validate_experience_collection()
    except (ConfigurationError, MappingError, RemoteClientError) as exc:
        log.warning("Experience KB cannot take this run's Experiences (%s); the run continues without them", exc)


def _resolve_target_gpu(args: argparse.Namespace, framework: str) -> None:
    """Settle ``args.gpu_type``, then refuse a board this session cannot serve."""
    # Resolve real target GPU: probe > --gpu-type hint; probe wins to catch wrong-host typos that corrupt KB.
    user_specified = _resolve_amd_gpu_type(args.gpu_type) or ""
    if int(args.nodes) >= 2:
        from ..multi_node._internal.gpu_probe import remote_autodetect_gpu_type

        probed = remote_autodetect_gpu_type() or ""
        if probed:
            print(f"GPU probe       : {probed} (remote {_resolve_mn_backend(args)})")
    else:
        probed = _autodetect_gpu_type() or ""
    gpu_type, gpu_warnings = _resolve_gpu_type(user_specified=user_specified, probed=probed)
    for line in gpu_warnings:
        print(line, file=sys.stderr)
    if probed and not user_specified:
        print(f"GPU type        : {gpu_type} (auto-detected)")
    args.gpu_type = gpu_type or None
    if gpu_type:
        os.environ["TARGET_GPU_TYPE"] = os.environ["GPU_TYPE"] = gpu_type
        print(f"GPU type        : {gpu_type}")
    else:
        os.environ.pop("TARGET_GPU_TYPE", None)
        os.environ.pop("GPU_TYPE", None)
        print("GPU type        : <unset>")
    _require_custom_entrypoint(framework, gpu_type=gpu_type)

    # Runs here, not in _preflight, because the question it asks -- will provenance be able to name the ISA? -- is
    # unanswerable until args.gpu_type is final.
    _check_gfx_arch_resolvable(args.gpu_type)


def _create_fresh_session(args: argparse.Namespace, *, framework: str, nodes_resolved: int) -> _SessionStart:
    """Create and lock the session directory, resolve its base launch, then write its manifest, launch info and state."""
    session_dir = make_session_dir(
        model_name=resolve_model_display_name(args), target=getattr(args, "session_dir", None)
    )
    # Single-optimizer guard: take the lock so the contract holds uniformly and the owner pid is published for the
    # session diagnostics.
    session_lock = _acquire_session_lock_or_exit(session_dir)
    # The base settles an open TP and EP, which the bound AgentX workload YAML records.
    base = resolve_session_base(session_dir, args, framework=framework)
    bind_benchmark(session_dir, args)
    manifest = write_manifest(session_dir, args=args)
    _persist_install_event(args, session_dir)
    # One-shot Langfuse startup marker so a run killed before a breakdown still leaves a correlatable trace.
    from hyperloom.inference_optimizer.trace.langfuse_emitter import record_session_start

    record_session_start(session_dir)
    print(f"Session dir     : {session_dir}")
    print(f"Session id      : {manifest['session_id']}  (manifest label only)")
    _print_session_skeleton(session_dir)

    # Machine-readable launch info: stable point for launcher scripts to harvest pid/session_dir/run_log.
    info = LaunchInfo(
        pid=os.getpid(),
        session_dir=session_dir,
        session_id=str(manifest["session_id"]),
        run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
        gpu_type=args.gpu_type or "",
        framework=args.framework or "",
        model=str(args.model),
    )
    _emit_launch_info(info, launch_info_file=getattr(args, "launch_info_file", None))
    # Placed here, not at the top of _run_optimize, because everything it weighs is resolved by now and none of it
    # was then: the framework (whether anything fans out), args.gpu_type (the CU fallback), and args.model, which
    # --quantize rewrites to the exported checkpoint -- sizing partitions against the source model would weigh the
    # wrong weights.
    compute_partition = _export_partition_shape(
        PartitionRequest(getattr(args, "compute_partition_mode", None), getattr(args, "streams_per_partition", None)),
        framework=framework,
        gpu_type=args.gpu_type,
        nodes=nodes_resolved,
        model=PartitionModel(str(args.model), getattr(args, "precision", None)),
    )
    gpu_power = resolve_launch_gpu_power(args, nodes=nodes_resolved, owner=str(manifest["session_id"]))
    _publish_gpu_power_settings(gpu_power)
    _seed_shared_state(
        session_dir,
        args,
        session_id=manifest["session_id"],
        framework=framework,
        settled=SettledLaunch(compute_partition=compute_partition, gpu_power_settings=gpu_power, base=base),
    )
    return _SessionStart(session_dir=session_dir, session_lock=session_lock, manifest=manifest)


def _restore_platform_state(args: argparse.Namespace, state: SharedState, session_lock: SessionLock) -> None:
    """Re-assert the partition shape, the power settings and the topology this session was measured under.

    Each resumes on the same restore / apply / persist path: an omitted flag
    re-asserts the archived value rather than silently taking a new default.
    """
    # The partition shape is part of the measurement contract, so it resumes on the same restore / apply / persist
    # path as the paths above.
    _restore_partition_shape_from_state(args, state)
    from hyperloom.orchestrator.loop.coordinator import reap_orphaned_servers_best_effort

    reap_orphaned_servers_best_effort(session_lock.session_dir, phase="resume")  # before an apply reads the cards
    # The power settings resume on the same assert / record path: an omitted flag re-asserts the archived value.
    gpu_power, _gpu_power_error = resume_gpu_power_settings(args, state)
    if _gpu_power_error:
        session_lock.release()
        print(f"ERROR: cannot resume this session -- {_gpu_power_error}.", file=sys.stderr)
        sys.exit(2)
    _publish_gpu_power_settings(gpu_power)
    state.gpu_power_settings = gpu_power
    state.compute_partition = _export_partition_shape(
        PartitionRequest(getattr(args, "compute_partition_mode", None), getattr(args, "streams_per_partition", None)),
        framework=state.framework or getattr(args, "framework", None),
        gpu_type=os.environ.get("GPU_TYPE") or state.gpu_type,
        # A resume must re-pass --nodes, so the persisted count is the one that says whether this session was ever
        # multi-node.
        nodes=max(int(getattr(args, "nodes", 1) or 1), int(getattr(state, "nodes", 1) or 1)),
        # Passed for the persisted model identity.
        model=PartitionModel(
            state.model_path or str(getattr(args, "model", "") or ""),
            state.precision or getattr(args, "precision", None),
            shared_state=state,
        ),
    )
    if state.compute_partition.get("mode"):
        print(f"  re-exported partition shape: {state.compute_partition['mode']}")
    if state.framework_repo_path:
        print(f"  re-exported FRAMEWORK_REPO_PATH: {state.framework_repo_path}")
    if state.bypass_scripts_dir:
        print(f"  re-exported HYPERLOOM_BYPASS_SCRIPTS_DIR: {state.bypass_scripts_dir}")
    # Feeds the IR-8 check.
    if state.nodes > 1 and int(getattr(args, "nodes", 1) or 1) <= 1:
        args.nodes = state.nodes
        os.environ["INFERENCE_OPTIMIZER_NODES"] = str(state.nodes)
        print(f"  re-exported nodes         : {state.nodes}")


async def _resume_session(args: argparse.Namespace) -> _SessionStart:
    """Reopen a recorded session, restoring the environment it was measured under."""
    session = _open_resumed_session(args)
    session_dir = session.session_dir
    state = SharedState.load_or_init(session_dir)
    # Restored before the staleness guard below, which resolves the AgentX backend from the environment: a
    # session whose backend came from a pin would otherwise be refused as a workload mismatch. An explicit flag
    # on this resume wins, else the persisted value.
    _resume_server_args, _resume_extra_env = _launch_pins(args, state.operator_server_args, state.operator_extra_env)
    _export_operator_launch_shape(server_args=_resume_server_args, extra_env=_resume_extra_env)
    _stale = measurement_state_is_stale(state)
    if _stale:
        print(
            f"ERROR: cannot resume this session -- {_stale}. "
            "Start a fresh session instead of mixing the two measurement sets.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Checked before any resume step persists state, so a refused resume leaves the session as it was.
    _resume_conflict = (
        latency_budget_scope_error(state.framework, getattr(args, "max_latency_ms", None))
        or constraint_budget_resume_conflict(args, state)
        or _custom_specialist_conflict(args, state)
    )
    if _resume_conflict:
        session.session_lock.release()
        print(f"ERROR: cannot resume this session -- {_resume_conflict}.", file=sys.stderr)
        sys.exit(2)
    print(f"Resuming session: {session_dir}")
    print(f"  manifest.session_id    : {session.manifest.get('session_id')}")
    print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
    print(f"  prior cumul_gain      : {state.cumulative_gain_validated:.2f}%")
    print(f"  prior current_best    : {state.current_best.get('action')}/{state.current_best.get('tput')}")
    print(f"  prior stop_reason     : {state.stop_reason or '(none)'}")

    _reexport_session_environment(
        args,
        state,
        session_dir=session_dir,
        session_id=str(session.manifest.get("session_id") or ""),
        launch_shape=_OperatorLaunchShape(server_args=_resume_server_args, extra_env=_resume_extra_env),
    )
    # After the re-export, not before: both read the GPU_TYPE and the framework it put back.
    _restore_platform_state(args, state, session.session_lock)
    _restore_phase_toggles(args, state, session_dir)
    _open_next_leg(args, state, session)
    return session


async def _seed_fresh_session(args: argparse.Namespace, *, nodes_resolved: int) -> _SessionStart:
    """Open a new session: resolve the target, seed its state, and gate the model before bring-up."""
    # Resolve model path: --model > $MODEL_PATH; fail fast rather than silently use the YAML hardcoded model.
    args.model = args.model or os.environ.get("MODEL_PATH") or ""
    if not args.model:
        print(
            "ERROR: model is required. Pass --model <path> or set MODEL_PATH env (or use --resume-from <session_dir> "
            "to continue an existing session).",
            file=sys.stderr,
        )
        sys.exit(2)
    # Re-export so the rounds _suite.config composes serve the resolved model, not the path the operator typed.
    from hyperloom.common.model_paths import resolve_serving_model_path

    os.environ["MODEL_PATH"] = resolve_serving_model_path(str(args.model)) or str(args.model)

    # Quantization prelude (one-shot, before any session/baseline work): if --quantize was passed, quantize the
    # source model now and rewrite args.model to the exported quantized model.
    await _run_quantization_prelude(args)

    # Resolve framework: --framework > $FRAMEWORK > "sglang" (session-wide; no framework mixing).
    framework = _registered_framework_or_exit(
        args.framework or os.environ.get("FRAMEWORK", ""),
        hint="set $FRAMEWORK accordingly or pass --framework",
    )
    _enforce_expected_framework(framework)
    os.environ["FRAMEWORK"] = framework
    print(f"Framework       : {framework}")
    _apply_operator_supplied_paths(args, framework)
    if framework == "atom":
        _apply_atom_auto_tighten(args)
    _resolve_target_gpu(args, framework)

    # ISL/OSL/CONC/TP/EP were settled and projected before _preflight; precision could not be, because it reads
    # the checkpoint and the model is only resolved above (issue #903).
    _resolve_precision(args)
    # MAX_MODEL_LEN is operator-overridable.
    max_model_len, max_model_len_source = _resolve_run_max_model_len(args)
    args.max_model_len = max_model_len
    os.environ["MAX_MODEL_LEN"] = str(max_model_len)
    os.environ["ISL"] = str(args.isl)
    os.environ["OSL"] = str(args.osl)
    # Profile-scoped OSL (issue #571): exported only when explicitly set, so the profile/roofline materializer can
    # decouple its OSL from the served workload.
    if getattr(args, "profile_osl", None) is not None:
        os.environ["PROFILE_OSL"] = str(args.profile_osl)
    os.environ["PRECISION"] = args.precision
    _fw_version_for_env = resolve_framework_version(args)
    if _fw_version_for_env:
        os.environ["FRAMEWORK_VERSION"] = _fw_version_for_env
    if _agentx_enabled():
        print(
            f"Workload        : AgentX replay, CONC={args.conc}; corpus and model context resolved by inference_suite"
        )
    else:
        print(
            f"Workload        : ISL={args.isl} OSL={args.osl} MAX_MODEL_LEN={max_model_len} ({max_model_len_source}) "
            f"PRECISION={args.precision} FRAMEWORK_VERSION={_fw_version_for_env or '<unset>'}"
        )

    session = _create_fresh_session(args, framework=framework, nodes_resolved=nodes_resolved)
    _start_model_gate(args, session.session_dir)
    # After the seed, before the heavy server bring-up: each check rejects a model that bring-up would fail on.
    if (
        _preflight_unsupported_model_arch(args, session.session_dir)
        or _preflight_model_config_compat(args, session.session_dir)
        or _preflight_context_window(args, session.session_dir)
    ):
        sys.exit(2)
    _finish_model_gate(args, session.session_dir)
    return session


@dataclass(frozen=True)
class _AgentLineup:
    """Which phases and which critic this run brings, once the toggles are read."""

    no_kernel: bool
    no_framework_agent: bool
    critic_choice: str
    critic_agent_root: Path | None


def _resolve_agent_lineup(args: argparse.Namespace) -> _AgentLineup:
    """Read the phase toggles and settle the critic backend before any backend is built.

    An unreachable ``--critic-agent`` runtime aborts here rather than at the
    first critic call, so the run does not pay for bring-up to find out.
    """
    no_kernel = getattr(args, "no_kernel", False)
    no_framework_agent = bool(getattr(args, "no_framework_agent", False))
    print_lineup_banners(args, no_kernel=no_kernel, no_framework_agent=no_framework_agent)
    # Resolve critic backend + runtime root before _build_backends; abort rc=2 if --critic-agent runtime unreachable.
    critic_choice = _resolve_critic_choice(args)
    if critic_choice == "mock" and args.critic_protocol != "auto":
        # The mock critic issues no review inference, so there is no transport for the flag to select.
        print(
            f"WARNING: --critic-protocol={args.critic_protocol} is ignored with the mock critic, "
            "which performs no review inference.",
            file=sys.stderr,
        )
    critic_agent_root: Path | None = None
    if critic_choice == "agent":
        critic_agent_root = _resolve_agent_root()
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
        _validate_agent_runtime(critic_agent_root)
    return _AgentLineup(
        no_kernel=no_kernel,
        no_framework_agent=no_framework_agent,
        critic_choice=critic_choice,
        critic_agent_root=critic_agent_root,
    )


def _install_system_prompts(
    coordinator: Coordinator,
    args: argparse.Namespace,
    objective: Objective,
    lineup: _AgentLineup,
    *,
    macro_cycle: int,
) -> dict[str, str]:
    """Install each role's boot system prompt on ``coordinator``, an operator override winning, and return them.

    The Coordinator has already entered the session's phase; it re-scopes the prompt at every later phase seam.
    """
    state = coordinator.shared_state
    directive = str((dict(getattr(state, "orchestration_memory", {}) or {})).get("next_cycle_directive", "") or "")
    prompt_context = OrchestrationPromptContext(
        no_kernel=lineup.no_kernel,
        no_framework_agent=lineup.no_framework_agent,
        framework=state.framework,
        objective=objective,
        max_minutes=int(round(float(args.max_hours) * 60)),
        # The role record always names Claude; the deployment's provider shape is what actually decides, so read the
        # transport off the built backend.
        transport=getattr(coordinator.backends.get("orchestration"), "transport", TRANSPORT_TOOLS),
        benchmark_mode=str(getattr(state, "benchmark_mode", "") or ""),
        agentx_corpus_shape=state.agentx_corpus_shape,
        agentx_grading=state.grading,
        agentx_backend=state.agentx_backend,
    )
    prompts = {
        "orchestration": args.orch_prompt
        or _build_orchestration_prompt(
            prompt_context, macro_cycle=macro_cycle, cycle_directive=directive, phase=state.phase
        ),
        "critic": args.critic_prompt or _load_critic_prompt(),
    }
    import functools as _functools

    coordinator.orch_prompt = _OrchestrationPrompt(
        overrides=prompts,
        is_user_supplied=bool(args.orch_prompt),
        rebuild=_functools.partial(_build_orchestration_prompt, prompt_context),
    )
    return prompts


def _print_backend_banner(args: argparse.Namespace, backends: dict[str, Any], lineup: _AgentLineup) -> None:
    """Print the backend each role runs on and the run's tick limits."""
    orchestration_str = _orchestration_banner(
        backends.get("orchestration"),
        claude_model=args.claude_model,
        codex_model=args.codex_model,
    )
    kernel_str = "DISABLED" if lineup.no_kernel else "programmatic"
    if lineup.critic_choice == "mock":
        critic_str = "mock"
    else:  # "agent"
        # Read off the backend rather than re-derived from the environment: --critic-protocol can override that
        # derivation, and the banner would then report the model of a transport this run is not using.
        _review_protocol = backends["critic"].protocol
        _review_model = args.claude_model if _review_protocol == "anthropic" else args.codex_model
        critic_str = (
            f"critic-agent(protocol={_review_protocol}, model={_review_model}, root={lineup.critic_agent_root})"
        )
    print(f"Backends        : orchestration={orchestration_str}, kernel={kernel_str}, critic={critic_str}")
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} (budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()


def _settle_custom_specialist_only(args: argparse.Namespace) -> None:
    """Read only mode once the operator's pins are in the environment, and refuse a fresh launch's custom conflicts.

    A resume checks its conflicts against the stored state once that is loaded.
    """
    try:
        args.custom_specialist_only = env_bool(CUSTOM_SPECIALIST_ONLY_ENV)
    except EnvValueError as exc:
        print(f"ERROR: {exc}.", file=sys.stderr)
        raise SystemExit(2) from exc
    conflict = "" if args.resume_from else _custom_specialist_conflict(args, SharedState())
    if conflict:
        print(f"ERROR: {conflict}.", file=sys.stderr)
        raise SystemExit(2)


@dataclass(frozen=True)
class _ProviderIntent:
    """Which provider side follows the other, as configured before preflight fills missing endpoints."""

    claude_follows_codex: bool
    codex_follows_claude: bool


async def _prepare_run(args: argparse.Namespace) -> _ProviderIntent:
    """Clear stale locks, run preflight, resolve the models and settle the pre-session workload checks."""
    # Stale aiter JIT lock sweep: killed runs leave locks that block subsequent starts (locks <5min preserved).
    aiter_sweep = clean_stale_aiter_locks()
    if aiter_sweep["dir"] and aiter_sweep["deleted"]:
        print(
            f"Stale aiter locks cleared: "
            f"dir={aiter_sweep['dir']} "
            f"deleted={aiter_sweep['deleted']} "
            f"skipped_fresh={aiter_sweep['skipped_fresh']} "
            f"errors={aiter_sweep['errors']}"
        )

    # Capture provider intent before _preflight() fills missing endpoints (preflight may populate OPENAI_BASE_URL from
    # ANTHROPIC_BASE_URL).
    claude_follows_codex = llm_config.is_openai_only()
    codex_follows_claude = llm_config.is_anthropic_only()
    try:
        resolved_urls = _preflight(args)
    except Exception as exc:
        try:
            _persist_preflight_failure_artifacts(args, exc)
        except (OSError, ValueError, TypeError):
            log.warning("failed to preserve SBD V6 preflight failure", exc_info=True)
        raise

    # The orchestration role's session refuses without a sandbox; reject before model resolution makes LLM calls.
    _check_codex_sandbox(
        _resolve_orchestration_backend(
            codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
        )
    )
    # Model resolution makes blocking LLM calls; the subscription-token client refuses to run inside this loop.
    await asyncio.to_thread(
        _resolve_models_for_run,
        args,
        resolved_urls,
        claude_follows_codex=claude_follows_codex,
        codex_follows_claude=codex_follows_claude,
    )
    _start_experience_kb()
    # Before either session branch: these are read by the fresh-launch seeding AND by the resume path, so this is the
    # one place that covers both.
    _preflight_agentx_backend(args)
    if not args.resume_from:
        _scope_error = latency_budget_scope_error(
            getattr(args, "framework", None) or os.environ.get("FRAMEWORK", ""), getattr(args, "max_latency_ms", None)
        )
        if _scope_error:
            print(f"ERROR: {_scope_error}.", file=sys.stderr)
            raise SystemExit(2)
    from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts

    resolve_benchmark_timeouts()
    return _ProviderIntent(claude_follows_codex=claude_follows_codex, codex_follows_claude=codex_follows_claude)


async def _open_knowledge_plane(
    args: argparse.Namespace, recipe_kb_client: Any, *, session_dir: Path, nodes_resolved: int
) -> Any:
    """Open the KnowledgePlane over the Recipe KB and bind the multi-node state to the session."""
    # KnowledgePlane owns Recipe KB even when PR Monitor is degraded.
    knowledge_plane = _bootstrap_knowledge_plane(args, recipe_kb_client=recipe_kb_client)

    from ..multi_node.state_paths import bind_state_file_to_session

    bind_state_file_to_session(session_dir)
    if nodes_resolved >= 2:
        await asyncio.to_thread(_prepare_multi_node_state, args, session_dir)
    return knowledge_plane


def _build_run_objective(args: argparse.Namespace) -> Objective:
    """Build the run's objective from the budget and stop-target flags."""
    objective_env = {"MAX_HOURS": str(args.max_hours)}
    for env_name, target in (
        ("TARGET_GAIN_PCT", args.target_gain),
        ("TARGET_TPUT_PER_GPU", args.target_tput),
        ("TARGET_DIR", args.target_baseline_dir),
        ("TARGET_WITHIN_ROOFLINE_PCT", args.target_roofline),
    ):
        objective_env[env_name] = str(target) if target else ""
    objective = build_objective(objective_env)
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
    return objective


def _build_run_backends(
    args: argparse.Namespace, intent: _ProviderIntent, lineup: _AgentLineup, session_dir: Path
) -> dict[str, Any]:
    """Build the LLM backends for the run's model routing and critic."""
    return _build_backends(
        ModelRouting(
            claude_model=args.claude_model,
            codex_model=args.codex_model,
            codex_follows_claude=intent.codex_follows_claude,
            claude_follows_codex=intent.claude_follows_codex,
            deepseek_model=llm_config.resolve_forge_llm_model(llm_config.AGENT_BACKEND_DEEPSEEK),
        ),
        CriticSetup(
            choice=lineup.critic_choice,
            agent_root=lineup.critic_agent_root,
            protocol=args.critic_protocol,
        ),
        session_dir=session_dir,
    )


def _build_coordinator(
    args: argparse.Namespace,
    state: SharedState,
    backends: dict[str, Any],
    *,
    session_dir: Path,
    knowledge_plane: Any,
) -> Coordinator:
    """Pin the session dir, apply ``--reset-state`` and construct the Coordinator."""
    # Expose active session_dir to in-process executors via the canonical pin env var; reinforced here for resume
    # paths.
    os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] = str(session_dir)
    # --reset-state backs up state.json and starts blank, before Coordinator is constructed.
    if getattr(args, "reset_state", False):
        _reset_state_file(session_dir, state)

    return Coordinator(
        session_dir,
        backends=backends,
        settings=CoordinatorSettings(
            model_class=getattr(args, "model_class", None) or os.environ.get("MODEL_CLASS", ""),
            phase_budget_pct=_build_phase_budget_pct(args),
        ),
        advisory=AdvisoryServices(
            # KnowledgePlane facade (None when --degraded-pr).
            knowledge_plane=knowledge_plane,
        ),
    )


def _run_options(args: argparse.Namespace) -> RunOptions:
    """Build the Coordinator run options from the budget and tick flags."""
    return RunOptions(
        max_minutes=args.max_hours * 60.0,
        tick_interval_sec=args.tick_interval_sec,
        max_ticks=args.max_ticks,
        install_signal_handlers=True,
        closing_grace_sec=args.closing_grace_sec,
    )


async def _run_optimize(args: argparse.Namespace) -> int:
    """Run the ``optimize`` subcommand end to end."""
    plug_manifest = resolve_plug_manifest(args)  # first: a refused flag must leave any session as it was
    nodes_resolved = _resolve_run_topology(args)
    _settle_custom_specialist_only(args)
    intent = await _prepare_run(args)
    # A fresh launch has nothing to restore, so settle the budget before the
    # manifest and the workload env read it. A resume keeps ``None`` until
    # ``_restore_budget_and_objective`` has had its chance at the session's own.
    if args.resume_from:
        session = await _resume_session(args)
    else:
        _apply_default_budget(args)
        session = await _seed_fresh_session(args, nodes_resolved=nodes_resolved)
    session_dir = session.session_dir
    session_lock = session.session_lock
    manifest = session.manifest
    plug_host = open_plug_host(plug_manifest, session_dir)

    # Recipe KB T0 anchor (after seed/restore for recipe_canonical_id, before Coordinator); skipped when --degraded-kb.
    recipe_kb_client = _bootstrap_recipe_kb(
        args,
        session_dir=session_dir,
        manifest=manifest,
        resume=bool(args.resume_from),
    )
    knowledge_plane = await _open_knowledge_plane(
        args, recipe_kb_client, session_dir=session_dir, nodes_resolved=nodes_resolved
    )

    objective = _build_run_objective(args)
    lineup = _resolve_agent_lineup(args)

    # T0 may have persisted warm-start state; preserve it before constructing the Coordinator.
    state = SharedState.load_or_init(session_dir)

    backends = _build_run_backends(args, intent, lineup, session_dir)
    coordinator = _build_coordinator(args, state, backends, session_dir=session_dir, knowledge_plane=knowledge_plane)
    _initial_macro_cycle = int(getattr(coordinator.shared_state, "macro_cycle", 0) or 0)
    prompts = _install_system_prompts(coordinator, args, objective, lineup, macro_cycle=_initial_macro_cycle)
    _install_executors(coordinator, args, session_dir=session_dir, knowledge_plane=knowledge_plane)
    # Persist effective system prompts for resume / drift inspection.
    _snapshot_system_prompts(
        session_dir,
        prompts=prompts,
        orchestration_phase=coordinator.shared_state.phase,
        macro_cycle=_initial_macro_cycle,
    )
    _print_backend_banner(args, backends, lineup)

    warn_without_target_gpu(args)

    stop_reason: str | None = None
    try:
        with running_plug_host(plug_host):
            stop_reason = await coordinator.run(
                objective=objective,
                options=_run_options(args),
            )
    finally:
        await _close_optimize_run(coordinator, session_lock, session_dir, stop_reason)

    _reconcile_crash_count(coordinator.shared_state, session_dir)
    # NOTE: conc_sweep is now a SWEEP-phase action auto-enqueued by the Coordinator, not a post-hook here.

    _print_final_summary(coordinator.shared_state, stop_reason, session_dir)
    return _exit_code_for_stop_reason(stop_reason, coordinator.shared_state.baseline_tput)


async def _close_optimize_run(coordinator: Any, session_lock: Any, session_dir: Path, stop_reason: str | None) -> None:
    """Stop the coordinator; the lock release and terminal artifacts still happen when stopping raises."""
    state = coordinator.shared_state
    effective_stop_reason = stop_reason or coordinator.stop_classification
    try:
        await coordinator.stop()
    finally:
        # Drop the single-optimizer session lock once the
        # coordinator has released its leases. The OS would drop it on process
        # exit anyway; this just frees it promptly for an intentional resume.
        session_lock.release()
        _write_cli_terminal_artifacts(session_dir, state, effective_stop_reason)
        try:
            state.save(session_dir)
        except (OSError, ValueError, TypeError):
            log.exception("failed to persist post-stop state (non-fatal)")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments and dispatch the requested subcommand."""
    # Force line-buffering so output piped through a non-TTY sink flushes every line immediately instead of
    # block-buffering, which would otherwise freeze the top-level log for the duration of a blocking benchmark
    # subprocess.
    for stream in (sys.stdout, sys.stderr):
        getattr(stream, "reconfigure", lambda **_: None)(line_buffering=True)

    # Absolutise before the parser defaults or any session path derive from it: subprocesses run with their own cwd,
    # so a relative value would diverge.
    user_data = os.environ.get(ENV_USER_DATA_PATH, "")
    if user_data and not Path(user_data).is_absolute():
        os.environ[ENV_USER_DATA_PATH] = str(Path(user_data).expanduser().absolute())

    parser = _build_parser()
    # Strict on purpose.
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
    if args.command == "optimize":
        dump_thread_stacks_on_sigusr1()
        agent_isolation.announce_agent_isolation()  # before any agent runs: the WARNING and the report's line
        # Resolve any --*-prompt that point at a file.
        for attr in ("orch_prompt", "critic_prompt"):
            v = getattr(args, attr)
            if v and Path(v).exists():
                setattr(args, attr, Path(v).read_text(encoding="utf-8"))
        try:
            configure_benchmark(args)
            _load_custom_specialist_args(args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        with gpu_power_restore_guard():
            return asyncio.run(_run_optimize(args))
    if args.command == "recover":
        return _run_recover_session(args)
    parser.print_help()
    return 2
