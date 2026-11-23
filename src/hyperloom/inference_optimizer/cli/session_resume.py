# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reopen a recorded session and restore the environment, platform state and budget it was measured under."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom_common import gpu_card

from ..model_config_utils import summarize_model_config
from ..session.manifest import load_manifest
from .bootstrap import (
    LaunchInfo,
    _begin_resume_leg,
    _emit_launch_info,
    _resume_budget_lines,
)
from .custom_specialist import _apply_custom_specialist_resume
from .launch_shape import (
    _positive_env_int,
    _resolve_precision,
    _resolve_workload_knobs,
    _seed_env_from_resumed_state,
)
from .install_event import _persist_install_event
from .model_gate import _record_resumed_model_gate
from .session_setup import (
    _SessionStart,
    _acquire_session_lock_or_exit,
    _apply_atom_auto_tighten,
    _apply_default_budget,
    _apply_operator_supplied_paths,
    _enforce_expected_framework,
    _persist_operator_supplied_paths,
    _read_session_card,
    _registered_framework_or_exit,
    _require_custom_entrypoint,
    _restore_operator_supplied_paths_from_state,
)

if TYPE_CHECKING:
    from hyperloom.orchestrator.state.shared_state import SharedState


#: Manifest objective kinds and the flag each one was parsed from.
_OBJECTIVE_KIND_TO_FLAG: Mapping[str, str] = {
    "gain_pct": "target_gain",
    "tput": "target_tput",
    "baseline": "target_baseline_dir",
    "roofline_pct": "target_roofline",
}


def _restore_budget_and_objective(args: Any, state: SharedState, manifest: Mapping[str, Any]) -> list[str]:
    """Fill the budget and the stop target from the archive when this resume omitted them.

    A bare resume would otherwise rebuild both from the flags: the budget would
    shorten a longer session and close the leg as ``time_exhausted``, and the
    objective would be dropped. An operator may pass only ``--resume-from``.
    An omitted ``--max-hours`` arrives as ``None``, so a smaller
    explicit budget still tightens the leg, which is what ``_start_run`` reads.

    Args:
        args: Parsed arguments for this resume; an explicit flag always wins.
        state: Loaded session state, whose ``max_minutes`` already carries every
            ``--extend-hours`` grant from earlier legs.
        manifest: The session manifest, which is where the objective is recorded.

    Returns:
        Lines to print, each already prefixed with ``  → ``.
    """
    lines: list[str] = []
    persisted_minutes = float(getattr(state, "max_minutes", 0) or 0)
    if getattr(args, "max_hours", None) is None and persisted_minutes > 0:
        args.max_hours = persisted_minutes / 60.0
        lines.append(f"  → restored budget: --max-hours {args.max_hours:.2f} (persisted)")

    # Restored as a set: a target named on this resume replaces the persisted
    # objective outright rather than joining it, which build_objective refuses.
    if any(getattr(args, flag, None) is not None for flag in _OBJECTIVE_KIND_TO_FLAG.values()):
        return lines
    recorded = manifest.get("objective") or {}
    # ``_objective_summary`` writes one entry, and ``objectives`` too when a
    # roofline target joins a throughput one.
    for entry in recorded.get("objectives") or [recorded]:
        flag = _OBJECTIVE_KIND_TO_FLAG.get(str(entry.get("kind") or ""))
        value = entry.get("value")
        if flag is None or value is None:
            continue
        setattr(args, flag, str(value) if flag == "target_baseline_dir" else float(value))
        lines.append(f"  → restored objective: --{flag.replace('_', '-')} {value}")
    return lines


def _restore_phase_toggles(args: argparse.Namespace, state, session_dir: Path) -> None:
    """Honour the phase and gate decisions the original run persisted.

    A CLI flag can still turn a phase off on this resume; it cannot turn one
    back on that the session recorded as disabled.
    """
    # Warm-replay gates: a non-default value is the only "operator set this" signal, since the parser gives these
    # flags real defaults rather than None.
    if args.no_warm_replay:
        state.warm_replay_enabled = False
    _wr_value = getattr(args, "warm_replay_min_confidence", None)
    if _wr_value is not None:
        state.warm_replay_min_confidence = _wr_value
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


@dataclass(frozen=True)
class _OperatorLaunchShape:
    """The operator's server args and extra env a resume re-applies: this resume's flags, else the session's record."""

    server_args: str
    extra_env: dict


def _reexport_session_identity(args: argparse.Namespace, state, session_dir: Path) -> None:
    """Re-export the resumed session's model path, framework and GPU type, and publish the card it runs on."""
    if state.model_path:
        resume_model_path = str(state.model_path)
        os.environ["MODEL_PATH"] = resume_model_path
        print(f"  re-exported MODEL_PATH: {resume_model_path}")
        # Backfill model_info for sessions created before the field existed (or whose config was unreadable at
        # launch); fail-soft to {}.
        if not state.model_info:
            state.model_info = summarize_model_config(resume_model_path)
            if state.model_info:
                state.save(session_dir)
                print("  backfilled model_info (from config.json)")
    state.framework = _registered_framework_or_exit(
        state.framework or os.environ.get("FRAMEWORK", ""),
        hint="it comes from the resumed session, or from $FRAMEWORK when the session recorded none",
    )
    _enforce_expected_framework(state.framework)
    os.environ["FRAMEWORK"] = state.framework
    print(f"  re-exported FRAMEWORK : {state.framework}")
    if state.framework == "atom":
        _apply_atom_auto_tighten(args)
    if state.gpu_type:
        os.environ["TARGET_GPU_TYPE"] = os.environ["GPU_TYPE"] = state.gpu_type
        print(f"  re-exported GPU_TYPE  : {state.gpu_type}")
    # Read again rather than restored: the facts are the machine's this leg runs on.
    nodes = max(int(getattr(args, "nodes", 1) or 1), int(getattr(state, "nodes", 1) or 1))
    gpu_card.publish_session_card(_read_session_card(nodes))


def _reexport_session_environment(
    args: argparse.Namespace,
    state,
    *,
    session_dir: Path,
    session_id: str,
    launch_shape: _OperatorLaunchShape,
) -> None:
    """Put the session's recorded environment back, so a fresh shell resumes what was measured.

    Without this a resume falls back to the YAML defaults for anything the
    original run pinned. Every knob resumes on the same ladder: an explicit
    flag on this resume wins, otherwise the archived value is re-exported, so a
    re-passed pin can still change it.
    """
    # Re-export session-level env from persisted state so a fresh-shell resume doesn't fall back to YAML defaults.
    _reexport_session_identity(args, state, session_dir)
    info = LaunchInfo(
        pid=os.getpid(),
        session_dir=session_dir,
        session_id=session_id,
        run_log=os.environ.get("INFERENCE_OPTIMIZER_RUN_LOG", ""),
        gpu_type=state.gpu_type or "",
        framework=state.framework,
        model=str(state.model_path or ""),
    )
    _emit_launch_info(info, launch_info_file=getattr(args, "launch_info_file", None))
    _seed_env_from_resumed_state(state, launch_shape.extra_env)
    # The seeding above puts the recorded workload on the environment rung, so it outranks a shell export while an
    # explicit --isl/--conc/... on this resume still wins. The resolved values are then projected back into env so
    # resume sees the same workload contract (not YAML defaults).
    _resolve_workload_knobs(args, state)
    _resolve_precision(args, state)
    # Same ladder as the knobs above; the environment rung is what lets a re-passed pin change it.
    _resume_max_model_len = (
        getattr(args, "max_model_len", None) or _positive_env_int("MAX_MODEL_LEN") or getattr(state, "max_model_len", 0)
    )
    for env_name, val in (
        ("TP", args.tp),
        ("EP", args.ep),
        ("CONC", args.conc),
        ("ISL", args.isl),
        ("OSL", args.osl),
        ("MAX_MODEL_LEN", _resume_max_model_len),
    ):
        if val:
            os.environ[env_name] = str(int(val))
            print(f"  re-exported {env_name:<14s}: {int(val)}")
    # Profile-scoped OSL: an explicit --profile-osl on this resume wins; otherwise re-export the value persisted
    # from the original run.
    _resume_profile_osl = getattr(args, "profile_osl", None) or getattr(state, "profile_osl", 0)
    if _resume_profile_osl:
        os.environ["PROFILE_OSL"] = str(int(_resume_profile_osl))
        state.profile_osl = int(_resume_profile_osl)
        print(f"  re-exported PROFILE_OSL   : {int(_resume_profile_osl)}")
    os.environ["PRECISION"] = args.precision
    print(f"  re-exported PRECISION     : {args.precision}")
    if getattr(state, "framework_version", ""):
        os.environ["FRAMEWORK_VERSION"] = state.framework_version
        print(f"  re-exported FRAMEWORK_VERSION: {state.framework_version}")
    # Operator launch shape: already exported above, ahead of the staleness guard that reads the pins.
    state.operator_server_args = launch_shape.server_args
    state.operator_extra_env = launch_shape.extra_env
    if launch_shape.server_args:
        print(f"  re-exported server_args   : {launch_shape.server_args}")
    if launch_shape.extra_env:
        print(f"  re-exported extra_env     : {','.join(sorted(launch_shape.extra_env))}")
    _apply_custom_specialist_resume(args, state)
    # Custom-workload paths: an explicit --framework-path / --benchmark-scripts-dir on this resume wins, else the
    # persisted value.
    _restore_operator_supplied_paths_from_state(args, state)
    _apply_operator_supplied_paths(args, state.framework)
    _require_custom_entrypoint(
        state.framework,
        gpu_type=os.environ.get("GPU_TYPE") or state.gpu_type,
    )
    _persist_operator_supplied_paths(state)


def _open_resumed_session(args: argparse.Namespace) -> _SessionStart:
    """Lock the session ``--resume-from`` names, refusing a directory that is not a resumable session.

    ``--resume-from`` names a subdirectory under the workspace root; USER_DATA_PATH itself stays where it is.
    """
    from ..session.paths import ENV_CURRENT_SESSION_DIR, ensure_session_skeleton, is_path_within, workspace_root

    ws = workspace_root()
    session_dir = Path(args.resume_from).expanduser().resolve()
    if getattr(args, "session_dir", None):
        print("ERROR: --session-dir names a new session; it cannot be combined with --resume-from.", file=sys.stderr)
        sys.exit(2)
    if not is_path_within(session_dir, ws):
        print(
            f"ERROR: --resume-from {session_dir!r} is not under $USER_DATA_PATH={ws}. Move USER_DATA_PATH to the "
            "workspace root (the parent of the per-session subdirs) and pass the per-session subdir via --resume-from.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not session_dir.is_dir():
        print(f"ERROR: --resume-from {session_dir!r} does not exist.", file=sys.stderr)
        sys.exit(2)
    # Pin before Coordinator/SharedState load so paths/subprocesses inherit the resolved location.
    os.environ[ENV_CURRENT_SESSION_DIR] = str(session_dir)
    ensure_session_skeleton(session_dir)

    # Single-optimizer guard: take the session lock before any state.json / lease access.
    session_lock = _acquire_session_lock_or_exit(session_dir)
    from ..session.resume_guard import ResumeBlocked, ensure_resume_safe
    from hyperloom.orchestrator.bus.resource_lock import local_owner_scope

    try:
        ensure_resume_safe(session_dir, owner_scope=local_owner_scope())
    except ResumeBlocked as exc:
        session_lock.release()
        print(f"ERROR: --resume-from {exc}", file=sys.stderr)
        sys.exit(2)
    _persist_install_event(args, session_dir)

    try:
        manifest = load_manifest(session_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: --resume-from failed: {exc}", file=sys.stderr)
        sys.exit(2)
    if not (session_dir / "state.json").exists():
        print(
            f"ERROR: --resume-from failed: {session_dir}/state.json missing "
            "(manifest exists but Coordinator never wrote SharedState)",
            file=sys.stderr,
        )
        sys.exit(2)
    return _SessionStart(session_dir=session_dir, session_lock=session_lock, manifest=manifest)


def _open_next_leg(args: argparse.Namespace, state: SharedState, session: _SessionStart) -> None:
    """Begin the resumed session's next leg: clear the last leg's stop, grant ``--extend-hours``, report the budget.

    ``target_reached`` is a terminal state requiring ``--force-resume`` to push past it; other reasons auto-clear.
    """
    prior_stop = state.stop_reason
    prior_crash = state.crash_count
    if prior_stop == "target_reached" and not getattr(args, "force_resume", False):
        print(
            f"\nERROR: --resume-from blocked by terminal stop_reason={prior_stop!r}.\n\n"
            f"  SKILL.md (Run-time signals): {prior_stop!r} is a deliberate terminal state.\n"
            "  The optimizer will not auto-resume past it because the prior run\n"
            "  declared exhaustion — picking up where it left off only repeats\n"
            "  the same exhaustion verdict.\n\n"
            "  Override paths:\n"
            "  1. Pass ``--force-resume`` if you have changed the workload /\n"
            "     search space / model / strategy and want to continue regardless.\n"
            "  2. Start a fresh session (different $USER_DATA_PATH) for a clean run.\n\n"
            f"  Reports for the prior run live under {session.session_dir}/reports/.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    _begin_resume_leg(state)
    extend_hours = float(args.extend_hours)
    if extend_hours > 0.0:
        state.extend_budget_minutes(extend_hours * 60.0)
    state.save(session.session_dir)
    _record_resumed_model_gate(
        args,
        session.session_dir,
        workload_overrides={
            key: str(getattr(state, key) or session.manifest.get(key) or "")
            for key in ("model_path", "model_name", "framework", "gpu_type")
        },
    )
    override_note = " (--force-resume override)" if prior_stop == "target_reached" else ""
    print(f"  → cleared stop_reason and crash_count (was {prior_crash}) for this leg{override_note}")
    for line in _restore_budget_and_objective(args, state, session.manifest):
        print(line)
    _apply_default_budget(args)
    for line in _resume_budget_lines(state, extend_hours=extend_hours):
        print(line)
