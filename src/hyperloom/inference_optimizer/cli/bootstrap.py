# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session bootstrap + summary helpers for the CLI.

Seeds SharedState, snapshots system prompts, prints the session skeleton /
final summary, and resolves reference-recipe / target-summary inputs. Must not
import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.common.env import forge_explicitly_enabled
from hyperloom.common.gpu_partition import published_shape
from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.actions.executors._workload_envs import (
    agentx_enabled as _agentx_enabled,
)
from hyperloom.orchestrator.phases.machine_state import bank_phase_segment
from hyperloom.orchestrator.state.shared_state import SharedState
from .backends import _build_robustness_options
from .parser import (
    DEFAULT_ISL,
    DEFAULT_OSL,
    DEFAULT_CONC,
    DEFAULT_TP,
    DEFAULT_EP,
    DEFAULT_PRECISION,
)
from ..session.paths import _SESSION_SKELETON
from ..session.session_paths import agent_prompt_snapshot
from .model_gate import _load_model_arch, _load_model_config_tags
from ..model_config_utils import summarize_model_config

log = logging.getLogger(__name__)


def parse_operator_extra_env(args: argparse.Namespace) -> dict[str, str]:
    """Parse ``--extra-env NAME=VALUE`` pins into a mapping.

    Args:
        args: Parsed CLI args carrying the repeatable ``extra_env`` list.

    Returns:
        The pins as a mapping; entries without an ``=`` or with a blank name are
        dropped.
    """
    pins: dict[str, str] = {}
    for item in getattr(args, "extra_env", None) or []:
        key, sep, value = str(item).partition("=")
        if sep and key.strip():
            pins[key.strip()] = value
    return pins


#: Default for the KERNEL-entry kernel_opt dispatch when neither spelling of the
#: flag is passed. Both parser dests default to ``None`` so "not passed" stays
#: distinguishable from an explicit value, which is what lets the deprecated
#: alias be honoured without letting it override the current flag.
AUTO_KERNEL_OPT_DEFAULT: bool = True


def resolve_auto_kernel_opt(args: argparse.Namespace) -> bool:
    """Resolve the KERNEL-entry auto-dispatch switch across both spellings.

    ``--auto-kernel-opt`` is the current flag; ``--continue-kernel-after-gemm``
    is its deprecated spelling, kept working because dropping it outright would
    turn a launcher's opt-out into a silent opt-in. The current flag wins when
    both are passed; the alias wins over nothing but the default.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Whether KERNEL entry should dispatch the source-level kernel_opt batch.
    """
    current = getattr(args, "auto_kernel_opt", None)
    legacy = getattr(args, "continue_kernel_after_gemm", None)
    if legacy is not None:
        message = (
            "--continue-kernel-after-gemm is deprecated; use --auto-kernel-opt "
            "(the switch gates the KERNEL-entry kernel_opt dispatch on both "
            "routes, not only after GEMM tuning)"
        )
        # Logged as well as warned. DeprecationWarning is silent under the
        # default filter unless the caller is __main__, and the old flag is
        # hidden from --help by SUPPRESS, so a launcher still passing it had no
        # way to learn the transition had started.
        log.warning("%s", message)
        warnings.warn(message, DeprecationWarning, stacklevel=2)
    if current is not None:
        return bool(current)
    if legacy is not None:
        return bool(legacy)
    return AUTO_KERNEL_OPT_DEFAULT


def resolve_model_display_name(args: argparse.Namespace) -> str:
    """Resolve the canonical model identity used for session naming / display.

    The quantization prelude rewrites ``args.model`` to an export dir whose
    basename is always ``quantized``, so it pins the source identity on
    ``args.model_display_name``; this helper prefers that and otherwise falls
    back to the model-path basename.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The pinned display name when set, else ``Path(args.model).name``.
    """
    override = (getattr(args, "model_display_name", "") or "").strip()
    if override:
        return override
    return Path(str(getattr(args, "model", "") or "")).name


# Bump when a change makes previously recorded AgentX measurements
# incomparable. Epoch 1: aligned to the InferenceX leaderboard invocation
# (upstream scenario + 062126 corpus + native context window + error-rate gate);
# everything measured before it used a context-truncated corpus with no error
# gate, so those numbers describe a different workload.
AGENTX_MEASUREMENT_EPOCH = 1


def agentx_state_is_stale(state: Any) -> str:
    """Return why a resumed session's AgentX state is unusable, or ``""``.

    Two independent reasons, both of which would otherwise corrupt the KEEP
    ledger silently: the session was measured in the other benchmark mode (the
    ledger is keyed on server args alone, so rows collide), or it was measured
    in an older AgentX epoch (same knobs, different workload).

    Args:
        state: The loaded :class:`SharedState`.

    Returns:
        A human-readable reason, or ``""`` when the state may be reused.
    """
    want_mode = "agentx" if _agentx_enabled() else "synthetic"
    had_mode = str(getattr(state, "benchmark_mode", "") or "")
    if had_mode and had_mode != want_mode:
        return (
            f"session was measured in benchmark_mode={had_mode!r} but this run is "
            f"{want_mode!r}; the KEEP ledger keys on server args only, so the two "
            "sets of measurements would overwrite each other"
        )
    if want_mode == "agentx":
        had_epoch = int(getattr(state, "agentx_epoch", 0) or 0)
        if had_epoch != AGENTX_MEASUREMENT_EPOCH:
            return (
                f"session carries AgentX epoch {had_epoch}, this build measures "
                f"epoch {AGENTX_MEASUREMENT_EPOCH}; the recorded results describe "
                "a different workload and cannot anchor or be compared against"
            )
    return ""


def _seed_shared_state(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    session_id: str,
    compute_partition: dict[str, Any] | None = None,
) -> SharedState:
    """Construct and persist the initial :class:`SharedState` for a run.

    Seeds the state from parsed CLI args, clamping the research-lane
    capacity to a safe range to protect quota and the PR-Monitor.

    Args:
        session_dir: Directory for the new session.
        args: Parsed CLI arguments.
        session_id: Identifier assigned to the session.
        compute_partition: The shape the launch validated, passed in rather than
            re-read because the environment carries a lossy subset of it: the
            published variables cannot express where the CU count came from, and
            an absent provenance flag would be reported as a board-table guess
            when the device was in fact probed. Falls back to the published
            variables when a caller has no verdict to hand over.

    Returns:
        The seeded :class:`SharedState` instance.
    """
    # research_lane capacity is locked for the session; clamp to [0, ceiling].
    from hyperloom.orchestrator.policy.gate import (
        detect_gpu_count,
        research_lane_ceiling,
    )

    research_lane_capacity = int(getattr(args, "research_lane_capacity", 1) or 1)
    research_lane_capacity = max(
        0,
        min(research_lane_ceiling(), research_lane_capacity),
    )
    gpu_specialist_capacity_raw = getattr(
        args,
        "gpu_specialist_capacity",
        None,
    )
    try:
        gpu_specialist_capacity = max(
            0,
            int(gpu_specialist_capacity_raw) if gpu_specialist_capacity_raw is not None else detect_gpu_count(),
        )
    except (TypeError, ValueError):
        gpu_specialist_capacity = detect_gpu_count()
    # Collect plateau threshold overrides; absent keys use defaults at compute time.
    plateau_overrides: dict[str, Any] = {}
    if getattr(args, "plateau_explore_keep_gain", None) is not None:
        plateau_overrides["explore_keep_gain_pct"] = float(args.plateau_explore_keep_gain)
    if getattr(args, "plateau_explore_empty_streak", None) is not None:
        plateau_overrides["explore_empty_streak"] = int(args.plateau_explore_empty_streak)
    if getattr(args, "plateau_explore_lookback", None) is not None:
        plateau_overrides["explore_lookback"] = int(args.plateau_explore_lookback)
    if getattr(args, "plateau_kernel_revert_streak", None) is not None:
        plateau_overrides["kernel_revert_streak"] = int(args.plateau_kernel_revert_streak)
    if getattr(args, "plateau_kernel_keep_gain", None) is not None:
        plateau_overrides["kernel_keep_gain_pct"] = float(args.plateau_kernel_keep_gain)
    if getattr(args, "plateau_kernel_lookback", None) is not None:
        plateau_overrides["kernel_lookback"] = int(args.plateau_kernel_lookback)

    # Resolve int workload knobs from the CLI arg, applying the shared fallback
    # default when unset. Inherited env is NOT a config source (issue #903); the
    # CLI resolver (`_resolve_workload_knobs`) has already folded any resume
    # state into ``args`` before this seed runs.
    def _int_arg(arg_name: str, default: int) -> int:
        """Resolve an int workload knob from ``args``, else the fallback default.

        Args:
            arg_name (str): Attribute name to read off ``args``.
            default (int): Fallback applied when the arg is unset/0.

        Returns:
            int: The resolved value, or ``default`` when the arg is unset/invalid.
        """
        val = getattr(args, arg_name, None)
        if val is None:
            return int(default)
        try:
            resolved = int(val)
        except (TypeError, ValueError):
            return int(default)
        return resolved if resolved > 0 else int(default)

    def _resolve_framework_version(args_in: Any) -> str:
        """Resolve ``framework_version`` for the recipe-snapshot canonical id.

        Ladder: explicit CLI/$FRAMEWORK_VERSION -> auto-detect package version
        -> "". Auto-detect runs only when both CLI and env are empty.
        """
        explicit = (getattr(args_in, "framework_version", None) or "").strip() or (
            os.environ.get("FRAMEWORK_VERSION", "") or ""
        ).strip()
        if explicit:
            return explicit
        framework = (getattr(args_in, "framework", None) or "").strip() or (
            os.environ.get("FRAMEWORK", "") or ""
        ).strip()
        if not framework:
            return ""
        from ..recipe_snapshot_constants import (
            DEFAULT_FRAMEWORK_VERSION_SLUG,
            detect_framework_version,
        )

        detected = detect_framework_version(framework)
        # Treat the failure-slug as "no info".
        return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected

    # --explore-overtime-kill-ratio mirror; <=0 disables the gate.
    explore_overtime_kill_ratio_raw = getattr(
        args,
        "explore_overtime_kill_ratio",
        None,
    )
    try:
        explore_overtime_kill_ratio = (
            float(explore_overtime_kill_ratio_raw) if explore_overtime_kill_ratio_raw is not None else 2.0
        )
    except (TypeError, ValueError):
        explore_overtime_kill_ratio = 2.0

    # --explore-variant-timeout-sec mirror; 0 (default) auto-derives the cap, positive pins it.
    explore_variant_timeout_raw = getattr(
        args,
        "explore_variant_timeout_sec",
        None,
    )
    try:
        explore_variant_timeout_sec_override = max(
            0,
            int(explore_variant_timeout_raw) if explore_variant_timeout_raw is not None else 0,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_sec_override = 0

    # --explore-variant-timeout-safety-margin mirror: auto-derive headroom over the soft kill ratio (neg -> 0).
    explore_variant_timeout_safety_margin_raw = getattr(
        args,
        "explore_variant_timeout_safety_margin",
        None,
    )
    try:
        explore_variant_timeout_safety_margin = max(
            0.0,
            float(explore_variant_timeout_safety_margin_raw)
            if explore_variant_timeout_safety_margin_raw is not None
            else 0.5,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_safety_margin = 0.5

    # KB architecture tags from config.json; fresh-launch only.
    _cfg_tags = _load_model_config_tags(str(args.model))

    # Persisted for the session breakdown; the runtime reads the env directly.
    _kernel_optimizer_record = "native" if forge_explicitly_enabled() else "geak"

    # Reference launch recipe (fresh-launch only, fail-soft): lowest-priority
    # base for the baseline server args.
    _ref_args, _ref_envs, _ref_model, _ref_source = _resolve_reference_recipe(args)

    # Canonical model identity (prefers the quantize prelude's pinned source name).
    _model_identity = resolve_model_display_name(args)
    state = SharedState(
        session_id=session_id,
        claw_session_id=(os.environ.get("CLAW_SESSION_ID") or "").strip(),
        sandbox_user_id=(os.environ.get("SANDBOX_USER_ID") or "").strip(),
        model_name=_model_identity,
        model_path=str(args.model),
        model_class=args.model_class or "",
        # Advisory architecture profile; fresh-launch only. Soft-degrade to {}.
        model_arch=_load_model_arch(
            session_dir,
            _model_identity,
            str(args.model),
        ),
        # Architecture-identity tags from config.json.
        model_architectures=_cfg_tags.get("architectures", []),
        model_type=_cfg_tags.get("model_type", ""),
        # config.json structural summary, persisted for downstream collectors.
        model_info=summarize_model_config(str(args.model)),
        framework=os.environ.get("FRAMEWORK", "sglang"),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        # Workload metadata mirrored from CLI/env.
        tp=_int_arg("tp", DEFAULT_TP),
        ep=_int_arg("ep", DEFAULT_EP),
        precision=(str(getattr(args, "precision", None) or DEFAULT_PRECISION).strip()),
        framework_version=_resolve_framework_version(args),
        conc=_int_arg("conc", DEFAULT_CONC),
        isl=_int_arg("isl", DEFAULT_ISL),
        osl=_int_arg("osl", DEFAULT_OSL),
        profile_osl=_int_arg("profile_osl", 0),
        max_model_len=_int_arg("max_model_len", 0),
        kernel_enabled=not getattr(args, "no_kernel", False),
        kernel_optimizer=_kernel_optimizer_record,
        auto_kernel_opt_enabled=resolve_auto_kernel_opt(args),
        target_summary=args.target_summary or _default_target_summary(args),
        baseline_tput=0.0,
        cumulative_gain_validated=0.0,
        reference_server_args=_ref_args,
        reference_envs=_ref_envs,
        reference_model=_ref_model,
        reference_source=_ref_source,
        # Operator launch shape; the process env carries it for one process only,
        # so a resume re-exports it from here rather than from argv.
        operator_server_args=str(getattr(args, "server_args", "") or "").strip(),
        operator_extra_env=parse_operator_extra_env(args),
        bypass_scripts_dir=os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "").strip(),
        framework_repo_path=os.environ.get("FRAMEWORK_REPO_PATH", "").strip(),
        benchmark_backend=os.environ.get("HYPERLOOM_BENCHMARK_BACKEND", "").strip().lower(),
        compute_partition=dict(compute_partition if compute_partition is not None else (published_shape() or {})),
        nodes=max(1, int(getattr(args, "nodes", 1) or 1)),
        robustness_options=_build_robustness_options(args),
        warm_replay_enabled=not bool(getattr(args, "no_warm_replay", False)),
        warm_replay_min_confidence=float(getattr(args, "warm_replay_min_confidence", 0.7)),
        warm_replay_min_reproduce_pct=float(getattr(args, "warm_replay_min_reproduce_pct", 0.8)),
        max_minutes=int((args.max_hours or 0) * 60),
        research_lane_capacity=research_lane_capacity,
        gpu_specialist_capacity=gpu_specialist_capacity,
        plateau_overrides=plateau_overrides,
        explore_overtime_kill_ratio=explore_overtime_kill_ratio,
        enable_roofline=bool(
            getattr(args, "enable_roofline", True),
        ),
        # Standalone FRAMEWORK_AGENT phase; --no-framework-agent skips it.
        framework_agent_phase_enabled=not bool(getattr(args, "no_framework_agent", False)),
        # FRAMEWORK local-exploration arm; --no-framework-local-explore opts out.
        framework_local_explore_enabled=not bool(getattr(args, "no_framework_local_explore", False)),
        # Enablement self-heal lanes; --enablement off opts out.
        enablement_mode=str(getattr(args, "enablement", "all") or "all"),
        # AgentX is a DELIBERATE eval opt-out, not an incidental one. Its client
        # (aiperf_client.sh) never invokes lm-eval, so a genuine AgentX baseline
        # carries no accuracy. ``baseline._maybe_stop_on_missing_baseline_accuracy``
        # explicitly rejects "RUN_EVAL=false in a YAML" as an excuse and would
        # stamp the baseline as an eval failure -- which blocks it from anchoring
        # ``baseline_tput``, leaving every variant's gain None and stalling or
        # stopping the session. Routing AgentX through the same channel as
        # ``--no-eval`` is what makes the opt-out legible to that guard.
        eval_disabled=bool(getattr(args, "no_eval", False)) or _agentx_enabled(),
        explore_variant_timeout_sec_override=explore_variant_timeout_sec_override,
        explore_variant_timeout_safety_margin=explore_variant_timeout_safety_margin,
        research_scout_enabled=bool(getattr(args, "research_scout", True)),
        research_scout_interval=max(1, int(getattr(args, "research_scout_interval", 3) or 3)),
        static_recon_enabled=bool(getattr(args, "static_recon", True)),
        target_advisory_enabled=bool(getattr(args, "target_advisory", True)),
        recipe_sediment_enabled=bool(getattr(args, "recipe_sediment", True)),
        # SWEEP-phase concurrency sweep flags (on by default, both workloads).
        conc_sweep_enabled=bool(getattr(args, "enable_conc_sweep", True)),
        benchmark_mode="agentx" if _agentx_enabled() else "synthetic",
        agentx_epoch=AGENTX_MEASUREMENT_EPOCH if _agentx_enabled() else 0,
        conc_sweep_concs=_parse_conc_sweep_concs(args),
        conc_sweep_total_budget_sec=int(
            getattr(args, "conc_sweep_total_budget_sec", 9000) or 0,
        ),
        conc_sweep_variant_timeout_sec=int(
            getattr(args, "conc_sweep_timeout_sec", 1800) or 1800,
        ),
    )
    state.save(session_dir)
    return state


def _snapshot_system_prompts(
    session_dir: Path,
    *,
    prompts: dict[str, str],
    orchestration_phase: str = "",
) -> None:
    """Persist each agent's effective system prompt to ``agents/<role>/system_prompt.snapshot.md``.

    The boot orchestration prompt is also written under its phase suffix; the
    Coordinator adds one file per later phase it re-scopes into.

    Args:
        session_dir (Path): The session root directory.
        prompts (dict[str, str]): Effective system prompt per agent role.
        orchestration_phase (str): Phase the boot orchestration prompt was
            scoped to; ``""`` writes only the unsuffixed snapshot.
    """
    for role, body in prompts.items():
        target = agent_prompt_snapshot(session_dir, role)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body or "(empty)", encoding="utf-8")
    boot_phase = orchestration_phase.strip()
    if boot_phase and "orchestration" in prompts:
        scoped = agent_prompt_snapshot(session_dir, "orchestration", phase=boot_phase)
        scoped.write_text(prompts["orchestration"] or "(empty)", encoding="utf-8")


def _print_session_skeleton(session_dir: Path) -> None:
    """Echo the freshly-created skeleton so launchers see the exact layout.

    Args:
        session_dir (Path): The session root directory whose skeleton
            subdirectories are listed.
    """
    print(f"Session layout under {session_dir}:")
    for sub in _SESSION_SKELETON:
        marker = "ok" if (session_dir / sub).is_dir() else "MISSING"
        print(f"  [{marker}] {sub}/")
    print("  [ok] manifest.json (written first)")


def _print_final_summary(
    state: SharedState,
    stop_reason: str,
    session_dir: Path | None = None,
) -> None:
    """Print the end-of-run summary block to stdout.

    Reports the stop reason, session id, model, baseline throughput, the
    validated cumulative gain (with a staleness warning when the optimization
    stack grew after the last validation), the current best config, pruned
    families, and crash count.
    On ``baseline_failed`` it also surfaces the real terminal root cause from
    ``reports/final.json``.

    Args:
        state (SharedState): The final shared state after the run completes.
        stop_reason (str): Why the run stopped (e.g. ``"target_reached"``).
        session_dir (Path | None): Session root, used to read the
            ``failure_summary`` block on failure runs.

    Returns:
        None
    """
    print()
    print("================ Final summary ================")
    print(f"  stop_reason          : {stop_reason}")
    print(f"  session_id           : {state.session_id}")
    print(f"  model                : {state.model_name}")
    from .. import framework_registry

    print(
        f"  baseline             : {framework_registry.format_primary_metric(getattr(state, 'framework', ''), state.baseline_tput)}"
    )
    if session_dir is not None and stop_reason == "baseline_failed":
        failure_summary = _read_failure_summary(session_dir)
        if failure_summary and failure_summary.get("root_cause"):
            print(
                f"  root_cause           : "
                f"[{failure_summary.get('root_cause_type', 'unknown')}] "
                f"{failure_summary.get('root_cause')}"
            )
            if failure_summary.get("server_log"):
                print(f"  server_log           : {failure_summary.get('server_log')}")
    if state.cumulative_gain_validated_ts:
        stale = (
            " ⚠ stack changed since validation"
            if len(state.optimization_stack) > state.cumulative_gain_validated_stack_len
            else ""
        )
        print(
            f"  cumulative_gain_val  : {state.cumulative_gain_validated:.2f}% "
            f"(validated_at_stack_len={state.cumulative_gain_validated_stack_len}, "
            f"ts={state.cumulative_gain_validated_ts}){stale}"
        )
    else:
        print("  cumulative_gain_val  : 0.00% ⚠ never validated — no `explore` KEEP has landed yet")
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    _print_kernel_opt_summary_line(state)
    print("===============================================")


def _bank_previous_leg_phase_segment(state: SharedState) -> None:
    """Bank the phase time the stopped leg spent but never recorded.

    Per-phase totals are banked at each transition out of a phase, so a leg that
    stopped mid-phase left its last segment live — and the resume boundary is
    about to floor that segment away as the idle gap it mostly is.
    :attr:`SharedState.stop_ts` is the only recorded evidence of when the leg
    ended; a clean stop or a crash leaves none, and then the segment stays
    unbanked. That under-charges the phase, which is the direction the phase
    clock tolerates: over-charging ends a phase early.

    The end is clamped to the present for the same reason: no leg can have run
    past the moment it is being resumed, so a ``stop_ts`` stamped ahead of now
    would bank the difference as spend the phase never had.

    Must run before ``resumed_ts`` is restamped, which would floor the segment
    to nothing.

    Args:
        state (SharedState): The loaded session state, mutated in place.
    """
    stop_unix = min(to_unix(state.stop_ts, 0.0) or 0.0, time.time())
    if stop_unix <= 0.0:
        return
    bank_phase_segment(state, until_unix=stop_unix)


def _begin_resume_leg(state: SharedState, *, reanchor_budget: bool) -> str:
    """Mark the start of a resumed run leg on ``state`` (caller persists).

    Every resume stamps :attr:`SharedState.resumed_ts`. The previous leg's
    CLOSE transition stays in ``phase_history`` and would otherwise keep
    speaking for the resumed run — a report reads it as the session's stop
    reason and end time — and this boundary is what dates it as a previous
    leg's. It is also what stops the phase clock charging the gap between the
    two legs to whichever phase the session stopped in.

    Only a previous leg that stopped for a recorded reason, or crashed
    repeatedly, re-anchors the wall-clock budget. That also clears
    ``deadline_unix`` so ``Coordinator.run`` can stamp a new one from the
    reset ``start_ts``; keeping the spent stamp would make ``--force-resume``
    after ``time_exhausted`` stop immediately. After a clean stop ``start_ts``
    and the stamp are deliberately kept, so remaining wall-clock is the
    persisted deadline, not this invocation's ``--max-hours``. Raising that
    flag on this path does not extend the stamp. The phase clock moves on
    either branch: the two answer different questions, and neither answer
    includes time nothing was running.

    Args:
        state (SharedState): The loaded session state, mutated in place.
        reanchor_budget (bool): Whether the budget restarts from this leg.

    Returns:
        str: The timestamp stamped as this leg's boundary.
    """
    _bank_previous_leg_phase_segment(state)
    state.resumed_ts = now_iso()
    if reanchor_budget:
        # CRITICAL: clear the leftover stop_reason or Orchestration heartbeats
        # forever think the work is done.
        state.stop_reason = ""
        state.stop_ts = ""
        state.closing_phase = False
        state.closing_started_unix = 0.0
        state.closing_report_task_id = ""
        # Reset persisted crash_count so a fresh resume isn't immediately tripped into "emergency".
        state.crash_count = 0
        # Reset start_ts to now so resume budget isn't seen as already-over-budget by the LLM.
        state.start_ts = state.resumed_ts
        # The stamp is the loop's budget. Leaving a spent one in place after
        # resetting start_ts would make this leg look already exhausted.
        state.deadline_unix = 0.0
        state.teardown_timings_sec = {}
    return state.resumed_ts


def _clean_stop_resume_budget_lines(state: SharedState, *, max_hours: float) -> list[str]:
    """Operator-facing resume notes when the wall-clock stamp is kept.

    Remaining time is :meth:`SharedState.remaining_minutes` (the stamp), not
    this invocation's ``--max-hours``. Raising that flag here does not extend
    the deadline.

    Args:
        state: Loaded session state after :func:`_begin_resume_leg`.
        max_hours: This invocation's ``--max-hours``.

    Returns:
        Lines to print, each already prefixed with ``  → ``.
    """
    elapsed_h = state.elapsed_minutes() / 60.0
    remaining_min = state.remaining_minutes()
    lines = [
        f"  → start_ts kept at {state.start_ts} (clean stop, no stop_reason): the persisted deadline is kept",
    ]
    if remaining_min is None:
        lines.append(f"  → {elapsed_h:.2f}h elapsed; no persisted deadline")
        return lines
    remaining_h = remaining_min / 60.0
    lines.append(f"  → budget: {elapsed_h:.2f}h elapsed, {remaining_h:.2f}h left on the persisted stamp")
    cli_hours = float(max_hours or 0.0)
    if remaining_min <= 0.0:
        lines.append(
            "  → WARNING: the stamped deadline is already spent; start a fresh "
            "session, or the run stops almost immediately"
        )
        lines.append(
            "  → raising --max-hours on a clean-stop resume does not extend the "
            "stamp; a recorded stop_reason re-anchors the budget"
        )
    elif cli_hours > 0.0:
        cli_left_min = cli_hours * 60.0 - elapsed_h * 60.0
        if abs(cli_left_min - remaining_min) > 1.0:
            lines.append(f"  → this invocation's --max-hours {cli_hours:.2f} does not extend or shrink that stamp")
    return lines


def _reconcile_crash_count(state: SharedState, session_dir: Path) -> None:
    """Reconcile persisted ``crash_count`` (state.json + final.json) up to the live in-memory value.

    Only ever raises the persisted value (max), never lowers it; best-effort, never fatal.
    """
    live = int(getattr(state, "crash_count", 0) or 0)

    # state.json: reload, bump if stale, atomic re-save.
    try:
        disk_state = SharedState.load_or_init(session_dir)
        if int(disk_state.crash_count or 0) < live:
            disk_state.crash_count = live
            disk_state.save(session_dir)
    except Exception:  # noqa: BLE001
        log.exception("crash_count reconcile (state.json) failed (non-fatal)")

    # reports/final.json: patch the single field in place if present.
    try:
        from ..session.session_paths import reports_dir

        final_json = reports_dir(session_dir) / "final.json"
        if final_json.exists():
            data = json.loads(final_json.read_text(encoding="utf-8"))
            if int(data.get("crash_count") or 0) < live:
                data["crash_count"] = live
                final_json.write_text(
                    json.dumps(data, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
    except Exception:  # noqa: BLE001
        log.exception("crash_count reconcile (final.json) failed (non-fatal)")


def _print_kernel_opt_summary_line(state: SharedState) -> None:
    """One-line forensic readout of kernel_opt attempts at session end (matches the on-disk report; best-effort)."""
    try:
        from hyperloom.orchestrator.kernel.attempt_summary import (
            build_kernel_optimization_summary,
        )

        session_dir = _resolve_session_dir_for_summary(state)
        if session_dir is None:
            return
        summary = build_kernel_optimization_summary(state, session_dir)
        totals = summary.get("totals") or {}
        attempted = int(totals.get("attempted") or 0)
        if attempted == 0 and int(totals.get("unattempted") or 0) == 0:
            return
        integrated = int(totals.get("integrated") or 0)
        rejected = int(totals.get("rejected") or 0)
        unattempted = int(totals.get("unattempted") or 0)
        print(
            f"  kernel_opt           : {attempted} attempted "
            f"({integrated} integrated, {rejected} rejected), "
            f"{unattempted} unattempted in top candidates"
        )
        takeaways = summary.get("top_takeaways") or []
        if len(takeaways) >= 2:
            print(f"  kernel_opt_top_cause : {takeaways[1]}")
        report_path = Path(session_dir) / "reports" / "kernel_optimization_summary.json"
        if report_path.is_file():
            print(f"  kernel_opt_report    : {report_path}")
    except Exception:  # noqa: BLE001 — stdout print must never fail the run
        pass


def _default_target_summary(args: argparse.Namespace) -> str:
    """Compose a human-readable objective summary from the CLI target flags.

    Used as the fallback ``target_summary`` when the operator did not pass an
    explicit ``--target-summary``. The phrasing depends on which target flag is
    set: ``--target-gain`` (percentage), ``--target-tput`` (tok/s/GPU for
    serving; for scriptable xDiT the target throughput is img/s and is shown as
    the equivalent per-image latency e2el_mean_ms), or neither (open-ended
    optimization within the time budget).

    Args:
        args (argparse.Namespace): Parsed ``optimize`` arguments (reads ``model``,
            ``target_gain``, ``target_tput``, ``max_hours``, ``framework``).

    Returns:
        str: A one-sentence description of the run's objective.
    """
    if args.target_gain:
        return (
            f"Establish baseline on {Path(args.model).name} then drive "
            f"cumulative_gain_validated to >= {args.target_gain}% within "
            f"{args.max_hours}h."
        )
    if args.target_tput:
        from .. import framework_registry

        target = framework_registry.format_primary_metric(getattr(args, "framework", None), args.target_tput)
        return f"Establish baseline on {Path(args.model).name} then reach {target} within {args.max_hours}h."
    return f"Optimize {Path(args.model).name} for up to {args.max_hours}h (no target)."


def _parse_conc_sweep_concs(args: argparse.Namespace) -> list[int]:
    """Parse ``--conc-sweep-concs`` into a list[int]; non-integers warned+dropped.

    An unset flag falls back to the ladder for the workload this session runs,
    which is why the flag defaults to ``None`` rather than to a ladder string:
    a typed value must be distinguishable from an omitted one.
    """
    from hyperloom.orchestrator.kernel.conc_sweep import default_concs_for_mode

    fallback = default_concs_for_mode("agentx" if _agentx_enabled() else "")
    raw = str(getattr(args, "conc_sweep_concs", "") or "").strip()
    if not raw:
        return fallback
    out: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            log.warning("conc_sweep: ignoring non-integer CONC token %r", t)
    return out or fallback


def _read_failure_summary(session_dir: Path) -> dict | None:
    """Read ``reports/final.json``'s ``failure_summary`` block, if present.

    Best-effort: returns ``None`` when the file is missing/unreadable or the
    block is absent (e.g. non-failure runs). Used to surface the real terminal
    root cause in the end-of-run summary on ``baseline_failed``.
    """
    try:
        from ..session.session_paths import reports_dir

        final_json = reports_dir(session_dir) / "final.json"
        data = json.loads(final_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    fs = data.get("failure_summary") if isinstance(data, dict) else None
    return fs if isinstance(fs, dict) else None


def _resolve_reference_recipe(
    args: argparse.Namespace,
) -> tuple[str, dict[str, str], str, str]:
    """Resolve the reference launch recipe for a fresh launch.

    Returns ``(server_args, envs, model, source)``, empty when
    ``--reference-script`` was not given. A flag pointing at a recipe that cannot
    be read or yields nothing usable exits with ``SystemExit(2)``.
    """
    source = (getattr(args, "reference_script", None) or "").strip()
    if not source:
        return ("", {}, "", "")

    framework = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    from ..reference_script import parse_reference_script

    try:
        recipe = parse_reference_script(source, framework=framework)
    except Exception as exc:
        print(f"ERROR: --reference-script {source!r} could not be parsed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if not recipe.server_args and not recipe.envs:
        print(
            f"ERROR: --reference-script {source!r} lifted no server flags and no env exports",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"Reference script: {source} ({len(recipe.server_args.split())} arg tokens, {len(recipe.envs)} env(s))")
    return (recipe.server_args, dict(recipe.envs), recipe.model or "", source)


def _resolve_session_dir_for_summary(state: SharedState) -> Path | None:
    """Best-effort session_dir lookup ($HYPERLOOM_SESSION_DIR) for the stdout kernel_opt line; ``None`` if unresolved."""
    env_sd = os.environ.get("HYPERLOOM_SESSION_DIR", "").strip()
    if env_sd:
        p = Path(env_sd).expanduser()
        if p.is_dir():
            return p
    return None
