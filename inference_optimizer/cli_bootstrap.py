# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session bootstrap + summary helpers for the CLI.

Extracted from ``cli.py`` (phase 4). Seeds SharedState, snapshots system
prompts, prints the session skeleton / final summary, and resolves
reference-recipe / target-summary inputs. Imports stdlib + orchestrator +
cli_model_gate only; must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .orchestrator.shared_state import SharedState
from .paths import _SESSION_SKELETON, workspace_root as _workspace_root_resolve
from .session_paths import agent_prompt_snapshot
from .cli_model_gate import _load_model_arch, _load_model_config_tags

log = logging.getLogger(__name__)

def _seed_shared_state(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    session_id: str,
) -> SharedState:
    """Construct and persist the initial :class:`SharedState` for a run.

    Seeds the state from parsed CLI args, clamping the research-lane
    capacity to a safe range to protect quota and the PR-Monitor.

    Args:
        session_dir: Directory for the new session.
        args: Parsed CLI arguments.
        session_id: Identifier assigned to the session.

    Returns:
        The seeded :class:`SharedState` instance.
    """
    # research_lane capacity is locked for the session; clamp to [0, ceiling] (2×GPU) to protect quota/PR-Monitor.
    from inference_optimizer.orchestrator.policy import (
        research_lane_ceiling,
    )
    research_lane_capacity = int(
        getattr(args, "research_lane_capacity", 1) or 1
    )
    research_lane_capacity = max(
        0, min(research_lane_ceiling(), research_lane_capacity),
    )
    gpu_specialist_capacity_raw = getattr(
        args, "gpu_specialist_capacity", None,
    )
    try:
        gpu_specialist_capacity = max(
            0,
            int(gpu_specialist_capacity_raw)
            if gpu_specialist_capacity_raw is not None else 0,
        )
    except (TypeError, ValueError):
        gpu_specialist_capacity = 0
    # Collect plateau threshold overrides; absent keys fall through to DEFAULT_PLATEAU_* at compute time.
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
    # EXPLORE HARD force-exit thresholds; either fires an explore_force_exit_low_budget exit (overrides all).
    if getattr(args, "explore_force_exit_hours_remaining", None) is not None:
        plateau_overrides["force_exit_hours_remaining"] = float(
            args.explore_force_exit_hours_remaining
        )
    if getattr(args, "explore_force_exit_budget_pct", None) is not None:
        plateau_overrides["force_exit_budget_pct"] = float(
            args.explore_force_exit_budget_pct
        )
    # Resolve workload metadata from CLI flags then env; parse duplicated here to avoid re-reading manifest.json.
    def _int_env_or_arg(arg_name: str, env_name: str) -> int:
        """Resolve an int workload knob from a CLI arg, falling back to env.

        Args:
            arg_name (str): Attribute name to read off ``args``.
            env_name (str): Environment variable consulted when the arg is unset/0.

        Returns:
            int: The resolved value, or 0 when neither source yields a valid int.
        """
        val = getattr(args, arg_name, None)
        if val is None or val == 0:
            raw = (os.environ.get(env_name, "") or "").strip()
            return int(raw) if raw.isdigit() else 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def _resolve_framework_version(args_in: Any) -> str:
        """Resolve ``framework_version`` for the recipe-snapshot canonical id.

        Ladder: explicit CLI/$FRAMEWORK_VERSION → auto-detect package __version__ → "" (canonical_id
        substitutes unknown_version). Auto-detect runs only when both CLI and env are empty.
        """
        explicit = (
            (getattr(args_in, "framework_version", None) or "").strip()
            or (os.environ.get("FRAMEWORK_VERSION", "") or "").strip()
        )
        if explicit:
            return explicit
        framework = (
            (getattr(args_in, "framework", None) or "").strip()
            or (os.environ.get("FRAMEWORK", "") or "").strip()
        )
        if not framework:
            return ""
        from .recipe_snapshot_constants import (
            DEFAULT_FRAMEWORK_VERSION_SLUG,
            detect_framework_version,
        )

        detected = detect_framework_version(framework)
        # Treat the failure-slug as "no info"; canonical_id redoes the fallback at use time.
        return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected

    # --explore-overtime-kill-ratio: mirror into fresh SharedState for ExploreExecutor; <=0 disables the gate.
    explore_overtime_kill_ratio_raw = getattr(
        args, "explore_overtime_kill_ratio", None,
    )
    try:
        explore_overtime_kill_ratio = (
            float(explore_overtime_kill_ratio_raw)
            if explore_overtime_kill_ratio_raw is not None else 1.10
        )
    except (TypeError, ValueError):
        explore_overtime_kill_ratio = 1.10

    # --explore-variant-timeout-sec mirror; 0 (default) auto-derives the cap, positive pins it.
    explore_variant_timeout_raw = getattr(
        args, "explore_variant_timeout_sec", None,
    )
    try:
        explore_variant_timeout_sec_override = max(
            0,
            int(explore_variant_timeout_raw)
            if explore_variant_timeout_raw is not None else 0,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_sec_override = 0

    # --explore-variant-timeout-safety-margin mirror: auto-derive headroom over the soft kill ratio (neg -> 0).
    explore_variant_timeout_safety_margin_raw = getattr(
        args, "explore_variant_timeout_safety_margin", None,
    )
    try:
        explore_variant_timeout_safety_margin = max(
            0.0,
            float(explore_variant_timeout_safety_margin_raw)
            if explore_variant_timeout_safety_margin_raw is not None else 0.5,
        )
    except (TypeError, ValueError):
        explore_variant_timeout_safety_margin = 0.5

    # KB architecture tags from config.json (architectures + model_type); fresh-launch only (resume rehydrates).
    _cfg_tags = _load_model_config_tags(str(args.model))

    # Reference launch recipe (fresh-launch only): explicit --reference-script
    # wins; else auto-discover an exact-match InferenceX single-node recipe.
    # Lowest-priority base for the baseline server args; fully fail-soft.
    _ref_args, _ref_envs, _ref_model, _ref_source = _resolve_reference_recipe(args)

    state = SharedState(
        session_id=session_id,
        claw_session_id=(os.environ.get("CLAW_SESSION_ID") or "").strip(),
        sandbox_user_id=(os.environ.get("SANDBOX_USER_ID") or "").strip(),
        model_name=Path(args.model).name,
        model_path=str(args.model),
        model_class=args.model_class or "",
        # Advisory architecture profile; fresh-launch only (resume rehydrates, must not clobber). Soft-degrade to {}.
        model_arch=_load_model_arch(
            _workspace_root_resolve(), Path(args.model).name
        ),
        # Architecture-identity tags from config.json stamped into recipe-snapshot extras (fine-tune carries base identity).
        model_architectures=_cfg_tags.get("architectures", []),
        model_type=_cfg_tags.get("model_type", ""),
        framework=os.environ.get("FRAMEWORK", "sglang"),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        # Workload metadata mirrored from CLI/env so downstream prompts see real values (else TP defaults to 1).
        tp=_int_env_or_arg("tp", "TP"),
        # ``ep`` mirrors EP env so fresh-shell resume recovers it for the KB warm-start same-shape filter.
        ep=_int_env_or_arg("ep", "EP"),
        precision=(
            str(getattr(args, "precision", None) or os.environ.get("PRECISION", "") or "").strip()
        ),
        framework_version=_resolve_framework_version(args),
        conc=_int_env_or_arg("conc", "CONC"),
        isl=_int_env_or_arg("isl", "ISL"),
        osl=_int_env_or_arg("osl", "OSL"),
        max_model_len=_int_env_or_arg("max_model_len", "MAX_MODEL_LEN"),
        kernel_enabled=not getattr(args, "no_kernel", False),
        continue_kernel_after_gemm=bool(
            getattr(args, "continue_kernel_after_gemm", True)
        ),
        target_summary=args.target_summary or _default_target_summary(args),
        baseline_tput=0.0,
        cumulative_gain=0.0,
        reference_server_args=_ref_args,
        reference_envs=_ref_envs,
        reference_model=_ref_model,
        reference_source=_ref_source,
        max_minutes=int((args.max_hours or 0) * 60),
        research_lane_capacity=research_lane_capacity,
        gpu_specialist_capacity=gpu_specialist_capacity,
        plateau_overrides=plateau_overrides,
        explore_overtime_kill_ratio=explore_overtime_kill_ratio,
        enable_roofline=bool(
            getattr(args, "enable_roofline", True),
        ),
        # Standalone FRAMEWORK_PR phase; --no-framework skips it (mirrors --no-kernel/kernel_enabled).
        framework_phase_enabled=not bool(getattr(args, "no_framework", False)),
        # --no-explore skips the EXPLORE phase entirely.
        explore_enabled=not bool(getattr(args, "no_explore", False)),
        explore_variant_timeout_sec_override=explore_variant_timeout_sec_override,
        explore_variant_timeout_safety_margin=explore_variant_timeout_safety_margin,
        research_scout_enabled=bool(getattr(args, "research_scout", True)),
        research_scout_interval=max(
            1, int(getattr(args, "research_scout_interval", 3) or 3)
        ),
        target_advisory_enabled=bool(getattr(args, "target_advisory", True)),
        recipe_sediment_enabled=bool(getattr(args, "recipe_sediment", True)),
        # SWEEP-phase post-sweep concurrency sweep flags (on by default); see orchestrator/conc_sweep.py.
        conc_sweep_enabled=bool(getattr(args, "enable_conc_sweep", True)),
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
) -> None:
    """Persist each agent's effective system prompt to ``agents/<role>/system_prompt.snapshot.md``."""
    for role, body in prompts.items():
        target = agent_prompt_snapshot(session_dir, role)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body or "(empty)", encoding="utf-8")

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
    per-round (informational) cumulative gain, the validated cumulative gain
    (with a staleness warning when the optimization stack grew after the last
    validation), the current best config, pruned families, and crash count.
    On ``baseline_failed`` it also surfaces the real terminal root cause from
    ``reports/final.json`` instead of a benign upstream WARN (#465).

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
    print(f"  baseline_tput        : {state.baseline_tput:.1f} tok/s/GPU")
    if session_dir is not None and stop_reason == "baseline_failed":
        failure_summary = _read_failure_summary(session_dir)
        if failure_summary and failure_summary.get("root_cause"):
            print(
                f"  root_cause           : "
                f"[{failure_summary.get('root_cause_type', 'unknown')}] "
                f"{failure_summary.get('root_cause')}"
            )
            if failure_summary.get("server_log"):
                print(
                    f"  server_log           : "
                    f"{failure_summary.get('server_log')}"
                )
    print(
        f"  cumulative_gain      : {state.cumulative_gain:.2f}% "
        f"(per-round sum — informational)"
    )
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
        print(
            "  cumulative_gain_val  : 0.00% "
            "⚠ never validated — no `explore` stack-rebench has succeeded yet"
        )
    print(f"  current_best         : {state.current_best}")
    print(f"  pruned_families      : {state.pruned_families}")
    print(f"  crash_count          : {state.crash_count}")
    _print_kernel_opt_summary_line(state)
    print("===============================================")

def _reconcile_crash_count(state: SharedState, session_dir: Path) -> None:
    """Reconcile persisted ``crash_count`` (state.json + final.json) up to the live in-memory value.

    Only ever raises the persisted value (max), never lowers it; best-effort, never fatal.
    """
    live = int(getattr(state, "crash_count", 0) or 0)

    # 1) state.json — reload, bump if stale, atomic re-save.
    try:
        disk_state = SharedState.load_or_init(session_dir)
        if int(disk_state.crash_count or 0) < live:
            disk_state.crash_count = live
            disk_state.save(session_dir)
    except Exception:  # noqa: BLE001
        log.exception("crash_count reconcile (state.json) failed (non-fatal)")

    # 2) reports/final.json — patch the single field in place if present.
    try:
        from .session_paths import reports_dir
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
        from .orchestrator.kernel_attempt_summary import (
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
        report_path = (
            Path(session_dir) / "reports" / "kernel_optimization_summary.json"
        )
        if report_path.is_file():
            print(f"  kernel_opt_report    : {report_path}")
    except Exception:  # noqa: BLE001 — stdout print must never fail the run
        pass

def _default_target_summary(args: argparse.Namespace) -> str:
    """Compose a human-readable objective summary from the CLI target flags.

    Used as the fallback ``target_summary`` when the operator did not pass an
    explicit ``--target-summary``. The phrasing depends on which target flag is
    set: ``--target-gain`` (percentage), ``--target-tput`` (tok/s/GPU), or
    neither (open-ended optimization within the time budget).

    Args:
        args (argparse.Namespace): Parsed ``optimize`` arguments (reads ``model``,
            ``target_gain``, ``target_tput``, ``max_hours``).

    Returns:
        str: A one-sentence description of the run's objective.
    """
    if args.target_gain:
        return (
            f"Establish baseline on {Path(args.model).name} then drive "
            f"cumulative_gain to >= {args.target_gain}% within "
            f"{args.max_hours}h."
        )
    if args.target_tput:
        return (
            f"Establish baseline on {Path(args.model).name} then reach "
            f"{args.target_tput} tok/s/GPU within {args.max_hours}h."
        )
    return f"Optimize {Path(args.model).name} for up to {args.max_hours}h (no target)."

def _parse_conc_sweep_concs(args: argparse.Namespace) -> list[int]:
    """Parse ``--conc-sweep-concs '1,2,4,8'`` into a list[int]; non-integers warned+dropped, empty -> 1..128 ladder."""
    raw = str(getattr(args, "conc_sweep_concs", "") or "").strip()
    if not raw:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    out: list[int] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            log.warning("conc_sweep: ignoring non-integer CONC token %r", t)
    return out or [1, 2, 4, 8, 16, 32, 64, 128]

def _read_failure_summary(session_dir: Path) -> dict | None:
    """Read ``reports/final.json``'s ``failure_summary`` block, if present.

    Best-effort: returns ``None`` when the file is missing/unreadable or the
    block is absent (e.g. non-failure runs). Used to surface the real terminal
    root cause in the end-of-run summary on ``baseline_failed`` (#465).
    """
    try:
        from .session_paths import reports_dir
        final_json = reports_dir(session_dir) / "final.json"
        data = json.loads(final_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    fs = data.get("failure_summary") if isinstance(data, dict) else None
    return fs if isinstance(fs, dict) else None

def _resolve_reference_recipe(
    args: argparse.Namespace,
) -> tuple[str, dict[str, str], str, str]:
    """Resolve the reference launch recipe for a fresh launch (fail-soft).

    Returns ``(server_args, envs, model, source)``. Discovery is gated on the
    operator opting in via ``--reference-script``:

    - **No ``--reference-script``** → return empty and do NOT auto-discover. The
      run is byte-for-byte identical to a build without this feature (0 degrade);
      this preserves the original behavior.
    - **``--reference-script`` resolves to a usable recipe** → use it.
    - **``--reference-script`` given but unreadable / yields no flags** → fall
      back to auto-discovering an ``exact`` InferenceX single-node match (a
      ``fuzzy`` match is logged as a candidate but NOT applied — a near-name
      model mismatch could break a working baseline).
    """
    source = (getattr(args, "reference_script", None) or "").strip()
    # No flag → original behavior; never auto-discover.
    if not source:
        return ("", {}, "", "")

    framework = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    from .reference_script import (
        discover_reference_script,
        parse_reference_script,
    )
    recipe = parse_reference_script(source, framework=framework)
    if recipe.server_args or recipe.envs:
        print(
            f"Reference script: {source} "
            f"({len(recipe.server_args.split())} arg tokens, "
            f"{len(recipe.envs)} env(s))"
        )
        return (recipe.server_args, dict(recipe.envs), recipe.model or "", source)

    # Explicit source unreadable / yielded nothing → auto-discover instead.
    print(
        f"Reference script: {source} not usable (unreadable or no flags "
        f"lifted); falling back to auto-discovery",
        file=sys.stderr,
    )
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if not inferencex_path:
        return ("", {}, "", "")
    path, tier = discover_reference_script(
        inferencex_path,
        model_path=str(args.model or ""),
        precision=(getattr(args, "precision", None) or os.environ.get("PRECISION", "")),
        gpu_type=str(getattr(args, "gpu_type", None) or os.environ.get("GPU_TYPE", "")),
        framework=framework,
    )
    if path and tier == "exact":
        recipe = parse_reference_script(path, framework=framework)
        print(f"Reference script: {path} (auto-discovered, exact match)")
        return (recipe.server_args, dict(recipe.envs), recipe.model or "", path)
    if path and tier == "fuzzy":
        print(
            f"Reference script: candidate {path} (fuzzy match — NOT applied; "
            f"pass --reference-script {path} to use it)",
            file=sys.stderr,
        )
    return ("", {}, "", "")

def _resolve_session_dir_for_summary(state: SharedState) -> Path | None:
    """Best-effort session_dir lookup ($HYPERLOOM_SESSION_DIR) for the stdout kernel_opt line; ``None`` if unresolved."""
    env_sd = os.environ.get("HYPERLOOM_SESSION_DIR", "").strip()
    if env_sd:
        p = Path(env_sd).expanduser()
        if p.is_dir():
            return p
    return None

