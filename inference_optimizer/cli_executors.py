"""Action-executor wiring for the CLI.

Holds the declarative real-executor table, the specialist / dynamic-action
executor factories, and ``_register_executors`` which wires everything onto
a live :class:`Coordinator`. Extracted from ``cli.py`` so the entry module
stays focused on parsing + the run flow. This module imports from the
orchestrator packages only — it must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .orchestrator.action_executors import (
    TargetAnalysisExecutor,
    baseline_executor,
    conc_sweep_executor,
    explore_executor,
    recover_executor,
    report_executor,
    session_breakdown_executor,
    sweep_executor,
)
from .orchestrator.action_executors.framework_pr import FrameworkPrExecutor
from .orchestrator.action_executors.integrate_patch import IntegratePatchExecutor
from .orchestrator.action_executors.profile import profile_executor
from .orchestrator.action_executors.roofline import make_roofline_executor
from .orchestrator.backends import ClaudeBackend
from .orchestrator.framework_paths import resolve_source_file_allowlist
from .protocol.action_surfaces import KERNEL_OWNED_ACTIONS

if TYPE_CHECKING:  # pragma: no cover - type-only import to avoid a runtime cycle
    from .orchestrator.coordinator import Coordinator


log = logging.getLogger(__name__)


async def _noop_prep(ctx) -> dict:
    return {"status": "succeeded", "kind": ctx.task.kind, "note": "noop-stub"}


# --- Executor wiring tables ------------------------------------------------
# Declarative mappings of action_kind → ExecutorFn so tests can introspect
# what's actually wired without re-parsing the imperative body of
# ``_register_executors``. Adding a new action with a real executor MUST
# update these tables; ``tests/test_action_catalogue.py`` enforces
# consistency between these tables and ``session_paths._runs_actions()``.

# Real executors enabled in every run mode (kernel + no-kernel).
_REAL_EXECUTORS_FULL: dict[str, Any] = {
    "baseline":          baseline_executor,
    # ``replay_warm_recipe`` runs the same Magpie subprocess as ``baseline``
    # but applies ``warm_start_recipe.best_config`` via ``task.params``; the
    # subprocess pipeline / timeout / salvage / parser are all identical, so
    # we reuse the BaselineExecutor instance (Coordinator interprets the
    # result differently, see ``_promote_replay_warm_recipe``).
    "replay_warm_recipe": baseline_executor,
    # ``profile`` is registered so the Coordinator-internal task path
    # (``--no-enable-roofline``) can dispatch it. PolicyGate denies
    # LLM-proposed ``delegate{action_name='profile'}``, so it is
    # effectively Coordinator-only.
    "profile":           profile_executor,
    "explore":           explore_executor,
    "sweep":             sweep_executor,
    # Coordinator-internal post-sweep concurrency comparison (on by default,
    # disable via ``--no-enable-conc-sweep``); PolicyGate denies LLM-proposed
    # ``conc_sweep`` so the only entry is the SWEEP-completion auto-enqueue.
    "conc_sweep":        conc_sweep_executor,
    "report":            report_executor,
    "session_breakdown": session_breakdown_executor,
    # ``recover`` cleans up leaked VRAM owners and, behind
    # ``HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1``, optionally shells out to
    # ``rocm-smi --gpureset``. See ``action_executors/recover.py``.
    "recover":           recover_executor,
}

# Kernel-owned action kinds dispatched via
# ``request{target_agent='kernel', kind=...}``. The executor body is a
# no-op in this process — actual work happens inside the kernel agent's
# request handlers — but the names must stay registered so SubAgentRunner
# does not raise ``no_executor`` on a stale task.
_NOOP_KINDS_KERNEL_ONLY: tuple[str, ...] = tuple(sorted(KERNEL_OWNED_ACTIONS))


def _build_specialist_executor(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    knowledge_plane: Any,
) -> "Callable[[Any], Awaitable[dict]]":
    """Build the specialist executor adapter (v0.8 §3.5 / §3.13 M5 + PR-A2).

    Returns an ``async fn(ctx) -> dict`` compatible with
    ``SubAgentRunner.ExecutorFn`` that wraps a :class:`SpecialistRunner`.
    Production wires the subprocess dispatcher (each specialist runs in a
    fresh ``claude`` subprocess inside a per-task git worktree); the
    ``--specialist-dispatch-mode`` flag (or a missing ``claude`` binary)
    falls back to the in-process :class:`ClaudeBackend`. The factory
    captures ``session_dir`` + ``knowledge_plane`` once at boot.
    """
    import shutil

    from .orchestrator.specialist_mcp_config import write_specialist_mcp_config
    from .orchestrator.specialist_runner import (
        DEFAULT_SPECIALIST_TOOLS,
        SpecialistRunner,
    )
    from .orchestrator.specialist_subprocess import SpecialistSubprocessConfig

    claude_model = (
        (getattr(args, "specialist_model", None) or args.claude_model)
        .strip()
    )
    max_turns = int(getattr(args, "specialist_max_turns", 8) or 8)
    per_turn_max_seconds = float(
        getattr(args, "specialist_per_turn_max_seconds", 600.0) or 600.0
    )
    dispatch_mode = (
        str(getattr(args, "specialist_dispatch_mode", "subprocess") or "subprocess")
        .strip().lower()
    )

    # Derive framework_source_roots from the canonical resolver so
    # the specialist worktree is rooted at the same set the orchestration
    # prompt + PolicyGate path-validator already trust.
    framework_source_roots = tuple(resolve_source_file_allowlist())
    claude_bin = shutil.which("claude") or ""
    use_subprocess = dispatch_mode != "inprocess" and bool(claude_bin)
    if dispatch_mode == "subprocess" and not claude_bin:
        log.warning(
            "specialist_dispatch_mode=subprocess requested but `claude` "
            "binary not found on PATH; falling back to in-process backend",
        )

    if use_subprocess:
        # Operator-supplied --specialist-mcp-config wins. When unset,
        # auto-generate ``<session_dir>/runtime/specialist_mcp.json``
        # from the live KnowledgePlane so the spawned claude subprocess
        # actually has the PR Monitor MCP server wired (without it the
        # ``mcp__pr_monitor__*`` tool names in the whitelist resolve to
        # nothing and the specialist falls back to WebSearch).
        mcp_config_path: str | None = str(
            getattr(args, "specialist_mcp_config", "") or ""
        ) or None
        if mcp_config_path is None and knowledge_plane is not None:
            try:
                pr_mcp_url = knowledge_plane.specialist_mcp_url()
            except AttributeError:
                pr_mcp_url = ""
            generated = write_specialist_mcp_config(
                session_dir=session_dir,
                pr_monitor_mcp_url=pr_mcp_url,
            )
            if generated is not None:
                mcp_config_path = str(generated)
        sub_config = SpecialistSubprocessConfig(
            claude_executable=claude_bin or "claude",
            model=claude_model,
            framework_source_roots=framework_source_roots,
            mcp_config_path=mcp_config_path,
            per_turn_max_seconds=per_turn_max_seconds,
        )
        runner = SpecialistRunner(
            subprocess_config=sub_config,
            session_dir=session_dir,
            default_tools=DEFAULT_SPECIALIST_TOOLS,
            default_max_turns=max_turns,
            per_turn_max_seconds=per_turn_max_seconds,
            knowledge_plane=knowledge_plane,
        )
    else:
        def _backend_factory(domain: Any) -> Any:
            # in-process Claude path (fallback).
            return ClaudeBackend(
                model=claude_model, max_turns_default=max_turns,
            )

        runner = SpecialistRunner(
            backend_factory=_backend_factory,
            session_dir=session_dir,
            default_tools=DEFAULT_SPECIALIST_TOOLS,
            default_max_turns=max_turns,
            per_turn_max_seconds=per_turn_max_seconds,
            knowledge_plane=knowledge_plane,
        )

    async def _executor(ctx: Any) -> dict:
        """Adapter: SubAgentRunner.run_task → SpecialistRunner.run.

        Always returns a dict (even on failure) so the dispatcher's
        ``transition('succeeded', ...)`` gets a well-formed payload;
        ``runner_status`` preserves SpecialistRunResult's distinctions for
        breakdown.specialist_runs analytics.
        """
        run_result = await runner.run(ctx)
        return {
            "runner_status": run_result.status,
            "task_id": run_result.task_id,
            "domain": run_result.domain,
            "gap_canonical_id": run_result.gap_canonical_id,
            "specialist_done": run_result.specialist_done,
            "turns_used": run_result.turns_used,
            "workspace": run_result.workspace,
            "transcript_path": run_result.transcript_path,
            "done_path": run_result.done_path,
            "error": run_result.error,
            "notes": list(run_result.notes or []),
            "allocated_gpu_ids": list(
                (run_result.specialist_done or {}).get("allocated_gpu_ids") or []
            ),
        }

    return _executor


def _build_dynamic_action_executor(
    args: argparse.Namespace,
) -> "Callable[[Any], Awaitable[dict]]":
    """Build a SubAgentRunner-compatible executor backed by
    :class:`DynamicActionRunner`.

    Production uses a real Claude backend; tests register a
    :class:`MockBackend` through the same path. Falls back to the
    stub executor when no ``claude`` binary is on PATH.
    """
    import shutil

    from .orchestrator.dynamic_action_runner import (
        DEFAULT_TURN_CAP,
        DEFAULT_WALL_CLOCK_BUDGET_SEC,
        DynamicActionRunner,
    )
    from .orchestrator.action_executors.dynamic_action import (
        dynamic_action_executor as _stub_executor,
    )
    from .orchestrator.framework_paths import resolve_source_file_allowlist

    claude_bin = shutil.which("claude") or ""
    if not claude_bin:
        log.warning(
            "dynamic_action: `claude` binary not on PATH; falling "
            "back to the stub executor (empty proposal_set).",
        )
        return _stub_executor

    model = (
        getattr(args, "dynamic_action_model", None)
        or getattr(args, "claude_model", "")
    ).strip()
    turn_cap_raw = getattr(args, "dynamic_action_turn_cap", None)
    turn_cap = int(turn_cap_raw) if turn_cap_raw else DEFAULT_TURN_CAP
    wall_clock_raw = getattr(args, "dynamic_action_wall_clock_sec", None)
    wall_clock = (
        float(wall_clock_raw) if wall_clock_raw
        else DEFAULT_WALL_CLOCK_BUDGET_SEC
    )
    backend = ClaudeBackend(
        model=model, max_turns_default=1, raw_completion=True,
    )
    runner = DynamicActionRunner(
        backend,
        wall_clock_budget_sec=wall_clock,
        turn_cap=turn_cap,
        framework_source_roots=tuple(resolve_source_file_allowlist()),
    )

    async def _executor(ctx: Any) -> dict:
        result = await runner.run(ctx)
        return result.to_dict()

    return _executor


def _register_executors(
    coordinator: "Coordinator",
    *,
    no_kernel: bool = False,
    compare_against_gpu: str | None = None,
    session_dir: Path | None = None,
    specialist_executor: "Callable[[Any], Awaitable[dict]] | None" = None,
    dynamic_action_executor: "Callable[[Any], Awaitable[dict]] | None" = None,
) -> None:
    """Wire all currently-available action executors onto ``coordinator``.

    Real executors come from ``_REAL_EXECUTORS_FULL`` (always). Kernel-owned
    kinds get ``_noop_prep`` (skipped when ``no_kernel``) so SubAgentRunner
    doesn't fail with ``no_executor``. ``target_analysis`` /
    ``integrate_patch`` / ``framework_pr`` / ``roofline`` are always wired
    (Coordinator-internal). ``specialist_executor`` /
    ``dynamic_action_executor`` are wired when provided; the latter falls
    back to the stub executor otherwise.
    """
    for kind, fn in _REAL_EXECUTORS_FULL.items():
        coordinator.sub.register_executor(kind, fn)

    coordinator.sub.register_executor(
        "target_analysis",
        TargetAnalysisExecutor(
            compare_against_gpu=(compare_against_gpu or "").strip(),
            session_dir=session_dir,
        ),
    )

    if specialist_executor is not None:
        coordinator.sub.register_executor("specialist", specialist_executor)

    if dynamic_action_executor is not None:
        coordinator.sub.register_executor(
            "dynamic_action", dynamic_action_executor,
        )
    else:
        from .orchestrator.action_executors.dynamic_action import (
            dynamic_action_executor as _stub_dynamic_action_executor,
        )
        coordinator.sub.register_executor(
            "dynamic_action", _stub_dynamic_action_executor,
        )

    # The real IntegratePatchExecutor reads the specialist's worktree
    # patches, applies them to framework_source_roots via ``git apply``,
    # runs a Magpie bench, and decides KEEP / REVERT. Single integration
    # point — specialists never apply patches themselves.
    coordinator.sub.register_executor(
        "integrate_patch",
        IntegratePatchExecutor(session_dir=session_dir),
    )

    # FRAMEWORK_PR phase per-candidate executor — Coordinator-internal only
    # (PolicyGate denies LLM ``delegate{action='framework_pr'}``).
    coordinator.sub.register_executor(
        "framework_pr",
        FrameworkPrExecutor(session_dir=session_dir),
    )

    # The composite ``roofline`` action runs profile + trace_analyze
    # atomically; Coordinator auto-enqueues it at PRELUDE and on every 10%
    # gain watermark crossing (independent of ``--no-kernel``), so it is
    # unconditionally registered. ``profile`` is the ``--no-enable-roofline``
    # alternative (registered via ``_REAL_EXECUTORS_FULL``).
    coordinator.sub.register_executor(
        "roofline",
        make_roofline_executor(shared_state=coordinator.shared_state),
    )

    if log.isEnabledFor(logging.DEBUG):
        for required_kind in ("roofline", "profile"):
            if required_kind not in coordinator.sub.executor_registry:
                log.debug(
                    "register_executors: %r missing from sub-agent registry "
                    "(no_kernel=%s); PRELUDE analysis task will fail with "
                    "no_executor",
                    required_kind, no_kernel,
                )

    if no_kernel:
        return

    for kind in _NOOP_KINDS_KERNEL_ONLY:
        coordinator.sub.register_executor(kind, _noop_prep)
