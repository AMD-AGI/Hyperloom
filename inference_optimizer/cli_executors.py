# Copyright Advanced Micro Devices, Inc. All rights reserved.

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


# Declarative action_kind -> ExecutorFn map so tests can introspect what's
# wired; adding a real-executor action MUST update this (test_action_catalogue
# enforces consistency with session_paths._runs_actions()).
_REAL_EXECUTORS_FULL: dict[str, Any] = {
    "baseline":          baseline_executor,
    # replay_warm_recipe reuses BaselineExecutor (same Magpie subprocess,
    # applies warm_start_recipe.best_config; Coordinator interprets it via
    # _promote_replay_warm_recipe).
    "replay_warm_recipe": baseline_executor,
    # profile: Coordinator-internal (--no-enable-roofline path); PolicyGate
    # denies LLM-proposed delegate.
    "profile":           profile_executor,
    "explore":           explore_executor,
    "sweep":             sweep_executor,
    # conc_sweep: Coordinator-internal post-sweep concurrency comparison
    # (disable via --no-enable-conc-sweep); LLM-proposed conc_sweep denied.
    "conc_sweep":        conc_sweep_executor,
    "report":            report_executor,
    "session_breakdown": session_breakdown_executor,
    # recover cleans up leaked VRAM owners (optional rocm-smi --gpureset
    # behind HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1).
    "recover":           recover_executor,
}

# Kernel-owned kinds (dispatched via request{target_agent='kernel'}); no-op
# executors here so SubAgentRunner doesn't raise no_executor on a stale task.
_NOOP_KINDS_KERNEL_ONLY: tuple[str, ...] = tuple(sorted(KERNEL_OWNED_ACTIONS))


def _build_specialist_executor(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    knowledge_plane: Any,
) -> "Callable[[Any], Awaitable[dict]]":
    """Build the specialist executor adapter (async fn(ctx) -> dict wrapping a
    SpecialistRunner). Production uses the subprocess dispatcher (claude in a
    per-task worktree); --specialist-dispatch-mode / missing claude falls back
    to the in-process ClaudeBackend.
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

    # Root the specialist worktree at the same set the prompt + PolicyGate
    # path-validator already trust.
    framework_source_roots = tuple(resolve_source_file_allowlist())
    claude_bin = shutil.which("claude") or ""
    use_subprocess = dispatch_mode != "inprocess" and bool(claude_bin)
    if dispatch_mode == "subprocess" and not claude_bin:
        log.warning(
            "specialist_dispatch_mode=subprocess requested but `claude` "
            "binary not found on PATH; falling back to in-process backend",
        )

    if use_subprocess:
        # Operator --specialist-mcp-config wins; else auto-generate one from
        # the live KnowledgePlane so the subprocess has the PR Monitor MCP
        # server wired (without it mcp__pr_monitor__* tools resolve to nothing).
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
        """Adapter SubAgentRunner.run_task -> SpecialistRunner.run. Always
        returns a dict (even on failure); runner_status preserves the
        SpecialistRunResult distinctions for breakdown analytics."""
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


def _register_executors(
    coordinator: "Coordinator",
    *,
    no_kernel: bool = False,
    compare_against_gpu: str | None = None,
    session_dir: Path | None = None,
    specialist_executor: "Callable[[Any], Awaitable[dict]] | None" = None,
) -> None:
    """Wire all available action executors onto ``coordinator``: the
    _REAL_EXECUTORS_FULL set, kernel-owned no-ops (skipped when no_kernel),
    the always-wired Coordinator-internal executors, and the optional
    specialist executor.
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

    # IntegratePatchExecutor: applies specialist worktree patches via git
    # apply, benches, decides KEEP/REVERT. Single integration point.
    coordinator.sub.register_executor(
        "integrate_patch",
        IntegratePatchExecutor(session_dir=session_dir),
    )

    # FRAMEWORK_PR per-candidate executor — Coordinator-internal only.
    coordinator.sub.register_executor(
        "framework_pr",
        FrameworkPrExecutor(session_dir=session_dir),
    )

    # roofline (profile + trace_analyze): auto-enqueued at PRELUDE + each 10%
    # watermark crossing (independent of --no-kernel), so always registered.
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
