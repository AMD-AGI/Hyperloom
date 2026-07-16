# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Action-executor wiring for the CLI.

Holds the declarative real-executor table, the specialist / dynamic-action
executor factories, and ``_register_executors`` which wires everything onto
a live :class:`Coordinator`. Imports from the orchestrator packages only — it
must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from hyperloom.orchestrator.actions.executors import (
    TargetAnalysisExecutor,
    baseline_executor,
    conc_sweep_executor,
    explore_executor,
    recover_executor,
    report_executor,
    session_breakdown_executor,
    sweep_executor,
)
from hyperloom.orchestrator.actions.executors.framework_agent import FrameworkAgentExecutor
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.executors.profile import profile_executor
from hyperloom.orchestrator.actions.executors.roofline import make_roofline_executor
from hyperloom.orchestrator.roles import ClaudeBackend
from hyperloom.orchestrator.framework.paths import resolve_source_file_allowlist
from ..protocol.action_surfaces import KERNEL_AGENT_OWNED_ACTIONS

if TYPE_CHECKING:  # pragma: no cover - type-only import to avoid a runtime cycle
    from hyperloom.orchestrator.loop.coordinator import Coordinator


log = logging.getLogger(__name__)


def _mcp_servers_from_config(config_path: str | None) -> tuple[str, ...]:
    """Return MCP server names declared by an operator-supplied config file."""
    if not config_path:
        return ()
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("specialist_mcp_config could not be inspected for tool gating: %r", exc)
        return ()
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    if not isinstance(servers, dict):
        return ()
    return tuple(str(name) for name in servers if str(name).strip())


async def _noop_prep(ctx) -> dict:
    """No-op preparation executor used as a stub.

    Args:
        ctx: Action context (only ``ctx.task.kind`` is read).

    Returns:
        A success result envelope tagged as a noop stub.
    """
    return {"status": "succeeded", "kind": ctx.task.kind, "note": "noop-stub"}


# Declarative action_kind -> ExecutorFn map (test_action_catalogue enforces
# consistency with session_paths._runs_actions()).
_REAL_EXECUTORS_FULL: dict[str, Any] = {
    "baseline": baseline_executor,
    # replay_warm_recipe reuses BaselineExecutor, applying warm_start_recipe.best_config.
    "replay_warm_recipe": baseline_executor,
    # profile: Coordinator-internal; PolicyGate denies LLM-proposed delegate.
    "profile": profile_executor,
    "explore": explore_executor,
    "sweep": sweep_executor,
    # conc_sweep: Coordinator-internal post-sweep concurrency comparison.
    "conc_sweep": conc_sweep_executor,
    "report": report_executor,
    "session_breakdown": session_breakdown_executor,
    # recover cleans up leaked VRAM owners.
    "recover": recover_executor,
}

# Kernel-owned kinds; no-op executors here so SubAgentRunner doesn't raise
# no_executor on a stale task.
_NOOP_KINDS_KERNEL_ONLY: tuple[str, ...] = tuple(sorted(KERNEL_AGENT_OWNED_ACTIONS))


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

    Args:
        args: Parsed CLI arguments (specialist model, turns, dispatch mode).
        session_dir: The current session directory.
        knowledge_plane: The live KnowledgePlane used to wire MCP config.

    Returns:
        Callable[[Any], Awaitable[dict]]: An async executor that runs a
        specialist and returns a result envelope dict.
    """
    import shutil

    from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config
    from hyperloom.orchestrator.specialists.runner import (
        DEFAULT_SPECIALIST_TOOLS,
        SpecialistRunner,
    )
    from hyperloom.orchestrator.specialists.domains import DEFAULT_SPECIALIST_MAX_TURNS
    from hyperloom.orchestrator.specialists.subprocess_ import SpecialistSubprocessConfig

    claude_model = (getattr(args, "specialist_model", None) or args.claude_model).strip()
    max_turns = int(
        getattr(args, "specialist_max_turns", DEFAULT_SPECIALIST_MAX_TURNS)
        or DEFAULT_SPECIALIST_MAX_TURNS
    )
    per_turn_max_seconds = float(getattr(args, "specialist_per_turn_max_seconds", 600.0) or 600.0)
    dispatch_mode = str(getattr(args, "specialist_dispatch_mode", "subprocess") or "subprocess").strip().lower()

    # Root the specialist worktree at the set the prompt + PolicyGate trust.
    framework_source_roots = tuple(resolve_source_file_allowlist())
    claude_bin = shutil.which("claude") or ""
    use_subprocess = dispatch_mode != "inprocess" and bool(claude_bin)
    if dispatch_mode == "subprocess" and not claude_bin:
        log.warning(
            "specialist_dispatch_mode=subprocess requested but `claude` "
            "binary not found on PATH; falling back to in-process backend",
        )

    if use_subprocess:
        # Operator --specialist-mcp-config wins; else auto-generate one from the
        # live KnowledgePlane so the subprocess has the PR Monitor MCP wired.
        mcp_config_path: str | None = str(getattr(args, "specialist_mcp_config", "") or "") or None
        forced_mcp_servers = _mcp_servers_from_config(mcp_config_path)
        if mcp_config_path is None and knowledge_plane is not None:
            try:
                pr_mcp_url = knowledge_plane.specialist_mcp_url()
            except AttributeError:
                pr_mcp_url = ""
            try:
                kb_mcp_url = knowledge_plane.cortex_specialist_mcp_url()
                kb_mcp_headers = knowledge_plane.cortex_specialist_mcp_headers()
            except AttributeError:
                kb_mcp_url = ""
                kb_mcp_headers = {}
            generated = write_specialist_mcp_config(
                session_dir=session_dir,
                pr_monitor_mcp_url=pr_mcp_url,
                cortex_kb_mcp_url=kb_mcp_url,
                cortex_kb_mcp_headers=kb_mcp_headers,
            )
            if generated is not None:
                mcp_config_path = str(generated)
                forced_mcp_servers = ()
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
            knowledge_plane=knowledge_plane,
            forced_mcp_servers=forced_mcp_servers,
        )
    else:

        def _backend_factory(domain: Any) -> Any:
            """Build an in-process Claude backend for a specialist domain.

            Args:
                domain: The specialist domain requesting a backend.

            Returns:
                A configured :class:`ClaudeBackend` instance.
            """
            return ClaudeBackend(
                model=claude_model,
                max_turns_default=max_turns,
            )

        runner = SpecialistRunner(
            backend_factory=_backend_factory,
            session_dir=session_dir,
            default_tools=DEFAULT_SPECIALIST_TOOLS,
            default_max_turns=max_turns,
            knowledge_plane=knowledge_plane,
        )

    async def _executor(ctx: Any) -> dict:
        """Adapter SubAgentRunner.run_task -> SpecialistRunner.run. Always
        returns a dict (even on failure); runner_status preserves the
        SpecialistRunResult distinctions for breakdown analytics.

        Args:
            ctx: The action context passed by ``SubAgentRunner.run_task``.

        Returns:
            dict: A result envelope with runner status, task/domain ids,
            transcript paths, and any allocated GPU ids.
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
            "allocated_gpu_ids": list((run_result.specialist_done or {}).get("allocated_gpu_ids") or []),
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
    _REAL_EXECUTORS_FULL set, kernel_agent-owned no-ops (skipped when no_kernel),
    the always-wired Coordinator-internal executors, and the optional
    specialist executor.

    Args:
        coordinator: The live Coordinator to register executors on.
        no_kernel: When True, skip wiring the kernel_agent-owned no-op executors.
        compare_against_gpu: Optional GPU type for the target-analysis executor.
        session_dir: Optional session directory passed to executors.
        specialist_executor: Optional specialist executor to register.
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

    # IntegratePatchExecutor: applies specialist worktree patches, benches,
    # decides KEEP/REVERT.
    coordinator.sub.register_executor(
        "integrate_patch",
        IntegratePatchExecutor(session_dir=session_dir),
    )

    # FRAMEWORK per-candidate executor — Coordinator-internal only.
    coordinator.sub.register_executor(
        "framework",
        FrameworkAgentExecutor(session_dir=session_dir),
    )

    # roofline (profile + trace_analyze): auto-enqueued at PRELUDE + each 10%
    # watermark crossing, so always registered.
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
                    required_kind,
                    no_kernel,
                )

    if no_kernel:
        return

    for kind in _NOOP_KINDS_KERNEL_ONLY:
        coordinator.sub.register_executor(kind, _noop_prep)
