# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Action-executor wiring for the CLI.

Holds the declarative real-executor table, the specialist / dynamic-action
executor factories, and ``_register_executors`` which wires everything onto
a live :class:`Coordinator`. Imports from the orchestrator packages only — it
must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import argparse
import logging
import os
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
)
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
from hyperloom.orchestrator.actions.executors.profile import profile_executor
from hyperloom.orchestrator.actions.executors.roofline import make_roofline_executor
from hyperloom.orchestrator.roles import ClaudeBackend
from hyperloom.orchestrator.framework.paths import resolve_source_file_allowlist

if TYPE_CHECKING:  # pragma: no cover - type-only import to avoid a runtime cycle
    from hyperloom.orchestrator.loop.coordinator import Coordinator


log = logging.getLogger(__name__)


# Declarative action_kind -> ExecutorFn map. Keep in sync with
# session_paths._RUNS_ACTIONS (not enforced by a test).
_REAL_EXECUTORS_FULL: dict[str, Any] = {
    "baseline": baseline_executor,
    # replay_warm_recipe reuses BaselineExecutor, applying warm_start_recipe.best_config.
    "replay_warm_recipe": baseline_executor,
    # profile: Coordinator-internal; PolicyGate denies LLM-proposed delegate.
    "profile": profile_executor,
    "explore": explore_executor,
    # conc_sweep: the Coordinator-internal CONC-ladder benchmark, and the only
    # sweep there is.
    "conc_sweep": conc_sweep_executor,
    "report": report_executor,
    "session_breakdown": session_breakdown_executor,
    # recover cleans up leaked VRAM owners.
    "recover": recover_executor,
}


def _build_specialist_executor(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    knowledge_plane: Any,
) -> "Callable[[Any], Awaitable[dict]]":
    """Build the specialist executor adapter (async fn(ctx) -> dict wrapping a
    SpecialistRunner). Production uses the subprocess dispatcher, spawning the
    agent CLI the deployment's credentials can drive (``claude`` on the Anthropic
    side, ``codex`` on the OpenAI side) in a per-task worktree.
    Explicit in-process dispatch uses the matching agent SDK backend.

    Args:
        args: Parsed CLI arguments (specialist model, turns, dispatch mode).
        session_dir: The current session directory.
        knowledge_plane: The live KnowledgePlane used to wire MCP config.

    Returns:
        Callable[[Any], Awaitable[dict]]: An async executor that runs a
        specialist and returns a result envelope dict.

    Raises:
        RuntimeError: If subprocess dispatch selects Codex but no Codex runtime
            is installed.
    """
    import shutil

    from hyperloom.common.hermes_runtime import hermes_external_sandbox_enabled, resolve_hermes_executable
    from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config
    from hyperloom.orchestrator.specialists.runner import SpecialistRunner
    from hyperloom.orchestrator.specialists.domains import DEFAULT_SPECIALIST_MAX_TURNS
    from hyperloom.orchestrator.specialists.subprocess_ import (
        AGENT_BACKEND_CODEX,
        AGENT_BACKEND_HERMES,
        SpecialistSubprocessConfig,
        resolve_codex_executable,
        resolve_specialist_agent_backend,
    )

    max_turns = int(getattr(args, "specialist_max_turns", DEFAULT_SPECIALIST_MAX_TURNS) or DEFAULT_SPECIALIST_MAX_TURNS)
    per_turn_max_seconds = float(getattr(args, "specialist_per_turn_max_seconds", 600.0) or 600.0)
    dispatch_mode = str(getattr(args, "specialist_dispatch_mode", "subprocess") or "subprocess").strip().lower()

    # Root the specialist worktree at the set the prompt + PolicyGate trust.
    framework_source_roots = tuple(resolve_source_file_allowlist())
    # Resolve the agent CLI once here so the backend, its executable and its
    # model are chosen together and a later dispatch cannot disagree with them.
    agent_backend = resolve_specialist_agent_backend()
    specialist_override = str(getattr(args, "specialist_model", None) or "").strip()
    selected_model = specialist_override or (
        os.environ.get("HYPERLOOM_HERMES_MODEL", "").strip() or str(args.codex_model).strip()
        if agent_backend == AGENT_BACKEND_HERMES
        else str(args.codex_model).strip()
        if agent_backend == AGENT_BACKEND_CODEX
        else str(args.claude_model).strip()
    )
    codex_bin = ""
    claude_bin = ""
    hermes_bin = ""
    if agent_backend == AGENT_BACKEND_CODEX:
        codex_bin = resolve_codex_executable()
        if dispatch_mode != "inprocess" and not codex_bin:
            raise RuntimeError(
                "this deployment configures only the OpenAI side, so specialists must run on "
                "the codex CLI, but no codex runtime was found. Install `codex` on PATH or "
                "install the Codex SDK runtime (pip install 'hyperloom-inference_optimizer[llm]'). "
                "Falling back to the claude CLI here would fail to authenticate on every "
                "specialist task."
            )
        agent_bin = codex_bin
    elif agent_backend == AGENT_BACKEND_HERMES:
        hermes_bin = resolve_hermes_executable()
        if not hermes_bin:
            raise RuntimeError("Hermes specialists requested but no hermes runtime was found")
        if dispatch_mode == "inprocess":
            raise RuntimeError("Hermes specialists use the original subprocess/worktree contract")
        agent_bin = hermes_bin
    else:
        claude_bin = shutil.which("claude") or ""
        agent_bin = claude_bin
    use_subprocess = dispatch_mode != "inprocess" and bool(agent_bin)
    if dispatch_mode == "subprocess" and not agent_bin:
        log.warning(
            "specialist_dispatch_mode=subprocess requested but `%s` "
            "binary not found on PATH; falling back to in-process backend",
            agent_backend,
        )

    if use_subprocess:
        # Operator --specialist-mcp-config wins; else auto-generate one from the
        # live KnowledgePlane so the subprocess has the PR Monitor MCP wired.
        mcp_config_path: str | None = str(getattr(args, "specialist_mcp_config", "") or "") or None
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
        # This setting controls the Claude runtime only. Codex containment is
        # resolved independently through the canonical sandbox policy.
        specialist_permission_mode = os.environ.get("HYPERLOOM_SPECIALIST_PERMISSION_MODE", "").strip()
        sub_config_kwargs: dict[str, Any] = {
            "agent_backend": agent_backend,
            "claude_executable": claude_bin or "claude",
            "codex_executable": codex_bin,
            "hermes_executable": hermes_bin or "hermes",
            "hermes_profile": os.environ.get("HYPERLOOM_HERMES_PROFILE", "hyperloomfaithful").strip(),
            "hermes_provider": os.environ.get("HYPERLOOM_HERMES_PROVIDER", "openai-codex").strip(),
            "hermes_external_sandbox": hermes_external_sandbox_enabled(),
            "model": selected_model,
            "framework_source_roots": framework_source_roots,
            "mcp_config_path": mcp_config_path,
            "per_turn_max_seconds": per_turn_max_seconds,
        }
        if specialist_permission_mode:
            sub_config_kwargs["permission_mode"] = specialist_permission_mode
        sub_config = SpecialistSubprocessConfig(**sub_config_kwargs)
        runner = SpecialistRunner(
            subprocess_config=sub_config,
            session_dir=session_dir,
            default_max_turns=max_turns,
        )
    else:

        def _backend_factory(domain: Any) -> Any:
            """Build the selected in-process agent SDK backend.

            Args:
                domain: The specialist domain requesting a backend.

            Returns:
                A configured Claude or Codex Agent SDK backend.
            """
            if agent_backend == AGENT_BACKEND_CODEX:
                from hyperloom.orchestrator.roles.codex_agent import CodexAgentBackend

                runtime_root = session_dir / "runtime" / "codex-specialist"
                return CodexAgentBackend(
                    model=selected_model,
                    cwd=runtime_root,
                    writable_roots=(runtime_root,),
                    call_timeout_s=per_turn_max_seconds,
                )
            from hyperloom.orchestrator.roles.agent_role import SPECIALIST_INTENTS

            return ClaudeBackend(
                model=selected_model,
                max_turns_default=max_turns,
                allowed_intents=SPECIALIST_INTENTS,
                # Same label the subprocess dispatch mode reports, so switching
                # modes does not move this spend between components.
                attribution_component="specialist",
                attribution_operation="run_agent",
            )

        runner = SpecialistRunner(
            backend_factory=_backend_factory,
            session_dir=session_dir,
            default_max_turns=max_turns,
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
    compare_against_gpu: str | None = None,
    session_dir: Path | None = None,
    specialist_executor: "Callable[[Any], Awaitable[dict]] | None" = None,
) -> None:
    """Wire all available action executors onto ``coordinator``: the
    _REAL_EXECUTORS_FULL set, the always-wired Coordinator-internal executors,
    and the optional specialist executor.

    Kernel-owned actions get no executor: they are REQUEST-only.

    Args:
        coordinator: The live Coordinator to register executors on.
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

    # roofline (profile + trace_analyze): auto-enqueued at PRELUDE + each 10%
    # watermark crossing, so always registered.
    coordinator.sub.register_executor(
        "roofline",
        make_roofline_executor(shared_state=coordinator.shared_state),
    )

    # targeted_build: off-loop compiled-component builds.  Coordinator-internal
    # only (in INTERNAL_ONLY_ACTION_NAMES); dispatched by the enablement phase,
    # not proposed by LLM agents.  The executor runs via run_task_registered so
    # cancel_inflight_actions can stop it at shutdown or on a spent budget.
    coordinator.sub.register_executor(
        "targeted_build",
        TargetedBuildExecutor(),
    )

    if log.isEnabledFor(logging.DEBUG):
        for required_kind in ("roofline", "profile"):
            if required_kind not in coordinator.sub.executor_registry:
                log.debug(
                    "register_executors: %r missing from sub-agent registry; "
                    "PRELUDE analysis task will fail with no_executor",
                    required_kind,
                )
