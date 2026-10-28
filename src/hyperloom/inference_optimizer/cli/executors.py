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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from hyperloom.orchestrator.actions.executors import (
    TargetAnalysisExecutor,
    baseline_executor,
    conc_sweep_executor,
    explore_executor,
    report_executor,
    session_breakdown_executor,
)
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
from hyperloom.orchestrator.actions.executors.profile import profile_executor
from hyperloom.orchestrator.actions.executors.roofline import RooflineExecutor
from hyperloom.inference_optimizer.framework_paths import resolve_kernel_search_roots

if TYPE_CHECKING:  # pragma: no cover - type-only import to avoid a runtime cycle
    from hyperloom.orchestrator.loop.coordinator import Coordinator


# Declarative action_kind -> ExecutorFn map.
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
}


def _required_cli(path: str, missing: str) -> str:
    if not path:
        raise RuntimeError(missing)
    return path


def _specialist_agent_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """The agent CLI every specialist runs on: its backend, model and executable.

    Resolved once at boot so the three are chosen together and a later
    dispatch cannot disagree with them.

    Raises:
        RuntimeError: If no runtime is found for the selected agent CLI.
    """
    from hyperloom.common.llm_config import (
        AGENT_BACKEND_CODEX,
        AGENT_BACKEND_DEEPSEEK,
        resolve_agent_backend,
        resolve_forge_llm_model,
    )
    from hyperloom.common.agent_cli import resolve_claude_executable, resolve_codex_executable
    from hyperloom.orchestrator.specialists.subprocess_ import resolve_dsh_executable

    backend = resolve_agent_backend()
    config: dict[str, Any] = {"agent_backend": backend, "claude_executable": "claude"}
    if backend == AGENT_BACKEND_DEEPSEEK:
        default_model = resolve_forge_llm_model(backend)
        config["dsh_executable"] = _required_cli(
            resolve_dsh_executable(),
            "INFERENCE_OPTIMIZER_AGENT_BACKEND=deepseek runs specialists on the dsh CLI, but none was "
            "found on PATH or in the DeepSeek Harness SDK runtime. Install the 'deepseek' extra "
            "(pip install 'hyperloom-inference_optimizer[deepseek]').",
        )
    elif backend == AGENT_BACKEND_CODEX:
        default_model = str(args.codex_model).strip()
        config["codex_executable"] = _required_cli(
            resolve_codex_executable(),
            "this deployment configures only the OpenAI side, so specialists must run on "
            "the codex CLI, but no codex runtime was found. Install `codex` on PATH or "
            "install the Codex SDK runtime (pip install 'hyperloom-inference_optimizer[llm]'). "
            "Falling back to the claude CLI here would fail to authenticate on every "
            "specialist task.",
        )
    else:
        default_model = str(args.claude_model).strip()
        config["claude_executable"] = _required_cli(
            resolve_claude_executable(),
            "specialists run on the claude CLI, but none was found in $HYPERLOOM_CLAUDE_CLI_PATH, "
            "$GEAK_CLAUDE_BIN or on PATH. src/hyperloom/inference_optimizer/assets/install.sh puts it on PATH.",
        )
    config["model"] = str(getattr(args, "specialist_model", None) or "").strip() or default_model
    return config


def _build_specialist_executor(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    knowledge_plane: Any,
) -> "Callable[[Any], Awaitable[dict]]":
    """Build the specialist executor adapter (async fn(ctx) -> dict wrapping a
    SpecialistRunner). Each task spawns the agent CLI the deployment's
    credentials can drive (``claude`` on the Anthropic side, ``codex`` on the
    OpenAI side) in a per-task worktree.

    Args:
        args: Parsed CLI arguments (specialist model and turns).
        session_dir: The current session directory.
        knowledge_plane: The live KnowledgePlane used to wire MCP config.

    Returns:
        Callable[[Any], Awaitable[dict]]: An async executor that runs a
        specialist and returns a result envelope dict.

    Raises:
        RuntimeError: If no runtime is found for the selected agent CLI: no
            ``claude`` CLI, or no Codex runtime.
    """
    from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config
    from hyperloom.orchestrator.specialists.runner import SpecialistRunner
    from hyperloom.orchestrator.specialists.domains import DEFAULT_SPECIALIST_MAX_TURNS
    from hyperloom.orchestrator.specialists.subprocess_ import SpecialistSubprocessConfig

    max_turns = int(getattr(args, "specialist_max_turns", DEFAULT_SPECIALIST_MAX_TURNS) or DEFAULT_SPECIALIST_MAX_TURNS)

    framework_source_roots = tuple(resolve_kernel_search_roots())
    # Operator --specialist-mcp-config wins; else auto-generate one from the
    # live KnowledgePlane so the subprocess has the PR Monitor MCP wired.
    mcp_config_path: str | None = str(getattr(args, "specialist_mcp_config", "") or "") or None
    if mcp_config_path is None and knowledge_plane is not None:
        generated = write_specialist_mcp_config(
            session_dir=session_dir,
            pr_monitor_mcp_url=knowledge_plane.specialist_mcp_url(),
        )
        if generated is not None:
            mcp_config_path = str(generated)
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(
            **_specialist_agent_cli_config(args),
            framework_source_roots=framework_source_roots,
            mcp_config_path=mcp_config_path,
        ),
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
        TargetAnalysisExecutor(compare_against_gpu=(compare_against_gpu or "").strip()),
    )

    if specialist_executor is not None:
        coordinator.sub.register_executor("specialist", specialist_executor)

    # IntegratePatchExecutor: applies specialist worktree patches, benches,
    # decides KEEP/REVERT.
    coordinator.sub.register_executor(
        "integrate_patch",
        IntegratePatchExecutor(session_dir=session_dir),
    )

    # roofline (profile + trace_analyze): auto-enqueued at PRELUDE + each 10%
    # watermark crossing, so always registered.
    coordinator.sub.register_executor(
        "roofline",
        RooflineExecutor(shared_state=coordinator.shared_state),
    )

    # targeted_build: off-loop compiled-component builds.  Coordinator-internal
    # only (in COORDINATOR_INTERNAL_ACTIONS); dispatched by the enablement phase,
    # not proposed by LLM agents.  The executor runs via run_task_registered so
    # cancel_inflight_actions can stop it at shutdown or on a spent budget.
    coordinator.sub.register_executor(
        "targeted_build",
        TargetedBuildExecutor(),
    )

    # kernel_agent: the KERNEL_AGENT phase's whole pipeline, run under the task's lanes.
    coordinator.sub.register_executor("kernel_agent", lambda ctx: coordinator.phase_kernel.run_agent(ctx))
