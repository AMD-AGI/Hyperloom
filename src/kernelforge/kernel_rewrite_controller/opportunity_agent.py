# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long-lived Agent that turns handoff evidence into operator rewrite tasks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from kernelforge.agent_backends.base import (
    AgentBackend,
    AgentHook,
    AgentHooks,
    AgentRunSpec,
    AgentToolPolicy,
    with_writable_sandbox,
)
from kernelforge.agent_backends.registry import create_registered_backend
from kernelforge.config import Config
from kernelforge.durable_io import atomic_write_text
from kernelforge.kernel_rewrite_controller.contracts import HandoffBundle
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.scheduler import ANALYSIS_BUDGET_SEC
from kernelforge.kernel_rewrite_controller.task_publisher import (
    TaskPublicationResult,
    publish_complete_staged_tasks,
)
from kernelforge.llm.git import git

ANALYSIS_STATUS_COMPLETED = "completed"
ANALYSIS_STATUS_FAILED = "failed"
ANALYSIS_STATUS_TIMED_OUT = "timed_out"

_ABSOLUTE_PATH_RE = re.compile(r"`(/[^`\n]+)`")
_PUBLISH_POLL_SEC = 0.5


@dataclass(frozen=True)
class OpportunityAnalysisResult:
    """Durable outcome of one opportunity-analysis Agent session."""

    status: str
    reason: str = ""
    published_task_count: int = 0
    rejected_task_count: int = 0
    started_at_unix: float = 0.0
    finished_at_unix: float = 0.0


class _StagingProtection:
    """Restrict Agent writes to the controller-owned staging directory."""

    def __init__(self, staging_root: Path) -> None:
        self.staging_root = staging_root.resolve()

    def hooks(self) -> AgentHooks:
        return AgentHooks(
            pre_tool_use=[
                AgentHook(
                    matcher="Edit|Write|MultiEdit|NotebookEdit",
                    callback=self._on_pre_write,
                )
            ]
        )

    async def _on_pre_write(self, input_data, _tool_use_id, _context) -> dict:
        tool_input = input_data.get("tool_input") or {}
        raw_path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path") or ""
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = self.staging_root / path
        try:
            path.resolve().relative_to(self.staging_root)
            return {}
        except ValueError:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "Opportunity analysis may only write task.json and driver.py "
                        "under the supplied staging directory."
                    ),
                }
            }


def _ensure_agent_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if (path / ".git").exists():
        return
    git("init", cwd=path)
    git("add", "-A", cwd=path)
    git(
        "-c",
        "user.name=KernelForge",
        "-c",
        "user.email=kernel-forge@localhost",
        "commit",
        "--allow-empty",
        "-m",
        "kernel rewrite opportunity analysis baseline",
        cwd=path,
    )


def _additional_directories(handoff: HandoffBundle) -> list[str]:
    directories = {handoff.root}
    for document in (handoff.workload, handoff.serving_context, handoff.trace_evidence):
        for raw in _ABSOLUTE_PATH_RE.findall(document):
            path = Path(raw).expanduser()
            if path.exists():
                directories.add(path if path.is_dir() else path.parent)
    return [str(path.resolve()) for path in sorted(directories, key=str)]


def _system_prompt() -> str:
    return """\
You are the KernelForge kernel rewrite opportunity analyst.

Analyze the supplied workload, serving context, trace evidence, and source trees.
TraceLens conclusions and kernel_candidates.json are hints, not authority. Inspect
the available evidence and correct them when necessary.

Do not start profiling, serving, or benchmark commands. Shell execution is not
available. Use read and search tools for investigation. You may write only under
the supplied staging directory.

For every worthwhile single-operator source rewrite opportunity, create one
subdirectory containing exactly:
  - task.json
  - driver.py

task.json must contain schema_version=1, the six identity fields (producer must
be "forge-loop"), repo_root, kernel_path relative to that repo, operator_name,
driver_path="driver.py", source_files, target_functions, shape_cases, priority,
reason, and evidence. Set base_commit to an empty string; the host pins the
current repo HEAD before publication.

driver.py must cover all known shapes for the six-tuple operator and implement
the existing forge-loop correctness and benchmark output contract. Publish each
complete task directory as soon as it is ready. Do not write state.json and do
not modify source repositories or handoff files.
"""


def _user_prompt(handoff: HandoffBundle, staging_root: Path) -> str:
    return f"""\
# Controller staging directory

`{staging_root}`

# workload.md

{handoff.workload}

# serving-context.md

{handoff.serving_context}

# trace-evidence.md

{handoff.trace_evidence}
"""


def _write_analysis_result(layout: ControllerLayout, result: OpportunityAnalysisResult) -> None:
    atomic_write_text(
        layout.agent_root / "analysis-result.json",
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
    )


class OpportunityAnalysisAgent:
    """Run one provider-backed analysis session and publish complete tasks."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        timeout_sec: int,
        max_turns: int,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        if not backend.capabilities.stop_hooks:
            raise ValueError("opportunity analysis requires a provider with tool hooks")
        self.backend = backend
        self.timeout_sec = int(timeout_sec)
        self.max_turns = int(max_turns)

    async def run(
        self,
        *,
        handoff: HandoffBundle,
        layout: ControllerLayout,
    ) -> OpportunityAnalysisResult:
        started = time.time()
        layout.agent_staging_root.mkdir(parents=True, exist_ok=True)
        if self.backend.capabilities.requires_workspace_cwd:
            _ensure_agent_workspace(layout.agent_staging_root)
        progress: list[str] = []
        spec = AgentRunSpec(
            system_prompt=_system_prompt(),
            user_prompt=_user_prompt(handoff, layout.agent_staging_root),
            cwd=str(layout.agent_staging_root),
            writable=True,
            timeout_sec=self.timeout_sec,
            additional_directories=_additional_directories(handoff),
            protected_paths=[
                str(handoff.root / "workload.md"),
                str(handoff.root / "serving-context.md"),
                str(handoff.root / "trace-evidence.md"),
            ],
            allow_untracked=True,
            allow_dirty_baseline=True,
            tool_policy=AgentToolPolicy(
                read=True,
                search=True,
                write=True,
                shell=False,
                max_turns=self.max_turns,
                permission_mode=os.environ.get("FORGE_PERMISSION_MODE", "acceptEdits"),
                bare=False,
            ),
            hooks=_StagingProtection(layout.agent_staging_root).hooks(),
            progress_log=progress,
        )

        backend_task = asyncio.create_task(self.backend.run(spec))
        publications: dict[str, TaskPublicationResult] = {}
        status = ANALYSIS_STATUS_COMPLETED
        reason = ""
        deadline = time.monotonic() + self.timeout_sec
        try:
            while not backend_task.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = ANALYSIS_STATUS_TIMED_OUT
                    reason = f"opportunity analysis exceeded {self.timeout_sec}s"
                    backend_task.cancel()
                    break
                await asyncio.wait({backend_task}, timeout=min(_PUBLISH_POLL_SEC, remaining))
                for result in publish_complete_staged_tasks(layout):
                    publications[result.source_dir.name] = result
            if not backend_task.cancelled():
                try:
                    agent_result = await backend_task
                    if agent_result.end_reason == "timeout":
                        status = ANALYSIS_STATUS_TIMED_OUT
                        reason = agent_result.stderr_tail or "opportunity analysis timed out"
                except asyncio.CancelledError:
                    if status != ANALYSIS_STATUS_TIMED_OUT:
                        raise
                except Exception as error:
                    status = ANALYSIS_STATUS_FAILED
                    reason = f"opportunity analysis failed: {error}"
        finally:
            if not backend_task.done():
                backend_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await backend_task
            for result in publish_complete_staged_tasks(layout):
                publications[result.source_dir.name] = result
            if progress:
                atomic_write_text(layout.agent_root / "progress.log", "\n".join(progress) + "\n")

        published = sum(result.published for result in publications.values())
        rejected = sum(not result.published for result in publications.values())
        outcome = OpportunityAnalysisResult(
            status=status,
            reason=reason,
            published_task_count=published,
            rejected_task_count=rejected,
            started_at_unix=started,
            finished_at_unix=time.time(),
        )
        _write_analysis_result(layout, outcome)
        return outcome


def run_opportunity_analysis(
    *,
    handoff: HandoffBundle,
    layout: ControllerLayout,
    controller_deadline_unix: float,
    backend: AgentBackend | None = None,
) -> OpportunityAnalysisResult:
    """Run the opportunity Agent within the one-hour/controller deadline cap."""
    remaining = max(0.0, float(controller_deadline_unix) - time.time())
    timeout_sec = max(1, int(min(ANALYSIS_BUDGET_SEC, remaining)))
    layout.agent_root.mkdir(parents=True, exist_ok=True)
    try:
        selected_backend = backend
        config = None
        if selected_backend is None:
            config = Config.from_env(
                workspace=str(layout.agent_root),
                agent_timeout_sec=timeout_sec,
            )
            runtime = with_writable_sandbox(config.agent_runtime())
            selected_backend = create_registered_backend(
                runtime,
                probe_cwd=str(layout.agent_root),
            )
        agent = OpportunityAnalysisAgent(
            backend=selected_backend,
            timeout_sec=timeout_sec,
            max_turns=config.max_turns if config is not None else 500,
        )
        return asyncio.run(agent.run(handoff=handoff, layout=layout))
    except Exception as error:
        result = OpportunityAnalysisResult(
            status=ANALYSIS_STATUS_FAILED,
            reason=f"opportunity analysis setup failed: {error}",
            started_at_unix=time.time(),
            finished_at_unix=time.time(),
        )
        _write_analysis_result(layout, result)
        return result


__all__ = [
    "ANALYSIS_STATUS_COMPLETED",
    "ANALYSIS_STATUS_FAILED",
    "ANALYSIS_STATUS_TIMED_OUT",
    "OpportunityAnalysisAgent",
    "OpportunityAnalysisResult",
    "run_opportunity_analysis",
]
