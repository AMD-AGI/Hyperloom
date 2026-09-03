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
                ),
                AgentHook(
                    matcher="Bash|Shell|Task.*|Agent",
                    callback=self._on_pre_disallowed_tool,
                ),
            ]
        )

    async def _on_pre_disallowed_tool(self, _input_data, _tool_use_id, _context) -> dict:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Opportunity analysis is limited to direct read, search, and staging write tools."
                ),
            }
        }

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

Follow this evidence workflow:
1. When analysis.md is referenced and readable, inspect it first and extract its
   hot-operator, time-share, shape, dtype, and source conclusions.
2. When kernel_candidates.json is referenced and readable, inspect every
   candidate before searching the wider source tree.
3. Cross-check both files against the current workload and serving context, raw
   trace, profiler/server logs, kernel_source_resolution.json, model config,
   runtime dispatch code, and editable source. Drop, merge, correct, or reorder
   candidates when those sources disagree.
4. Treat either file as uninformative when it is missing, unreadable, empty,
   stale for the current serving configuration, or contains no actionable
   operator attribution. In that case, continue investigating the other handoff
   evidence and source trees for hot kernels; never return no_opportunity solely
   because TraceLens artifacts are absent or weak.
5. Label each candidate's evidence as measured (trace timing), corroborated
   (runtime dispatch/log hit), or inferred (workload and source reasoning only).
   Never invent a GPU-time percentage for inferred evidence, and rank measured
   candidates ahead of otherwise comparable inferred candidates.

Apply these non-negotiable opportunity rules:
1. Publish only operators that the current end-to-end inference workload
   actually executes under the supplied serving arguments and environment.
   Establish the active runtime path from trace/log evidence or by evaluating
   the deterministic dispatch conditions against the current serving state.
   Skip code that merely exists in the repository but is inactive here.
2. Publish only operators with editable implementation source in one supplied
   Git repository. If the active implementation is available only as a binary,
   shared library, HSACO, or other generated artifact without a tracked editable
   generator source, skip it.
3. Prefer the largest measured end-to-end GPU-time share. Assign lower numeric
   priority values to higher-share operators. When exact percentages are
   unavailable, rank only from clearly labeled corroborated evidence and never
   fabricate a percentage.
4. Derive driver cases from the current workload and serving state, including
   its prefill/decode phases, TP/EP partitioning, concurrency, sequence lengths,
   active backend, tensor shapes, dtypes, layouts, and dispatch boundaries.
5. For every case, correctness and performance must invoke the same operator
   with the same shapes, dtypes, layouts, and semantic inputs. Performance must
   time CUDA/HIP graph replays over preallocated inputs; do not use eager timing
   or silently fall back to eager execution.

Do not start profiling, serving, or benchmark commands. Shell execution is not
available. Use read and search tools for investigation. You may write only under
the supplied staging directory.

For every worthwhile single-operator source rewrite opportunity, create one
subdirectory containing exactly:
  - task.json
  - driver.py

task.json must use this exact top-level structure:
{
  "schema_version": 1,
  "identity": {
    "producer": "forge-loop",
    "kernel_name": "<normalized operator name>",
    "framework": "<framework>",
    "framework_version": "<version>",
    "backend": "<backend>",
    "gpu": "<gpu>"
  },
  "base_commit": "",
  "repo_root": "<absolute Git top-level>",
  "kernel_path": "<repo-relative source path>",
  "operator_name": "<name that normalizes to identity.kernel_name>",
  "driver_path": "driver.py",
  "source_files": ["<repo-relative path>"],
  "target_functions": ["<function>"],
  "shape_cases": [{
    "name": "<workload-derived case>",
    "phase": "<prefill|decode>",
    "shape": {"<dimension>": 1},
    "dtype": "<runtime dtype>"
  }],
  "priority": 0,
  "reason": "<why this measured workload may improve>",
  "evidence": [{
    "level": "<measured|corroborated|inferred>",
    "kind": "<evidence kind>",
    "path": "<path or source reference>"
  }]
}
Do not place identity fields at the top level. evidence must be a JSON list,
even when one detailed evidence object is sufficient. The host pins base_commit
to the current repo HEAD before publication.
All identity values must use normalized lowercase ASCII. For example, write
`"gpu": "mi355x"`, never `"MI355X"`. identity.backend describes the
kernel-building expertise, not the platform; it must be one of `ck`, `flydsl`,
`triton`, `gluon`, `aiter`, `hip`, `hipblaslt`, or `fusion`. Do not publish an
operator whose implementation language has no matching registered backend.
kernel_path and every source_files entry must be tracked, repo-relative files in
the single repo_root at its current HEAD. Put cross-repository source references
in evidence instead of source_files; one task cannot modify multiple repos.

driver.py must cover all known shapes for the six-tuple operator and implement
the forge-loop contract: `python3 driver.py` prints a correctness line such as
`SNR: <db> dB` or `allclose: True/False`; `python3 driver.py --warmup 3
--iters 20 --bench-mode` measures CUDA/HIP graph replays and prints
`case_ms: <case> <ms>` for every case plus one `mean_ms: <ms>`;
`python3 driver.py --profile-run` selects one representative case, runs only
the target kernel for 1-3 synchronized iterations without reference work or
timing output, and exits zero. Do not search other Hyperloom or KernelForge
trees for task or driver examples; this prompt is the authoritative contract.

Publish the strongest plausible task before investigating secondary candidates.
The host and forge-loop own validation, so do not spend the analysis budget
trying to prove an implementation. Do not write state.json and do not modify
source repositories or handoff files.
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
