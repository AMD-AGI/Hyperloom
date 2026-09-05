# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist dispatcher that leaves the right files behind without an agent.

What a specialist subprocess returns is a workspace -- a done-file, a
heartbeat, a process log, patches -- so this writes one rather than stubbing a
return value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.rehearsal.clock import VirtualClock
from hyperloom.orchestrator.specialists.subprocess_ import SpecialistSubprocessResult

__all__ = ["CRASHED", "COMPLETED", "EMPTY", "HUNG", "ScriptedSpecialist", "SpecialistStep"]

#: The specialist finished and wrote a done-file the runner can consume.
COMPLETED = "completed"

#: The specialist finished and wrote a done-file carrying nothing to apply.
EMPTY = "empty"

#: The specialist died. No done-file, so the runner has to synthesise one.
CRASHED = "crashed"

#: The specialist stopped touching its heartbeat and was reaped for it.
HUNG = "hung"

_OUTCOMES = frozenset({COMPLETED, EMPTY, CRASHED, HUNG})

#: Elapsed time a reaped-for-silence step reports, above any staleness window.
_HUNG_ELAPSED_SEC = 900.0


@dataclass(frozen=True)
class SpecialistStep:
    """One specialist dispatch: how it ended and what it left in the workspace.

    Attributes:
        name: Label for readable assertions.
        outcome: One of the module's four outcome constants.
        done: The ``specialist_done.json`` payload, for a completed step.
        patches: Patch files to write under the workspace, path to content.
        transcript: Lines written to ``process.log``.
        duration_sec: Seconds the step took, charged to the clock.
        exit_code: Exit code; defaults to what the outcome implies.
        error: Error text the dispatcher reports.
    """

    name: str = ""
    outcome: str = COMPLETED
    done: dict[str, Any] = field(default_factory=dict)
    patches: dict[str, str] = field(default_factory=dict)
    transcript: tuple[str, ...] = ()
    duration_sec: float = 120.0
    exit_code: int | None = None
    error: str = ""

    def __post_init__(self) -> None:
        """Reject a step whose outcome cannot be played.

        Raises:
            ValueError: On an unknown outcome.
        """
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"unknown specialist outcome {self.outcome!r}; expected one of {sorted(_OUTCOMES)}")

    @property
    def writes_done_file(self) -> bool:
        """bool: Whether this step leaves a done-file behind."""
        return self.outcome in (COMPLETED, EMPTY)


@dataclass
class ScriptedSpecialist:
    """Stands in for :class:`~..specialists.subprocess_.SpecialistSubprocessDispatcher`.

    Assign one to a runner's ``subprocess_dispatcher`` and every dispatch is
    answered from the script instead of an agent CLI.

    Attributes:
        steps: The dispatches to serve, in order.
        clock: The clock each step's duration is charged to.
        task_ids: Every task id that asked for a step, in order.
        workspaces: Every workspace it wrote into, in order.
    """

    steps: tuple[SpecialistStep, ...]
    clock: VirtualClock = field(default_factory=VirtualClock)
    task_ids: list[str] = field(default_factory=list)
    workspaces: list[Path] = field(default_factory=list)

    @property
    def served(self) -> int:
        """int: How many dispatches have been answered."""
        return len(self.task_ids)

    @property
    def retried(self) -> bool:
        """bool: Whether a later dispatch arrived under a task id no earlier one used."""
        return len(set(self.task_ids)) > 1

    async def run(
        self,
        *,
        task_id: str,
        workspace: Path,
        worktree: Path | None = None,
        **_ignored: Any,
    ) -> SpecialistSubprocessResult:
        """Play the next step into ``workspace`` and report it.

        Args:
            task_id: The dispatching task's id, kept so a retry is visible.
            workspace: ``runs/specialist/<task_id>/``, where the step writes.
            worktree: The per-task worktree; the done-file goes there when there
                is one, matching where the runner looks first.
            **_ignored: The rest of the dispatch surface, unused here.

        Returns:
            SpecialistSubprocessResult: The step's outcome.

        Raises:
            IndexError: When the script has no step left for this dispatch.
        """
        if self.served >= len(self.steps):
            raise IndexError(
                f"scripted specialist has {len(self.steps)} steps; dispatch {self.served + 1} was requested"
            )
        step = self.steps[self.served]
        self.task_ids.append(task_id)
        self.workspaces.append(Path(workspace))

        workspace.mkdir(parents=True, exist_ok=True)
        process_log = workspace / "process.log"
        process_log.write_text("\n".join(step.transcript) + ("\n" if step.transcript else ""), encoding="utf-8")
        # A missing heartbeat is the only difference a liveness consumer can see
        # between a specialist that is thinking and one that has stopped.
        if step.outcome != HUNG:
            (workspace / "heartbeat.json").write_text(
                json.dumps({"task_id": task_id, "status": "running"}) + "\n",
                encoding="utf-8",
            )

        done_root = worktree if worktree is not None else workspace
        payload: dict[str, Any] | None = None
        if step.writes_done_file:
            payload = dict(step.done) if step.outcome == COMPLETED else {}
            done_root.mkdir(parents=True, exist_ok=True)
            (done_root / "specialist_done.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        patches: list[str] = []
        for relative, content in step.patches.items():
            path = workspace / "patches" / str(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            patches.append(str(path))

        elapsed = _HUNG_ELAPSED_SEC if step.outcome == HUNG else step.duration_sec
        self.clock.advance(elapsed)
        self.clock.mark(f"specialist:{step.name or step.outcome}:{task_id}")
        return SpecialistSubprocessResult(
            done_payload=payload,
            exit_code=step.exit_code if step.exit_code is not None else _exit_code_for(step.outcome),
            elapsed_seconds=elapsed,
            stale_heartbeat=step.outcome == HUNG,
            process_log_path=str(process_log),
            patches=patches,
            error=step.error,
        )


def _exit_code_for(outcome: str) -> int | None:
    """Return ``0`` for a step that finished, ``1`` for a crash, ``None`` for a kill."""
    if outcome in (COMPLETED, EMPTY):
        return 0
    if outcome == CRASHED:
        return 1
    return None
