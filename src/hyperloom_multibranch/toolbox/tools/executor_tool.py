# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An existing action executor of the single-flow orchestrator, run as one asynchronous tool.

The executors (profile, build, accuracy, ...) take a runner context and return a result dict. This adapter builds that
context from a tool request, so the executor's own implementation is the one that runs; nothing of it is copied.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.op_shares import ANALYSIS_JSON, OPS_SUMMARY

ExecutorFn = Callable[[RunnerContext], Awaitable[dict]]
_FAILED = frozenset({"failed", "error", "timeout", "cancelled"})


def result_failed(result: Mapping[str, Any]) -> bool:
    """The executors report their own verdict in ``status`` or ``ok``; anything but a failure is a success."""
    return result.get("ok") is False or str(result.get("status") or "").lower() in _FAILED


def describe(result: Mapping[str, Any], *, ok_text: str, keys: tuple[str, ...] = ()) -> str:
    """One line for the inbox: the executor's own error when it failed, otherwise ``ok_text`` plus chosen fields."""
    if result_failed(result):
        reason = (
            result.get("error")
            or result.get("failure_summary")
            or result.get("error_class")
            or result.get("status")
            or "failed"
        )
        return str(reason)
    shown = ", ".join(f"{k}={result[k]}" for k in keys if result.get(k) not in (None, "", [], {}))
    return f"{ok_text}" + (f" ({shown})" if shown else "")


class _Reporting:
    """What both adapters share: a tool's result becomes its :class:`ToolOutput`; a subclass may add to it."""

    _ok_text: str
    _keys: tuple[str, ...]

    def output(self, result: Mapping[str, Any], request: ToolRequest) -> ToolOutput:
        return ToolOutput(
            result=result,
            summary=describe(result, ok_text=self._ok_text, keys=self._keys),
            ok=not result_failed(result),
        )


class ExecutorTool(_Reporting):
    """``name`` -> the executor ``factory(workdir)`` builds; the executor's task kind is ``kind``."""

    def __init__(
        self,
        name: str,
        kind: str,
        factory: Callable[[Path], ExecutorFn],
        *,
        ok_text: str,
        summary_keys: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._kind = kind
        self._factory = factory
        self._ok_text = ok_text
        self._keys = summary_keys

    def context(self, request: ToolRequest, workdir: Path) -> RunnerContext:
        """The runner context the executor expects, with the job's work directory as its session and workspace."""
        job_id = workdir.name
        task = Task(
            task_id=job_id,
            kind=self._kind,
            state="running",
            params=dict(request.params),
            idempotency_key=job_id,
        )
        extra = {"workspace": str(workdir), "session_dir": str(workdir)}
        return RunnerContext(task=task, lease=None, extra=extra)

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        executor = self._factory(workdir)
        result = await executor(self.context(request, workdir))
        return self.output(result, request)


Handler = Callable[..., Awaitable[dict]]


class HandlerTool(_Reporting):
    """A kernel-request handler ``handler(payload, session_dir=...)`` of the orchestrator, run as one tool."""

    local_scratch = False

    def __init__(
        self,
        name: str,
        handler: Handler,
        *,
        ok_text: str,
        summary_keys: tuple[str, ...] = (),
        required: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._handler = handler
        self._ok_text = ok_text
        self._keys = summary_keys
        self._required = required

    def validate(self, request: ToolRequest) -> None:
        missing = [key for key in self._required if not request.params.get(key)]
        if missing:
            raise InvalidRequest(f"{self.name} needs {', '.join(missing)}")

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        if not self.local_scratch:
            result = await self._handler(dict(request.params), session_dir=workdir)
            return self.output(result, request)
        scratch = Path(tempfile.mkdtemp(prefix="hl-tool-"))
        try:
            state = SharedState.state_path(workdir)
            if state.is_file():
                shutil.copy2(state, SharedState.state_path(scratch))
            result = await self._handler(dict(request.params), session_dir=scratch)
            return self.output(_copy_back(result, scratch, workdir), request)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _copy_back(result: Mapping[str, Any], scratch: Path, workdir: Path) -> dict[str, Any]:
    """Copy the files ``result`` names (and the ``analysis.json`` and ops summary beside a report) from ``scratch`` into ``workdir``.

    The handler ran on local disk because shared storage rejects the ``:`` in the names TraceLens' splitter writes;
    such files are never copied. Paths in the result are rewritten to the copies.
    """
    moved: dict[str, str] = {}

    def land(text: Any) -> Any:
        if not isinstance(text, str) or not text.startswith(str(scratch) + "/"):
            return text
        source = Path(text).resolve()
        if not source.is_relative_to(scratch.resolve()):
            return text
        if text not in moved:
            moved[text] = text
            if source.is_file() and ":" not in source.name:
                target = workdir / source.relative_to(scratch.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                moved[text] = str(target)
        return moved[text]

    def walk(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(v) for v in value]
        return land(value)

    out = walk(result)
    for key in ("trace_report_path", "analysis_md_path"):
        for name in (ANALYSIS_JSON, OPS_SUMMARY):
            sibling = Path(str(result.get(key) or "")).parent / name
            if str(sibling).startswith(str(scratch) + "/"):
                land(str(sibling))
    return out


class ScratchHandlerTool(HandlerTool):
    """A handler run in a local scratch directory, its named results copied back into the job's working directory."""

    local_scratch = True
