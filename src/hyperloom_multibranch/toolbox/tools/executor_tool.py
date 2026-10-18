# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An existing action executor of the single-flow orchestrator, run as one asynchronous tool.

The executors (profile, roofline, build, ...) take a runner context and return a result dict. This adapter builds that
context from a tool request, so the executor's own implementation is the one that runs; nothing of it is copied.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolOutput, ToolRequest

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
        result = await self._handler(dict(request.params), session_dir=workdir)
        return self.output(result, request)
