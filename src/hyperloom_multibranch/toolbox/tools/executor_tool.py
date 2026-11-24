# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A handler of the single-flow orchestrator, run as one asynchronous tool.

A handler takes a payload and a session directory and returns a result dict. This adapter passes it the tool request's
params, so the handler's own implementation is the one that runs; nothing of it is copied.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from hyperloom_common.shared_storage import copy_shared_file, make_shared_dirs
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.op_shares import ANALYSIS_JSON, OPS_SUMMARY

_FAILED = frozenset({"failed", "error", "timeout", "cancelled"})


def result_failed(result: Mapping[str, Any]) -> bool:
    """The handlers report their own verdict in ``status`` or ``ok``; anything but a failure is a success."""
    return result.get("ok") is False or str(result.get("status") or "").lower() in _FAILED


def describe(result: Mapping[str, Any], *, ok_text: str) -> str:
    """One line for the inbox: the handler's own error when it failed, otherwise ``ok_text``."""
    if result_failed(result):
        reason = result.get("error") or result.get("error_class") or result.get("status") or "failed"
        return str(reason)
    return ok_text


Handler = Callable[..., Awaitable[dict]]


class HandlerTool:
    """A kernel-request handler ``handler(payload, session_dir=...)`` of the orchestrator, run as one tool."""

    local_scratch = False

    def __init__(
        self,
        name: str,
        handler: Handler,
        *,
        ok_text: str,
        required: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._handler = handler
        self._ok_text = ok_text
        self._required = required

    def validate(self, request: ToolRequest) -> None:
        missing = [key for key in self._required if not request.params.get(key)]
        if missing:
            raise InvalidRequest(f"{self.name} needs {', '.join(missing)}")

    def output(self, result: Mapping[str, Any], request: ToolRequest) -> ToolOutput:
        """The tool's result as its :class:`ToolOutput`; a subclass may add to it."""
        return ToolOutput(result=result, summary=describe(result, ok_text=self._ok_text), ok=not result_failed(result))

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
                make_shared_dirs(target.parent)  # the job's directory: a later leg, maybe another account, reads it
                copy_shared_file(source, target)
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
