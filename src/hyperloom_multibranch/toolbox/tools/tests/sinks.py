# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fakes shared by the tool tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom_multibranch.contracts.notebook.records import ToolResultPointer
from hyperloom_multibranch.toolbox.tools.base import ToolOutput, ToolRequest


class RecordingSink:
    """Keeps every pointer it is given; refuses the first ``refuse`` pushes."""

    def __init__(self, refuse: int = 0) -> None:
        self.pointers: list[ToolResultPointer] = []
        self.refuse = refuse
        self.calls = 0

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        self.calls += 1
        if self.calls <= self.refuse:
            raise ConnectionError("inbox unavailable")
        self.pointers.append(pointer)


class ScriptedTool:
    """A tool that returns, raises or blocks as the test says."""

    def __init__(self, name: str = "scripted", *, output: ToolOutput | None = None, raises: Exception | None = None):
        self.name = name
        self.output = output or ToolOutput(result={"n": 1}, summary="done")
        self.raises = raises
        self.started = asyncio.Event()
        self.block: asyncio.Event | None = None
        self.seen: list[tuple[ToolRequest, Path]] = []

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        self.seen.append((request, workdir))
        self.started.set()
        if self.block is not None:
            await self.block.wait()
        if self.raises is not None:
            raise self.raises
        return self.output
