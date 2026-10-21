# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdio MCP server that lets a read-only specialist measure one variant."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from kernelforge.orchestrator.specialist_probe import TOOL_NAMES, ProbeRequestError, ProbeSession
from kernelforge.tools import mcp_stdio

#: Re-exported so the two servers stay one import away from the shared wire
#: format; both spellings name the same objects.
InvalidParamsError = mcp_stdio.InvalidParamsError

SERVER_NAME = "kernelforge-specialist-probe"

__all__ = ["SERVER_NAME", "TOOL_DEFINITIONS", "TOOL_NAMES", "InvalidParamsError", "ProbeServer", "main"]


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "probe_variant",
        "description": (
            "Time ONE named case at ONE point in the dispatch-constant space, "
            "by re-running the workspace driver in a scratch directory of its "
            "own. Nothing in the canonical workspace is edited. Use it to "
            "settle a question about a constant that you would otherwise have "
            "to argue. A result from this tool is exploratory, not an "
            "acceptance-gate result.\n"
            "Three numbers come back with every result: how many probes and "
            "how many seconds of wall clock are left of this ROUND's budget, "
            "both shared with the other specialists analysing it at the same "
            "time, and how many seconds are left of YOUR OWN session -- so a "
            "spent round budget and a session that is nearly over are things "
            "you can tell apart. No probe may run that would "
            "leave too little of your session to write the analysis, and a "
            "refused or unavailable probe costs one of the count just as a "
            "measured one does. The GPU is measured one run at a time, so a "
            "probe may spend part of its budget waiting for the device and be "
            "abandoned if it does not come free. When a result says the budget "
            "or the session clock is spent, stop probing and report the "
            "remaining questions as unmeasured."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for the question this probe settles, cited "
                        "from the analysis, e.g. 'block-1024-vs-256'."
                    ),
                },
                "driver_script": {
                    "type": "string",
                    "description": (
                        "Benchmark driver to run, as a path inside the "
                        "canonical workspace. A path outside it is refused."
                    ),
                },
                "case_id": {
                    "type": "string",
                    "description": "Scored case to time, by name.",
                },
                "constants": {
                    "type": "object",
                    "description": (
                        "Declared dispatch constants to vary, upper-case name "
                        "-> value; they reach the driver as environment "
                        "variables named FORGE_SWEEP_<NAME>, which only a knob "
                        "instrumented for forge reads. Empty measures the "
                        "unmodified source as this probe's own reference."
                    ),
                },
                "prefix_constants": {
                    "type": "boolean",
                    "description": (
                        "Default true. Set false to export each name EXACTLY as "
                        "written, which is the only way to reach a knob the "
                        "source already reads under its own name (e.g. "
                        "GPTOSS_SWIGLU_MXFP4_BF16_BOUND). Such a knob does not "
                        "print forge's 'sweep_const:' echo, so the result comes "
                        "back marked unread and unconfirmed: measure a probe "
                        "with no constants in the same round and compare, or "
                        "the number says nothing. A name the measurement itself "
                        "runs on (PATH, HIP_VISIBLE_DEVICES, a cache directory) "
                        "is refused: those are not knobs of the kernel."
                    ),
                },
                "timeout_sec": {
                    "type": "integer",
                    "description": (
                        "Ceiling for this probe; clamped down to the remaining "
                        "wall-clock budget and to what your session can spare."
                    ),
                },
            },
            "required": ["label", "driver_script", "case_id"],
        },
    },
]


class ProbeServer:
    """Serve the probe tool for one specialist session over stdio."""

    def __init__(self) -> None:
        self._session = ProbeSession()

    async def handle_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the probe tool and wrap its record as MCP content."""
        if name != "probe_variant":
            raise InvalidParamsError(f"unknown tool: {name}")
        try:
            result = await self._session.run(arguments)
        except ProbeRequestError as error:
            raise InvalidParamsError(str(error)) from error
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

    async def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one supported MCP request and return its result object."""
        return await mcp_stdio.dispatch_envelope(
            method,
            params,
            server_name=SERVER_NAME,
            tool_definitions=TOOL_DEFINITIONS,
            handle_tool_call=self.handle_tool_call,
        )


async def _serve() -> None:
    """Serve JSON-RPC requests until stdin closes or an exit notification arrives."""
    await mcp_stdio.serve(ProbeServer().dispatch)


def main() -> None:
    """Run the specialist probe MCP server over standard input and output."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
