# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tool schemas the KernelForge stdio MCP servers hand to agents."""

from __future__ import annotations

from typing import Any

import pytest

from kernelforge.mcp_server import pr_stdio_server, probe_stdio_server


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(tool["inputSchema"], id=tool["name"])
        for server in (pr_stdio_server, probe_stdio_server)
        for tool in server.TOOL_DEFINITIONS
    ],
)
def test_tool_schema_has_no_top_level_combinator(schema: dict[str, Any]) -> None:
    """The Messages API rejects anyOf/oneOf/allOf at the top of an input_schema, and Claude Code then drops the tool from the agent's list without an error unless a remote flag flattens it."""
    assert not {"anyOf", "oneOf", "allOf"} & schema.keys()
