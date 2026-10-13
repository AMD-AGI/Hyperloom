# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tool schemas the KernelForge stdio MCP servers hand to agents."""

from __future__ import annotations

from typing import Any

import pytest

from kernelforge.knowledge.pr_knowledge import mcp_server as pr_server
from kernelforge.orchestrator import specialist_probe_server


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(tool["inputSchema"], id=tool["name"])
        for server in (pr_server, specialist_probe_server)
        for tool in server.TOOL_DEFINITIONS
    ],
)
def test_tool_schema_has_no_top_level_combinator(schema: dict[str, Any]) -> None:
    """The Messages API rejects anyOf/oneOf/allOf at the top of an input_schema, and Claude Code then drops the tool from the agent's list without an error unless a remote flag flattens it."""
    assert not {"anyOf", "oneOf", "allOf"} & schema.keys()


def test_pr_check_inclusion_needs_only_the_pr_and_the_tree() -> None:
    """The commit defaults to the tree's HEAD and the merge sha is optional evidence."""
    (schema,) = [tool["inputSchema"] for tool in pr_server.TOOL_DEFINITIONS if tool["name"] == "pr_check_inclusion"]

    assert schema["required"] == ["number", "tree"]
    assert {name: field["type"] for name, field in schema["properties"].items()} == {
        "number": "integer",
        "tree": "string",
        "commit": "string",
        "merge_commit_sha": "string",
    }
