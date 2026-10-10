# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tool schemas Hyperloom's in-process MCP servers hand to Claude Code."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.roles.mcp_context_tools import CONTEXT_TOOL_SPECS
from hyperloom.orchestrator.roles.mcp_emit_intent import EMIT_INTENT_TOOL_INPUT_SCHEMA, EMIT_INTENT_TOOL_NAME


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(EMIT_INTENT_TOOL_INPUT_SCHEMA, id=EMIT_INTENT_TOOL_NAME),
        *(pytest.param(schema, id=name) for name, _description, schema, _method in CONTEXT_TOOL_SPECS),
    ],
)
def test_tool_schema_has_no_top_level_combinator(schema: dict[str, Any]) -> None:
    """The Messages API rejects anyOf/oneOf/allOf at the top of an input_schema, and Claude Code then drops the tool from the model's list without an error unless a remote flag flattens it."""
    assert not {"anyOf", "oneOf", "allOf"} & schema.keys()
