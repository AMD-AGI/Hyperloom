# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tool schemas Hyperloom's in-process MCP servers hand to Claude Code."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import IntentType
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mcp_context_tools import CONTEXT_TOOL_SPECS
from hyperloom.orchestrator.roles.mcp_emit_intent import EMIT_INTENT_TOOL_NAME, build_emit_intent_tool_input_schema


@pytest.mark.parametrize(
    "schema",
    [
        *(
            pytest.param(
                build_emit_intent_tool_input_schema(role.allowed_intents), id=f"{EMIT_INTENT_TOOL_NAME}-{name}"
            )
            for name, role in default_role_registry().items()
        ),
        pytest.param(build_emit_intent_tool_input_schema(IntentType), id=f"{EMIT_INTENT_TOOL_NAME}-every-intent"),
        *(pytest.param(schema, id=name) for name, _description, schema, _method in CONTEXT_TOOL_SPECS),
    ],
)
def test_tool_schema_has_no_top_level_combinator(schema: dict[str, Any]) -> None:
    """The Messages API rejects anyOf/oneOf/allOf at the top of an input_schema, and Claude Code then drops the tool from the model's list without an error unless a remote flag flattens it."""
    assert not {"anyOf", "oneOf", "allOf"} & schema.keys()
