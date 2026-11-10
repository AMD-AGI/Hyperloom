# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the PR Monitor tool surface and the specialist MCP config."""

from __future__ import annotations

import json

from hyperloom.orchestrator.policy.gate import PR_MONITOR_TOOL_NAMES as _PR_MONITOR_TOOL_NAMES


def test_pr_monitor_tool_names_count():
    assert len(_PR_MONITOR_TOOL_NAMES) == 12


def test_mcp_config_writes_pr_monitor_only(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import (
        SPECIALIST_MCP_CONFIG_FILENAME,
        write_specialist_mcp_config,
    )

    path = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="http://pr.test/mcp/",
    )
    assert path is not None and path.name == SPECIALIST_MCP_CONFIG_FILENAME
    cfg = json.loads(path.read_text())
    servers = cfg["mcpServers"]
    assert servers["pr_monitor"] == {"type": "http", "url": "http://pr.test/mcp/"}
    assert "recipe_kb" not in servers


def test_mcp_config_returns_none_when_nothing_wireable(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config

    assert write_specialist_mcp_config(session_dir=tmp_path, pr_monitor_mcp_url="") is None
