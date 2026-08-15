# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the slimmed PRMonitorClient stub and KnowledgePlane facade."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.knowledge.pr_monitor import PRMonitorClient
from hyperloom.orchestrator.specialists.runner import (
    DEFAULT_SPECIALIST_TOOLS,
    PR_MONITOR_MCP_TOOLS,
    SpecialistRunner,
)


def test_pr_monitor_client_from_args_default_enabled():
    c = PRMonitorClient.from_args()
    assert c.enabled is True


def test_pr_monitor_client_from_args_disabled():
    c = PRMonitorClient.from_args(enabled=False)
    assert c.enabled is False


@pytest.fixture
def plane_with_disabled_pr() -> KnowledgePlane:
    return KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )


def test_plane_default_disabled_states(plane_with_disabled_pr):
    plane = plane_with_disabled_pr
    assert plane.pr_monitor_enabled is False
    assert plane.specialist_mcp_url() == ""


def test_plane_enabled_returns_mcp_url():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=True),
        pr_monitor_mcp_url="http://pr.test/mcp/",
    )
    assert plane.pr_monitor_enabled is True
    assert plane.specialist_mcp_url() == "http://pr.test/mcp/"


def test_default_specialist_tools_include_all_pr_monitor_mcp_tools():
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in DEFAULT_SPECIALIST_TOOLS
    assert len(PR_MONITOR_MCP_TOOLS) == 12


def test_specialist_runner_strips_pr_monitor_when_plane_disabled():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None,
        knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in PR_MONITOR_MCP_TOOLS:
        assert t not in tools


def test_specialist_runner_keeps_pr_monitor_when_plane_enabled():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=True),
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None,
        knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in tools


def test_specialist_runner_explicit_mcp_config_is_authoritative():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=True),
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None,
        knowledge_plane=plane,
        forced_mcp_servers=("pr_monitor",),
    )
    tools = runner._resolve_tools()
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in tools


def test_specialist_runner_without_plane_keeps_default_tools():
    runner = SpecialistRunner(backend_factory=lambda d: None)
    tools = runner._resolve_tools()
    for t in DEFAULT_SPECIALIST_TOOLS:
        assert t in tools


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
