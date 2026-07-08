# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the slimmed PRMonitorClient stub and KnowledgePlane facade."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.knowledge.pr_monitor import (
    DEFAULT_PR_MONITOR_URL,
    PRMonitorClient,
)
from hyperloom.orchestrator.specialists.runner import (
    CORTEX_KB_READONLY_MCP_TOOLS,
    DEFAULT_SPECIALIST_TOOLS,
    PR_MONITOR_MCP_TOOLS,
    SpecialistRunner,
)


# 1. PRMonitorClient stub
def test_pr_monitor_client_from_args_default_url():
    c = PRMonitorClient.from_args()
    assert c.base_url == DEFAULT_PR_MONITOR_URL.rstrip("/")
    assert c.enabled is True


def test_pr_monitor_client_from_args_disabled():
    c = PRMonitorClient.from_args(enabled=False)
    assert c.enabled is False


def test_pr_monitor_client_from_args_env(monkeypatch):
    monkeypatch.delenv("PR_MONITOR_URL", raising=False)
    monkeypatch.setenv("PRIMUS_CORTEX_PR_URL", "http://env-host/v1/")
    c = PRMonitorClient.from_args()
    assert c.base_url == "http://env-host/v1"


def test_pr_monitor_client_timeout_sec_ignored():
    # timeout_sec is accepted for call-site compat but silently ignored
    c = PRMonitorClient.from_args(url="http://x/v1", timeout_sec=2.5)
    assert c.base_url == "http://x/v1"


# 2. KnowledgePlane facade
@pytest.fixture
def plane_with_disabled_pr() -> KnowledgePlane:
    return KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
    )


def test_plane_default_disabled_states(plane_with_disabled_pr):
    plane = plane_with_disabled_pr
    assert plane.pr_monitor_enabled is False
    assert plane.cortex_enabled is False
    assert plane.specialist_mcp_url() == ""


def test_plane_enabled_returns_mcp_url():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=True),
        pr_monitor_mcp_url="http://pr.test/mcp/",
    )
    assert plane.pr_monitor_enabled is True
    assert plane.specialist_mcp_url() == "http://pr.test/mcp/"


def test_plane_reset_round_caches_is_noop():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(),
    )
    plane.reset_round_caches()  # must not raise


def test_plane_cortex_enabled_when_url_set():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
        cortex_kb_mcp_url="http://gbrain.test/mcp",
        cortex_kb_mcp_headers={"Authorization": "Bearer t"},
    )
    assert plane.cortex_enabled is True
    assert plane.cortex_specialist_mcp_url() == "http://gbrain.test/mcp"
    assert plane.cortex_specialist_mcp_headers() == {"Authorization": "Bearer t"}


# 3. SpecialistRunner tool-list gating
def test_default_specialist_tools_include_all_pr_monitor_mcp_tools():
    for t in PR_MONITOR_MCP_TOOLS:
        assert t in DEFAULT_SPECIALIST_TOOLS
    assert len(PR_MONITOR_MCP_TOOLS) == 12


def test_default_specialist_tools_include_cortex_kb_readonly():
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
        assert t in DEFAULT_SPECIALIST_TOOLS


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
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
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


def test_specialist_runner_keeps_cortex_kb_when_mcp_wired():
    plane = KnowledgePlane.from_clients(
        pr_monitor=PRMonitorClient.from_args(enabled=False),
        cortex_kb_mcp_url="http://gbrain.test/mcp",
        cortex_kb_mcp_headers={"Authorization": "Bearer t"},
    )
    runner = SpecialistRunner(
        backend_factory=lambda d: None,
        knowledge_plane=plane,
    )
    tools = runner._resolve_tools()
    for t in CORTEX_KB_READONLY_MCP_TOOLS:
        assert t in tools


def test_specialist_runner_without_plane_keeps_default_tools():
    runner = SpecialistRunner(backend_factory=lambda d: None)
    tools = runner._resolve_tools()
    for t in DEFAULT_SPECIALIST_TOOLS:
        assert t in tools


# 4. MCP config writer
def test_mcp_config_writes_cortex_kb_server_with_headers(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import (
        SPECIALIST_MCP_CONFIG_FILENAME,
        write_specialist_mcp_config,
    )

    path = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="http://pr.test/mcp/",
        cortex_kb_mcp_url="http://gbrain.test/mcp",
        cortex_kb_mcp_headers={"Authorization": "Bearer secret"},
    )
    assert path is not None and path.name == SPECIALIST_MCP_CONFIG_FILENAME
    cfg = json.loads(path.read_text())
    servers = cfg["mcpServers"]
    assert servers["pr_monitor"] == {"type": "http", "url": "http://pr.test/mcp/"}
    assert servers["cortex_kb"]["type"] == "http"
    assert servers["cortex_kb"]["url"] == "http://gbrain.test/mcp"
    assert servers["cortex_kb"]["headers"] == {"Authorization": "Bearer secret"}


def test_mcp_config_omits_cortex_kb_when_url_absent(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config

    path = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="http://pr.test/mcp/",
    )
    assert path is not None
    cfg = json.loads(path.read_text())
    assert "cortex_kb" not in cfg["mcpServers"]
    assert "pr_monitor" in cfg["mcpServers"]


def test_mcp_config_returns_none_when_nothing_wireable(tmp_path):
    from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config

    assert write_specialist_mcp_config(session_dir=tmp_path, pr_monitor_mcp_url="") is None
