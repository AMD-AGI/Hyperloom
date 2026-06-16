# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the editable ``oob-mcp-server`` tree (import name ``agent_mcp_server``).

Requires ``pip install -e OOB/.`` (declared in the tests-coverage workflow). If the
package is missing, the whole module is skipped so local ``pytest`` without OOB
still passes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("agent_mcp_server")


@pytest.fixture(autouse=True)
def _oob_db_reset(monkeypatch, tmp_path):
    """Use an isolated SQLite file and reset cached settings/db path."""
    import agent_mcp_server.config as ac
    import agent_mcp_server.database as adb

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "oob_ut.db"))
    ac.get_settings.cache_clear()
    adb._db_path = None
    yield
    ac.get_settings.cache_clear()
    adb._db_path = None


@pytest.mark.asyncio
async def test_database_init_and_task_roundtrip():
    from agent_mcp_server.database import TaskDB, init_db

    await init_db()
    row = await TaskDB.create(
        task_id="tid-a",
        user_id="user-a",
        input_type="files",
        prompt="optimize",
        config={"k": 1},
        runtime_config={"r": 2},
    )
    assert row["id"] == "tid-a"
    assert row["config"] == {"k": 1}
    assert row["runtime_config"] == {"r": 2}

    got = await TaskDB.get("tid-a")
    assert got is not None and got["user_id"] == "user-a"

    all_u = await TaskDB.list_by_user("user-a")
    assert len(all_u) == 1

    pending_only = await TaskDB.list_by_user("user-a", status="pending")
    assert len(pending_only) == 1

    await TaskDB.update("tid-a", status="running")
    running = await TaskDB.list_running()
    assert any(r["id"] == "tid-a" for r in running)

    same = await TaskDB.update("tid-a")
    assert same is not None

    before = datetime.now(timezone.utc) - timedelta(days=500)
    expired = await TaskDB.list_expired(before)
    assert isinstance(expired, list)


@pytest.mark.asyncio
async def test_taskdb_row_json_invalid_keeps_string(tmp_path, monkeypatch):
    """Invalid JSON in config column is left as string after load."""
    import agent_mcp_server.database as adb
    from agent_mcp_server.database import TaskDB, init_db

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "oob_row.db"))
    adb._db_path = None
    await init_db()
    await TaskDB.create("x1", "u", "t", config=None)
    db_path = adb.get_db_path()
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute("UPDATE tasks SET config = ? WHERE id = ?", ("not-json", "x1"))
        await db.commit()
    row = await TaskDB.get("x1")
    assert row is not None
    assert row["config"] == "not-json"


def test_convergence_parse_and_logic(tmp_path):
    from agent_mcp_server.convergence_check import (
        check_convergence,
        get_best_speedup,
        get_summary,
        parse_speedups,
    )

    missing = tmp_path / "missing.md"
    assert parse_speedups(str(missing)) == []
    assert get_best_speedup(str(missing)) == 1.0
    assert get_summary(str(missing)) == ""

    report = tmp_path / "rep.md"
    report.write_text(
        "## Attempt 1: first\n**Speedup**: 1.1x\n## Attempt 2: second\n**Speedup**: 1.2x\n",
        encoding="utf-8",
    )
    sp = parse_speedups(str(report))
    assert sp == [1.1, 1.2]
    assert not check_convergence(sp, threshold=0.005)
    assert check_convergence([1.0, 2.0, 1.5], threshold=1.0)  # loose threshold
    assert get_best_speedup(str(report)) == 1.2
    summary = get_summary(str(report))
    assert "Round 1" in summary and "1.1x" in summary


def test_settings_defaults_and_env(monkeypatch, tmp_path):
    from agent_mcp_server.config import Settings, get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("HOST", "127.0.0.1")
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.default_max_turns == 50


def test_agent_result_dataclass():
    from agent_mcp_server.backends.base import AgentResult

    r = AgentResult(trajectory=[1], turns=2, cost_usd=0.5, success=True)
    assert r.turns == 2 and r.success is True


def test_backend_names():
    from agent_mcp_server.backends.claude_backend import ClaudeBackend
    from agent_mcp_server.backends.codex_backend import CodexBackend
    from agent_mcp_server.backends.cursor_backend import CursorBackend

    assert ClaudeBackend().name == "claude"
    assert CodexBackend().name == "codex"
    assert CursorBackend().name == "cursor"


def test_get_agent_tools_count():
    from agent_mcp_server.tools.agent_tools import get_agent_tools

    tools = get_agent_tools()
    assert len(tools) >= 5
    names = {t.name for t in tools}
    assert "agent_create_task" in names


@pytest.mark.asyncio
async def test_handle_agent_tool_routes():
    from agent_mcp_server.tools.agent_tools import handle_agent_tool

    tm = MagicMock()
    tm.local_mode = True
    tm.create_task = AsyncMock(return_value={"ok": True})
    out = await handle_agent_tool(
        "agent_create_task",
        tm,
        api_key="k" * 20,
        workspace_id="ws-1",
    )
    assert out == {"ok": True}
    tm.create_task.assert_awaited_once()

    tm2 = MagicMock()
    tm2.local_mode = True
    tm2.get_task = AsyncMock(return_value=None)
    miss = await handle_agent_tool("agent_get_task", tm2, task_id="nope")
    assert "error" in miss

    tm3 = MagicMock()
    tm3.local_mode = False
    sub = await handle_agent_tool("agent_submit_task", tm3, task_id="t")
    assert "API key required" in sub["error"]

    bad = await handle_agent_tool("unknown_tool", tm, workspace_id="x")
    assert "Unknown tool" in bad["error"]


@pytest.mark.asyncio
async def test_handle_agent_tool_keyerror():
    from agent_mcp_server.tools.agent_tools import handle_agent_tool

    tm = MagicMock()
    tm.local_mode = True
    tm.create_task = AsyncMock(side_effect=KeyError("workspace_id"))
    err = await handle_agent_tool("agent_create_task", tm, api_key="x")
    assert "Missing required parameter" in err["error"]


@pytest.mark.asyncio
@patch("agent_mcp_server.safe_client._client")
async def test_safe_client_get_workspaces_list(mock_client_factory):
    from agent_mcp_server.safe_client import SaFEClient

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [{"workspaceId": "w1", "workspaceName": "N1"}]

    inner = AsyncMock()
    inner.get = AsyncMock(return_value=resp)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=inner)
    cm.__aexit__ = AsyncMock(return_value=None)
    mock_client_factory.return_value = cm

    client = SaFEClient("secret", base_url="https://example.test")
    out = await client.get_workspaces()
    assert out[0]["workspaceId"] == "w1"

    wid = await client.get_default_workspace_id()
    assert wid == "w1"

    same = await client.resolve_workspace_id("w1")
    assert same == "w1"


@pytest.mark.asyncio
@patch("agent_mcp_server.safe_client._client")
async def test_safe_client_resolve_missing_workspace(mock_client_factory):
    from agent_mcp_server.safe_client import SaFEClient

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [{"workspaceId": "w1", "workspaceName": "N1"}]

    inner = AsyncMock()
    inner.get = AsyncMock(return_value=resp)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=inner)
    cm.__aexit__ = AsyncMock(return_value=None)
    mock_client_factory.return_value = cm

    client = SaFEClient("k", base_url="https://example.test")
    with pytest.raises(ValueError, match="not found"):
        await client.resolve_workspace_id("missing")


def test_task_manager_static_helpers():
    from agent_mcp_server.task_manager import TaskManager

    assert TaskManager._trajectory_round_number(Path("trajectory_round_7.jsonl")) == 7
    assert TaskManager._trajectory_round_number(Path("bad.jsonl")) == 0

    payload = {"nested": {"total_cost_usd": "2.25"}}
    assert TaskManager._extract_cost_usd(payload) == 2.25
    assert TaskManager._extract_cost_usd({"x": 1}) is None


def test_task_manager_workspace_paths(monkeypatch, tmp_path):
    from agent_mcp_server.config import get_settings
    from agent_mcp_server.task_manager import TaskManager

    monkeypatch.setenv("NFS_BASE_PATH", str(tmp_path))
    get_settings.cache_clear()
    tm = TaskManager()
    p = tm._get_workspace_dir("u1", "t1")
    assert p == tmp_path / "tasks" / "u1" / "t1" / "workspace"


def test_benchmark_tools_definitions():
    from agent_mcp_server.tools.benchmark_tools import get_benchmark_tools

    tools = get_benchmark_tools()
    assert isinstance(tools, list)
    assert any(t.name == "benchmark_submit" for t in tools)


@pytest.mark.asyncio
@patch("agent_mcp_server.safe_client._client")
async def test_safe_client_get_workspaces_dict_shape(mock_client_factory):
    """``get_workspaces`` accepts list or dict payloads from the API."""
    from agent_mcp_server.safe_client import SaFEClient

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"items": [{"workspaceId": "w9"}]}

    inner = AsyncMock()
    inner.get = AsyncMock(return_value=resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=inner)
    cm.__aexit__ = AsyncMock(return_value=None)
    mock_client_factory.return_value = cm

    client = SaFEClient("secret", base_url="https://example.test")
    out = await client.get_workspaces()
    assert out[0]["workspaceId"] == "w9"


def test_safe_client_init_requires_base(monkeypatch):
    import agent_mcp_server.safe_client as sc

    monkeypatch.setattr(sc, "SAFE_API_BASE", "")
    from agent_mcp_server.safe_client import SaFEClient

    with pytest.raises(ValueError, match="SAFE_API_BASE"):
        SaFEClient("k", base_url=None)
