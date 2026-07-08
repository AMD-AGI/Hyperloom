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

_agent_mcp_server = pytest.importorskip("agent_mcp_server")
_repo_oob_root = Path(__file__).resolve().parents[4] / "OOB"
_pkg_file_raw = getattr(_agent_mcp_server, "__file__", "") or ""
_pkg_file = Path(_pkg_file_raw).resolve() if _pkg_file_raw else None
if not _repo_oob_root.exists() or _pkg_file is None or not _pkg_file.is_relative_to(_repo_oob_root.resolve()):
    pytest.skip(
        "editable OOB/ agent_mcp_server tree is not installed from this repo",
        allow_module_level=True,
    )


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


# ── task_manager: pure helpers ────────────────────────────────────


def _make_task_manager(monkeypatch, tmp_path):
    """Construct a TaskManager with an isolated NFS base path."""
    from agent_mcp_server.config import get_settings
    from agent_mcp_server.task_manager import TaskManager

    monkeypatch.setenv("NFS_BASE_PATH", str(tmp_path))
    get_settings.cache_clear()
    return TaskManager()


def test_task_manager_map_status():
    from agent_mcp_server.task_manager import TaskManager

    m = TaskManager._map_status
    assert m("", "succeeded") == "completed"
    assert m("", "completed") == "completed"
    assert m("", "failed") == "failed"
    assert m("", "running") == "running"
    assert m("succeeded", "") == "completed"
    assert m("failed", "") == "failed"
    assert m("pending", "") == "running"
    assert m("unknown", "unknown") is None


def test_task_manager_default_prompt_and_runtime_config(monkeypatch, tmp_path):
    tm = _make_task_manager(monkeypatch, tmp_path)

    assert "Optimize" in tm._get_default_prompt()

    cfg = tm._get_runtime_config({"cpu": 16, "gpu_count": None})
    assert cfg["cpu"] == 16  # override applied
    assert "image" in cfg  # default preserved
    # None override does not clobber the default.
    assert cfg["gpu_count"] == tm.settings.default_gpu_count


def test_task_manager_build_env_vars(monkeypatch, tmp_path):
    tm = _make_task_manager(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAW_SESSION_ID", "sess-1")
    env = tm._build_env_vars("task-9")
    assert env["TASK_ID"] == "task-9"
    assert env["TRACE_COMPONENT"] == "oob"
    assert env["CLAW_SESSION_ID"] == "sess-1"


def test_task_manager_default_system_prompt(monkeypatch, tmp_path):
    tm = _make_task_manager(monkeypatch, tmp_path)
    # The packaged template exists, so a non-empty prompt comes back.
    assert tm._get_default_system_prompt().strip()


@pytest.mark.parametrize("agent", ["codex", "cursor", "claude"])
def test_task_manager_build_local_cmd(monkeypatch, tmp_path, agent):
    tm = _make_task_manager(monkeypatch, tmp_path)
    cmd = tm._build_local_cmd(
        agent=agent,
        prompt="do it",
        model="m-1",
        max_turns=5,
        system_prompt="be careful",
        workspace_dir=tmp_path / "ws",
        task_dir=tmp_path / "task",
    )
    assert isinstance(cmd, list) and cmd
    if agent == "codex":
        assert cmd[0] == "codex"
    elif agent == "cursor":
        assert any("cursor_backend" in c for c in cmd)
    else:
        assert cmd[0] == "claude"
        assert "--system-prompt" in cmd


@pytest.mark.parametrize(
    "agent, key_env",
    [("codex", "OPENAI_API_KEY"), ("cursor", "CURSOR_API_KEY"), ("claude", "ANTHROPIC_API_KEY")],
)
def test_task_manager_build_local_env(monkeypatch, tmp_path, agent, key_env):
    tm = _make_task_manager(monkeypatch, tmp_path)
    env = tm._build_local_env(agent, "secret-key", "task-3", system_prompt="sp")
    assert env[key_env] == "secret-key"
    assert env["OOB_SYSTEM_PROMPT"] == "sp"
    assert env["TASK_ID"] == "task-3"


def test_task_manager_parse_llm_proxy(monkeypatch, tmp_path):
    from agent_mcp_server.config import get_settings
    from agent_mcp_server.task_manager import TaskManager

    monkeypatch.setenv("NFS_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("LLM_PROXY_URL", "https://proxy.test:8443/api/v1")
    get_settings.cache_clear()
    tm = TaskManager()
    proxy = tm._parse_llm_proxy()
    assert proxy["scheme"] == "https"
    assert proxy["host"] == "proxy.test"
    assert proxy["port"] == "8443"
    assert proxy["path"] == "/api/v1"
    # trailing /v1 stripped for the anthropic base.
    assert proxy["anthropic_path"] == "/api"


@pytest.mark.parametrize("agent", ["claude", "codex", "cursor"])
def test_task_manager_build_execution_command(monkeypatch, tmp_path, agent):
    tm = _make_task_manager(monkeypatch, tmp_path)
    script = tm._build_execution_command(
        task_id="t-1",
        task_dir=tmp_path / "task",
        workspace_dir=tmp_path / "ws",
        config={"agent": agent, "model": "m-x"},
        api_key="key-1",
    )
    assert isinstance(script, str) and "Multi-round optimization loop" in script
    if agent == "codex":
        assert "codex exec" in script
    elif agent == "claude":
        assert "claude" in script and "auth_proxy.py" in script
    else:
        assert "@cursor/sdk" in script


def test_task_manager_summarize_usage_none(monkeypatch, tmp_path):
    tm = _make_task_manager(monkeypatch, tmp_path)
    # No workspace dir at all.
    assert tm._summarize_task_usage("u", "t") is None
    # Workspace exists but no trajectory files.
    ws = tm._get_workspace_dir("u", "t")
    ws.mkdir(parents=True)
    assert tm._summarize_task_usage("u", "t") is None


def test_task_manager_summarize_usage_claude_and_codex(monkeypatch, tmp_path):
    import json as _json

    tm = _make_task_manager(monkeypatch, tmp_path)
    ws = tm._get_workspace_dir("u", "t")
    ws.mkdir(parents=True)

    # Round 1: Claude-format usage inside message.
    r1 = ws / "trajectory_round_1.jsonl"
    r1.write_text(
        _json.dumps(
            {
                "message": {
                    "model": "claude-x",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 2,
                        "cache_read_input_tokens": 1,
                    },
                },
                "total_cost_usd": 0.5,
            }
        )
        + "\n"
        + "not-json\n",
        encoding="utf-8",
    )
    # Round 2: Codex-format usage on turn.completed.
    r2 = ws / "trajectory_round_2.jsonl"
    r2.write_text(
        _json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 20, "output_tokens": 4},
                "cost_usd": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = tm._summarize_task_usage("u", "t")
    assert summary is not None
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 9
    assert summary["events_with_usage"] == 2
    assert summary["cost_available"] is True
    assert summary["cost_usd"] == 1.5
    assert "claude-x" in summary["models"]
    assert len(summary["rounds"]) == 2


# ── benchmark_tools: handlers via mocked SaFEClient ───────────────


@pytest.mark.asyncio
async def test_benchmark_handle_requires_key_and_base(monkeypatch):
    import agent_mcp_server.tools.benchmark_tools as bt

    out = await bt.handle_benchmark_tool("benchmark_status", api_key=None)
    assert "API key required" in out["error"]

    monkeypatch.setattr(bt, "SAFE_API_BASE", "")
    out = await bt.handle_benchmark_tool("benchmark_status", api_key="k")
    assert "SAFE_API_BASE" in out["error"]

    monkeypatch.setattr(bt, "SAFE_API_BASE", "https://safe.test")
    out = await bt.handle_benchmark_tool("nope", api_key="k")
    assert "Unknown benchmark tool" in out["error"]


@pytest.mark.asyncio
async def test_benchmark_submit_status_result_cancel(monkeypatch, tmp_path):
    import agent_mcp_server.tools.benchmark_tools as bt

    monkeypatch.setattr(bt, "SAFE_API_BASE", "https://safe.test")
    monkeypatch.setattr(bt, "SAFE_NFS_PATH", str(tmp_path / "bench"))

    client = MagicMock()
    client.get_default_workspace_id = AsyncMock(return_value="ws-1")
    client.resolve_workspace_id = AsyncMock(return_value="ws-2")
    client.create_workload = AsyncMock(return_value={"id": "wl-1"})
    client.get_workload = AsyncMock(return_value={"status": "running", "phase": "running"})
    client.stop_workload = AsyncMock(return_value={})
    monkeypatch.setattr(bt, "SaFEClient", lambda *a, **k: client)

    submit = await bt.handle_benchmark_tool(
        "benchmark_submit",
        api_key="k",
        files=[{"filename": "a.py", "content": "print(1)"}],
        command="python a.py",
    )
    bid = submit["benchmark_id"]
    assert submit["workload_id"] == "wl-1"
    assert submit["status"] == "submitted"

    status = await bt.handle_benchmark_tool("benchmark_status", api_key="k", benchmark_id=bid)
    assert status["status"] == "running"

    # benchmark.log was not written by the (mocked) workload → file-not-found
    # branch returns available_files list.
    result = await bt.handle_benchmark_tool("benchmark_result", api_key="k", benchmark_id=bid)
    assert "available_files" in result or "content" in result

    cancel = await bt.handle_benchmark_tool("benchmark_cancel", api_key="k", benchmark_id=bid)
    assert cancel["status"] == "cancelled"

    missing = await bt.handle_benchmark_tool("benchmark_status", api_key="k", benchmark_id="nope")
    assert "not found" in missing["error"]


# ── task_manager: async lifecycle (db + filesystem, mocked SaFE) ───


@pytest.mark.asyncio
async def test_task_manager_create_get_outputs_lifecycle(monkeypatch, tmp_path):
    from agent_mcp_server.database import init_db

    tm = _make_task_manager(monkeypatch, tmp_path)
    await init_db()

    with pytest.raises(ValueError, match="Unsupported agent"):
        await tm.create_task(user_id="u", agent="bogus", workspace_id="ws")

    with pytest.raises(ValueError, match="workspace_id is required"):
        await tm.create_task(user_id="u", agent="claude", workspace_id=None)

    task = await tm.create_task(
        user_id="u",
        agent="claude",
        workspace_id="ws-x",
        files=[{"filename": "k.py", "content": "print(1)"}],
        prompt="optimize",
    )
    tid = task["id"]
    assert task["status"] == "pending"

    got = await tm.get_task(tid)
    assert got is not None and got["user_id"] == "u"

    outs = await tm.get_outputs(tid)
    assert any(f["path"] == "k.py" for f in outs["files"])

    content = await tm.get_file_content(tid, "k.py")
    assert content["content"] == "print(1)"

    missing = await tm.get_file_content(tid, "nope.py")
    assert "File not found" in missing["error"]

    traversal = await tm.get_file_content(tid, "../../etc/passwd")
    assert traversal["error"] == "Path traversal not allowed"

    cancelled = await tm.cancel_task(tid, api_key="k")
    assert cancelled["status"] == "cancelled"

    assert await tm.get_task("nope") is None
    with pytest.raises(ValueError, match="Task not found"):
        await tm.get_outputs("nope")


@pytest.mark.asyncio
async def test_task_manager_submit_remote(monkeypatch, tmp_path):
    import agent_mcp_server.task_manager as tmmod
    from agent_mcp_server.database import init_db

    tm = _make_task_manager(monkeypatch, tmp_path)
    await init_db()
    task = await tm.create_task(user_id="u", agent="claude", workspace_id="ws-x")
    tid = task["id"]

    client = MagicMock()
    client.create_workload = AsyncMock(return_value={"id": "wl-9"})
    monkeypatch.setattr(tmmod, "SaFEClient", lambda *a, **k: client)

    out = await tm._submit_remote(tid, api_key="k")
    assert out["status"] == "running"
    assert out["safe_workload_id"] == "wl-9"
