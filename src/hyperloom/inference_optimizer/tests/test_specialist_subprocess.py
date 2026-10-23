# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SpecialistRunner subprocess + worktree tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.deadline import Deadline
from hyperloom.common.git_safety import worktree_head
from hyperloom.common.env_safety import specialist_run_as

from .conftest import init_git_repo, make_fake_claude

from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES

from hyperloom.orchestrator.actions.executors._ray_serving import RayInfeasibleError

from hyperloom.orchestrator.specialists.runner import (
    SPECIALIST_TOOL_DENYLIST,
    SpecialistRunner,
)
from hyperloom.orchestrator.specialists import subprocess_, wall_budget
from hyperloom.orchestrator.specialists import launch_env
from hyperloom.orchestrator.specialists.launch_env import prepare_demote
from hyperloom.orchestrator.specialists.agent_cli_config import _build_specialist_env
from hyperloom.orchestrator.specialists.leaf import build_leaf_agents_json
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _setup_worktree,
    ReapWatch,
    SpecialistSpawn,
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


def test_build_specialist_env_inherits_provider_secrets_by_default(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-api-value")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: anthropic-api-value")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access-key-value")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token-value")
    monkeypatch.setenv("KB_SERVICE_TOKEN", "kb-token-value")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "/tmp/session")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _build_specialist_env()
    assert env["PATH"] == "/usr/bin"
    assert env["ANTHROPIC_API_KEY"] == "anthropic-api-value"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "Ocp-Apim-Subscription-Key: anthropic-api-value"
    assert env["AWS_ACCESS_KEY_ID"] == "aws-access-key-value"
    assert "GITHUB_TOKEN" not in env
    assert "KB_SERVICE_TOKEN" not in env
    assert "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR" not in env
    assert "LD_PRELOAD" not in env


def test_build_specialist_env_forwards_hub_settings_but_not_the_hub_token(monkeypatch):
    hub = {
        "HF_ENDPOINT": "https://hf-mirror.example",
        "HF_HOME": "/shared/hf",
        "HF_HUB_CACHE": "/shared/hf/hub",
        "HF_HUB_OFFLINE": "1",
    }
    for name, value in hub.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HF_TOKEN", "hf-token-value")
    env = _build_specialist_env()
    assert {name: env.get(name) for name in hub} == hub
    assert "HF_TOKEN" not in env


def test_build_specialist_env_forwards_the_claude_config_dir(monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/claude-config")
    assert _build_specialist_env()["CLAUDE_CONFIG_DIR"] == "/tmp/claude-config"


def test_build_specialist_env_forwards_oauth_token_without_mirroring_it(monkeypatch):
    """A subscription-only parent must hand the token down untouched."""
    oauth_env = "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN"))
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv(oauth_env, "sk-ant-oat01-fake")
    env = _build_specialist_env()
    assert env[oauth_env] == "sk-ant-oat01-fake"
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_build_specialist_env_secret_inheritance_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-api-value")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: anthropic-api-value")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access-key-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token-value")
    env = _build_specialist_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "AWS_REGION" in env
    assert "GITHUB_TOKEN" not in env


@pytest.fixture
def fake_framework_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The checkout the session optimises, named the way a session names it."""
    repo = tmp_path / "framework"
    init_git_repo(repo)
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))
    return repo


def _make_runner_ctx(task_id: str = "t-spec-1") -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="specialist",
        state="queued",
        params={
            "domain": "serving_specialist",
            "framework": "sglang",
            "gap_canonical_id": "gap.test.example",
            "max_turns": 2,
        },
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={})


def test_runner_accepts_subprocess_config_only():
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(),
    )
    assert runner.subprocess_dispatcher is not None


def test_denylist_blocks_dangerous_process_tools():
    """KillShell and SlashCommand are in the denylist to enforce the prompt-rule against global process cleanup that could kill the serving / benchmark process."""
    assert "KillShell" in SPECIALIST_TOOL_DENYLIST
    assert "SlashCommand" in SPECIALIST_TOOL_DENYLIST


def test_kb_mcp_tools_not_in_denylist():
    """KB MCP server names must not appear in the denylist."""
    denylisted_kb_mcp = [t for t in SPECIALIST_TOOL_DENYLIST if t.startswith("mcp__") and "kb" in t.lower()]
    assert denylisted_kb_mcp == [], f"stale KB MCP entries in the denylist: {denylisted_kb_mcp}"


@pytest.mark.asyncio
async def test_worktree_of_a_pip_installed_framework_is_its_snapshot_not_another_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """With no checkout of its own, the framework still hands its specialist its own code -- never InferenceX's."""
    harness = tmp_path / "InferenceX"
    init_git_repo(harness, seed_file="benchmark_lib.sh", seed_text="run\n")
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "envs.py").write_text("VLLM_USE_X = 0\n", encoding="utf-8")
    monkeypatch.setenv("INFERENCEX_PATH", str(harness))
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(framework_source_roots=(str(harness), str(package))),
        session_dir=session_dir,
    )
    ctx = _make_runner_ctx("t-spec-pip")
    ctx.task.params["session_framework_tree"] = f"{package}/"
    workspace = session_dir / "runs" / "specialist" / "t-spec-pip"
    workspace.mkdir(parents=True)

    worktree, source, err = runner._maybe_setup_worktree(ctx, workspace=workspace)

    assert err == ""
    assert source is not None and source.root == package and not source.checkout
    assert worktree is not None and (worktree / "envs.py").read_text(encoding="utf-8") == "VLLM_USE_X = 0\n"
    assert not (worktree / "benchmark_lib.sh").exists()
    assert not (package / ".git").exists()
    listed = subprocess.run(
        ["git", "-C", str(harness), "worktree", "list"], capture_output=True, text=True, check=True
    ).stdout
    assert str(worktree) not in listed


def test_an_integrate_in_flight_holds_the_snapshot(tmp_path: Path):
    from hyperloom.orchestrator.state.shared_state import SharedState

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig(), session_dir=session_dir)
    assert runner._integrate_in_flight() is False

    state = SharedState.load_or_init(session_dir)
    state.pending_integrate = {"task_id": "t-integrate", "patches": []}
    state.save(session_dir)
    assert runner._integrate_in_flight() is True


def test_setup_worktree_creates_branch_off_base(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    workspace = tmp_path / "workspace"
    worktree, err = _setup_worktree(
        fake_framework_repo,
        workspace / "worktree",
        "specialist-test1",
    )
    assert err == "", err
    assert worktree is not None
    assert worktree.is_dir()
    cp = subprocess.run(
        ["git", "-C", str(fake_framework_repo), "branch", "--list", "specialist-test1"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "specialist-test1" in cp.stdout


@pytest.mark.asyncio
async def test_subprocess_path_harvests_done_file(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """The fake ``claude`` writes specialist_done.json; the runner reads it and returns status=succeeded."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="done_only")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-done")

    result = await runner.run(ctx)

    assert result.status == "succeeded"
    assert result.specialist_done["proposal_set"]
    assert result.specialist_done["domain"] == "serving_specialist"
    workspace = session_dir / "runs" / "specialist" / "t-spec-done"
    assert (workspace / "specialist_done.json").exists()
    assert (workspace / "process.log").exists()
    assert (workspace / "worktree").is_dir()


@pytest.mark.asyncio
async def test_subprocess_run_is_one_specialist_llm_call_on_the_trajectory(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """The subprocess books as a specialist llm.call carrying the result-row totals; its tools hang off that call."""
    from hyperloom.inference_optimizer.session.session_paths import llm_calls_path
    from hyperloom.inference_optimizer.trace import trajectory_trace as tt

    fake_claude = make_fake_claude(tmp_path / "bin", behavior="done_with_stream_json")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(
            claude_executable=str(fake_claude),
            model="",
            framework_source_roots=(str(fake_framework_repo),),
            poll_interval_seconds=0.2,
        ),
        session_dir=session_dir,
        default_max_turns=2,
    )
    with tt.trajectory_scope(
        session_dir=session_dir,
        component="coordinator",
        task_id="t-spec-traj",
        parent_span_id="t-spec-traj",
    ):
        result = await runner.run(_make_runner_ctx("t-spec-traj"))
    assert result.status == "succeeded"

    events = tt.load_events(session_dir)
    calls = [e for e in events if e["event_type"] == tt.EVENT_LLM_CALL]
    assert [e["status"] for e in calls] == [tt.STATUS_STARTED, tt.STATUS_COMPLETED]
    call = calls[-1]
    assert (call["component"], call["agent"], call["task_id"]) == ("specialist", "serving_specialist", "t-spec-traj")
    assert call["parent_span_id"] == "t-spec-traj"
    assert call["attributes"]["input_tokens"] == 50632
    assert call["attributes"]["cache_read_input_tokens"] == 291392
    assert call["attributes"]["output_tokens"] == 7542
    assert call["attributes"]["model"] == "glm-5-3"

    tools = [e for e in events if e["event_type"] == tt.EVENT_TOOL]
    assert [t["attributes"]["name"] for t in tools] == ["Bash"]
    assert (tools[0]["component"], tools[0]["agent"]) == ("specialist", "serving_specialist")
    assert tools[0]["parent_span_id"] == call["span_id"]
    assert tools[0]["call_id"] == call["call_id"]

    rows = [json.loads(line) for line in llm_calls_path(session_dir).read_text(encoding="utf-8").splitlines() if line]
    specialist_rows = [r for r in rows if r["component"] == "specialist"]
    assert len(specialist_rows) == 1
    assert specialist_rows[0]["call_id"] == call["call_id"]
    assert specialist_rows[0]["input_tokens"] == 50632
    assert specialist_rows[0]["output_tokens"] == 7542


@pytest.mark.asyncio
async def test_local_specialist_spawn_uses_file_stdin(
    tmp_path: Path,
    fake_framework_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The local specialist path feeds the user prompt via stdin."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="done_only")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    seen_stdin: list[Any] = []
    real_popen = subprocess.Popen

    def _recording_popen(cmd, *args, **kwargs):
        if cmd and str(cmd[0]) == str(fake_claude):
            seen_stdin.append(kwargs.get("stdin"))
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess_.subprocess, "Popen", _recording_popen)
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(
            claude_executable=str(fake_claude),
            model="",
            framework_source_roots=(str(fake_framework_repo),),
            poll_interval_seconds=0.2,
        ),
        session_dir=session_dir,
        default_max_turns=2,
    )

    result = await runner.run(_make_runner_ctx("t-spec-stdin"))

    assert result.status == "succeeded"
    assert len(seen_stdin) == 1
    assert seen_stdin[0] is not subprocess.DEVNULL


@pytest.mark.asyncio
async def test_subprocess_path_injects_allocated_gpu_env(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="done_with_env")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-gpu")
    ctx.extra["gpu_ids"] = [2, 3]

    result = await runner.run(ctx)

    assert result.status == "succeeded"
    assert result.specialist_done["hip_visible"] == "2,3"
    assert result.specialist_done["cuda_visible"] == "2,3"
    assert result.specialist_done["rocr_visible"] == "2,3"
    assert result.specialist_done["allocated_gpu_ids"] == [2, 3]


@pytest.mark.asyncio
async def test_subprocess_path_injects_llm_stability_env(
    tmp_path: Path,
    fake_framework_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The dispatcher injects low-risk claude-code stability flags but does not set API_TIMEOUT_MS by default; liveness is governed by the process.log / heartbeat stale reaper."""
    # Ensure no inherited values mask the setdefault under test.
    for var in (
        "API_TIMEOUT_MS",
        "DISABLE_AUTOUPDATER",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    ):
        monkeypatch.delenv(var, raising=False)

    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="done_with_llm_env")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-llmenv")

    result = await runner.run(ctx)

    assert result.status in ("succeeded", "empty_synthesised")
    assert result.specialist_done["api_timeout_ms"] == ""
    assert result.specialist_done["disable_autoupdater"] == "1"
    assert result.specialist_done["disable_nonessential"] == "1"


@pytest.mark.asyncio
async def test_readonly_research_scout_skips_worktree(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="done_only")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-scout")
    ctx.task.params.update(
        {
            "domain": "research_scout_specialist",
            "gap_canonical_id": "gap.research_scout.round0",
            "mode": "research",
        }
    )

    result = await runner.run(ctx)

    assert result.status == "succeeded"
    workspace = session_dir / "runs" / "specialist" / "t-spec-scout"
    assert (workspace / "specialist_done.json").exists()
    assert not (workspace / "worktree").exists()


@pytest.mark.asyncio
async def test_subprocess_path_collects_patches(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """A done file + worktree patch threads the patch path into specialist_done['patches_written']."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="done_with_patch")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-patch")

    result = await runner.run(ctx)

    assert result.status == "succeeded"
    patches = result.specialist_done["patches_written"]
    assert isinstance(patches, list) and len(patches) == 1
    assert patches[0].endswith("001_test.patch")
    worktree = session_dir / "runs" / "specialist" / "t-spec-patch" / "worktree"
    assert (worktree / "patches" / "001_test.patch").exists()


@pytest.mark.asyncio
async def test_subprocess_crash_falls_back_to_empty_synthesised(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """A crash with no done.json synthesises an empty specialist_done and a stale-like status."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="crash")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-crash")

    result = await runner.run(ctx)
    assert result.status in ("empty_synthesised", "stale")
    assert result.specialist_done["proposal_set"] == []
    assert "subprocess" in (result.error or "")


@pytest.mark.asyncio
async def test_subprocess_path_isolates_writes_to_worktree(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """Worktree patches must NOT appear in the base repo's working tree until ``integrate_patch`` applies them."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="done_with_patch")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-iso")

    result = await runner.run(ctx)
    assert result.status == "succeeded"

    worktree = session_dir / "runs" / "specialist" / "t-spec-iso" / "worktree"
    assert (worktree / "patches" / "001_test.patch").exists()
    assert not (fake_framework_repo / "patches" / "001_test.patch").exists()
    assert not (fake_framework_repo / "dummy.txt").exists()


@pytest.mark.asyncio
async def test_subprocess_recovers_partial_when_no_final(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """A specialist that wrote only the partial (then died before the final done.json) surfaces the partial as a non-empty result."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="partial_then_crash")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-partial")

    result = await runner.run(ctx)
    # Salvaged work keeps the findings but must not read as a clean run.
    assert result.status == "partial"
    assert "recovered_from_partial" in result.notes
    assert result.specialist_done["proposal_set"]
    assert result.specialist_done.get("_recovered_from_partial") is True
    assert result.specialist_done["proposal_set"]


@pytest.mark.asyncio
async def test_the_dispatch_deadline_kills_a_hung_specialist(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """A small Coordinator-injected ``wall_budget_sec`` must kill a hung specialist well before the legacy ``max_turns × per_turn`` ceiling (here 2 × 15 = 30s)."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="hang")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-budget")
    ctx.extra["specialist_deadline"] = Deadline.after(1.0)

    started = time.monotonic()
    result = await runner.run(ctx)
    elapsed = time.monotonic() - started

    assert elapsed < 15.0
    assert result.status in ("stale", "empty_synthesised")
    assert "timeout" in (result.error or "")


class _FakeProc:
    """Minimal stand-in for ``subprocess.Popen`` for reaper unit tests."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.returncode: int | None = None
        self.alive = True

    def poll(self) -> int | None:
        if self.alive:
            return None
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_reap_loop_process_log_activity_prevents_stale_kill(
    tmp_path: Path,
):
    """A specialist that streams to process.log but never self-writes heartbeat.json must NOT be reaped as stale."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    process_log = workspace / "process.log"
    process_log.write_text("start\n", encoding="utf-8")

    cfg = SpecialistSubprocessConfig(
        heartbeat_stale_seconds=1.0,
        poll_interval_seconds=0.2,
    )
    disp = SpecialistSubprocessDispatcher(config=cfg)
    proc = _FakeProc()

    async def _keep_streaming() -> None:
        # Touch process.log past the stale threshold, then exit cleanly.
        for i in range(15):  # ~3s, 3x the stale threshold
            process_log.write_text(f"line {i}\n", encoding="utf-8")
            await asyncio.sleep(0.2)
        proc.alive = False

    writer = asyncio.create_task(_keep_streaming())
    outcome = await disp._reap_loop(
        proc,
        ReapWatch(workspace=workspace, done_files=()),
        deadline=Deadline.after(60.0),
        started=time.monotonic(),
    )
    _ = await writer

    assert outcome["stale_heartbeat"] is False, outcome
    assert outcome["timed_out"] is False, outcome
    assert outcome["exit_code"] == 0


@pytest.mark.asyncio
async def test_reap_loop_kills_when_no_activity_at_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """With neither heartbeat.json nor process.log activity, the reaper still reaps a silent/hung subprocess as stale."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # No process.log, no heartbeat.json — total silence.

    cfg = SpecialistSubprocessConfig(
        heartbeat_stale_seconds=0.5,
        poll_interval_seconds=0.2,
    )
    disp = SpecialistSubprocessDispatcher(config=cfg)
    proc = _FakeProc()  # stays alive; only staleness can stop it

    # Stub _kill so the reaper never signals a real process group.
    killed = {"v": False}

    def _fake_kill(p: Any) -> None:
        killed["v"] = True
        p.alive = False

    monkeypatch.setattr(
        SpecialistSubprocessDispatcher,
        "_kill",
        staticmethod(_fake_kill),
    )

    outcome = await disp._reap_loop(
        proc,
        ReapWatch(workspace=workspace, done_files=()),
        deadline=Deadline.after(60.0),
        started=time.monotonic(),
    )
    assert outcome["stale_heartbeat"] is True, outcome
    assert killed["v"] is True


def _silent_log(workspace: Path, *, age_sec: float) -> Path:
    """A ``process.log`` whose last write was ``age_sec`` ago."""
    workspace.mkdir()
    log = workspace / "process.log"
    log.write_text("event\n", encoding="utf-8")
    then = time.time() - age_sec
    os.utime(log, (then, then))
    return log


def _cli(code: str) -> subprocess.Popen:
    """An agent CLI stand-in in its own session, as the dispatcher launches one."""
    return subprocess.Popen([sys.executable, "-c", code], start_new_session=True)


def _end(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)


async def _reap(proc: subprocess.Popen, workspace: Path, *, deadline_sec: float) -> dict[str, Any]:
    disp = SpecialistSubprocessDispatcher(
        config=SpecialistSubprocessConfig(heartbeat_stale_seconds=1.0, poll_interval_seconds=0.1)
    )
    return await disp._reap_loop(
        proc,
        ReapWatch(workspace=workspace, done_files=()),
        deadline=Deadline.after(deadline_sec),
        started=time.monotonic(),
    )


@pytest.mark.asyncio
async def test_reap_loop_keeps_a_cli_waiting_on_a_tool_it_started_after_its_last_output(tmp_path: Path):
    """A long tool call leaves process.log silent; the tool the CLI started after that output keeps it alive."""
    workspace = tmp_path / "ws"
    _silent_log(workspace, age_sec=60.0)
    proc = _cli("import subprocess; subprocess.run(['sleep', '30'])")
    try:
        outcome = await _reap(proc, workspace, deadline_sec=3.0)
    finally:
        _end(proc)
    assert (outcome["stale_heartbeat"], outcome["timed_out"]) == (False, True)


@pytest.mark.asyncio
async def test_reap_loop_reaps_a_silent_cli_whose_children_predate_its_last_output(tmp_path: Path):
    """A child older than the CLI's last output is not a tool call in flight, so silence still reaps the CLI."""
    from hyperloom.common.proctree import descendants

    workspace = tmp_path / "ws"
    proc = _cli("import subprocess; subprocess.run(['sleep', '30'])")
    try:
        deadline = time.monotonic() + 5.0
        while not descendants(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(1.0)
        _silent_log(workspace, age_sec=0.0)
        outcome = await _reap(proc, workspace, deadline_sec=10.0)
    finally:
        _end(proc)
    assert (outcome["stale_heartbeat"], outcome["timed_out"]) == (True, False)


@pytest.mark.asyncio
async def test_reap_loop_reaps_a_hung_cli_while_something_refreshes_heartbeat_json(tmp_path: Path):
    """heartbeat.json is writable by the agent, so a loop refreshing it does not keep a hung CLI alive."""
    workspace = tmp_path / "ws"
    _silent_log(workspace, age_sec=60.0)
    heartbeat = workspace / "heartbeat.json"
    proc = _cli("import time; time.sleep(30)")

    async def _refresh() -> None:
        while True:
            heartbeat.write_text('{"status": "running"}\n', encoding="utf-8")
            await asyncio.sleep(0.1)

    writer = asyncio.create_task(_refresh())
    try:
        outcome = await _reap(proc, workspace, deadline_sec=3.0)
    finally:
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
        _end(proc)
    assert (outcome["stale_heartbeat"], outcome["timed_out"]) == (True, False)


# ── extend_lease moves the live wall-clock deadline ──────────────────────────
@pytest.fixture
def _live_reaper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A reaper whose only stop condition is the hard wall-clock cap."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "process.log").write_text("alive\n", encoding="utf-8")

    cfg = SpecialistSubprocessConfig(
        # Far above the run so only the wall-clock cap can end the loop.
        heartbeat_stale_seconds=3600.0,
        poll_interval_seconds=0.05,
    )
    disp = SpecialistSubprocessDispatcher(config=cfg)
    proc = _FakeProc()
    monkeypatch.setattr(
        SpecialistSubprocessDispatcher,
        "_kill",
        staticmethod(lambda p: setattr(p, "alive", False)),
    )
    return disp, proc, workspace


@pytest.mark.asyncio
async def test_reap_loop_times_out_at_base_budget_without_extension(_live_reaper):
    """Baseline: with no extension the run dies at its original cap."""
    disp, proc, workspace = _live_reaper
    wall_budget.clear_wall_budget_extension("task-base")

    started = time.monotonic()
    outcome = await disp._reap_loop(
        proc,
        ReapWatch(workspace=workspace, done_files=(), task_id="task-base"),
        deadline=Deadline.after(0.3),
        started=time.monotonic(),
    )
    elapsed = time.monotonic() - started

    assert outcome["timed_out"] is True, outcome
    # Killed at ~0.3s.
    assert elapsed < 5.0, elapsed


@pytest.mark.asyncio
async def test_reap_loop_deadline_moves_when_extension_granted_mid_run(_live_reaper):
    """The regression this fix exists for."""
    disp, proc, workspace = _live_reaper
    wall_budget.clear_wall_budget_extension("task-live")

    started = time.monotonic()
    loop = asyncio.create_task(
        disp._reap_loop(
            proc,
            ReapWatch(workspace=workspace, done_files=(), task_id="task-live"),
            deadline=Deadline.after(0.3),
            started=time.monotonic(),
        )
    )
    # Grant the extension while the run is still in flight, before the original 0.3s cap would have fired.
    await asyncio.sleep(0.15)
    wall_budget.grant_wall_budget_extension("task-live", 0.6)
    # The reaper recomputes `max_seconds + wall_budget_extension(task_id)` every poll, so this is the deadline it now
    # enforces.
    assert wall_budget.wall_budget_extension("task-live") == 0.6
    outcome = await loop
    elapsed = time.monotonic() - started

    assert outcome["timed_out"] is True, outcome
    # Survived past the base cap — the load-independent half of the proof (a slow box only ever pushes this later,
    # never earlier).
    assert elapsed > 0.7, elapsed
    wall_budget.clear_wall_budget_extension("task-live")


@pytest.mark.asyncio
async def test_reap_loop_ignores_extension_for_a_different_task(_live_reaper):
    """Extensions are per-task; another task's grant must not leak across."""
    disp, proc, workspace = _live_reaper
    wall_budget.clear_wall_budget_extension("task-mine")
    wall_budget.grant_wall_budget_extension("task-other", 600)

    started = time.monotonic()
    outcome = await disp._reap_loop(
        proc,
        ReapWatch(workspace=workspace, done_files=(), task_id="task-mine"),
        deadline=Deadline.after(0.3),
        started=time.monotonic(),
    )
    elapsed = time.monotonic() - started

    assert outcome["timed_out"] is True, outcome
    # The other task's 600s grant would have kept this alive far past any plausible scheduling delay, so a bound this
    # loose still proves isolation.
    assert elapsed < 30.0, elapsed
    wall_budget.clear_wall_budget_extension("task-other")


def test_wall_budget_extension_registry_guards():
    """Blank ids and non-positive grants are no-ops, not stored entries."""
    wall_budget.clear_wall_budget_extension("guard-task")
    # Non-positive extra_sec must not create an entry.
    assert wall_budget.grant_wall_budget_extension("guard-task", 0) == 0.0
    assert wall_budget.grant_wall_budget_extension("guard-task", -30) == 0.0
    assert wall_budget.wall_budget_extension("guard-task") == 0.0
    # Blank / whitespace task ids are ignored rather than keyed on "".
    assert wall_budget.grant_wall_budget_extension("", 600) == 0.0
    assert wall_budget.grant_wall_budget_extension("   ", 600) == 0.0
    assert wall_budget.wall_budget_extension("") == 0.0
    # Lookups are whitespace-insensitive so a padded id still finds its grant.
    wall_budget.grant_wall_budget_extension("  guard-task  ", 120)
    assert wall_budget.wall_budget_extension("guard-task") == 120.0
    # Clearing an unknown id is a no-op, not a KeyError.
    wall_budget.clear_wall_budget_extension("never-seen")
    wall_budget.clear_wall_budget_extension("guard-task")
    assert wall_budget.wall_budget_extension("guard-task") == 0.0


def test_wall_budget_extension_stops_at_the_room_its_dispatch_left():
    """However much ``extend_lease`` asks for, the kill instant moves no further than its dispatch allowed."""
    wall_budget.limit_wall_budget_extension("capped-task", 300)
    try:
        assert wall_budget.grant_wall_budget_extension("capped-task", 200) == 200.0
        assert wall_budget.grant_wall_budget_extension("capped-task", 600) == 300.0
        assert wall_budget.wall_budget_extension("capped-task") == 300.0
        # A dispatch that already reaches the phase end leaves no room at all.
        wall_budget.clear_wall_budget_extension("capped-task")
        wall_budget.limit_wall_budget_extension("capped-task", -5)
        assert wall_budget.grant_wall_budget_extension("capped-task", 600) == 0.0
    finally:
        wall_budget.limit_wall_budget_extension("capped-task", None)
        wall_budget.clear_wall_budget_extension("capped-task")


# ── P2/T4: needs_gpu specialist runs inside a GpuSpecialistLease actor ────────
class _FakeGpuSpecialistLease:
    """Fake GpuSpecialistLease: start_async() writes done.json + log, then 'exits'."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self.started: dict[str, Any] | None = None
        self.env: dict[str, str] | None = None
        self.alive = True
        self.stopped = False

    def start_async(self, launch) -> None:
        # §3.3 non-blocking start: record + stage the done file, mark the pid ready so poll_started() returns
        # immediately on the next tick.
        self.started = {
            "cmd": launch.cmd,
            "cwd": launch.cwd,
            "log_path": launch.log_path,
            "stdin_path": launch.stdin_path,
        }
        self.env = dict(launch.env or {})
        Path(launch.log_path).write_text("stream-json log line\n", encoding="utf-8")
        # Graceful done — the reaper harvests this and exits.
        (self._workspace / "specialist_done.json").write_text(json.dumps({"proposal_set": []}), encoding="utf-8")
        self.alive = False
        self._pid = 9999

    def poll_started(self) -> int | None:
        return getattr(self, "_pid", None)

    def is_alive(self) -> bool:
        return self.alive

    def exit_code(self) -> int | None:
        return None if self.alive else 0

    def stop(self) -> bool:
        self.stopped = True
        self.alive = False
        return True

    def close(self) -> bool:
        self.alive = False
        return True


@pytest.mark.asyncio
async def test_run_routes_through_gpu_lease_and_strips_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """With a gpu_lease, run() launches inside the actor (no local Popen) and strips *_VISIBLE_DEVICES so Ray owns the card assignment (P2/T4)."""
    workspace = tmp_path / "ws"
    lease = _FakeGpuSpecialistLease(workspace)

    # Any local Popen on the Ray path is a bug — make it explode.
    import hyperloom.orchestrator.specialists.subprocess_ as sp

    def _boom(*_a, **_k):
        raise AssertionError("local Popen must not run when a gpu_lease is set")

    monkeypatch.setattr(sp.subprocess, "Popen", _boom)
    # Pretend the parent has serving GPU visibility that must NOT leak through.
    # The env allowlist already blocks every mask name; the pop below is the
    # second barrier, and it is asserted over the whole mask set so widening
    # the allowlist cannot quietly re-open a spelling it does not cover.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "6,7")
    monkeypatch.setenv("HSA_VISIBLE_DEVICES", "6,7")
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "6,7")

    cfg = SpecialistSubprocessConfig(poll_interval_seconds=0.05)
    disp = SpecialistSubprocessDispatcher(config=cfg)
    result = await disp.run(
        SpecialistSpawn(
            task_id="t-gpu",
            workspace=workspace,
            worktree=None,
            worktree_base=None,
            system_prompt="sys",
            user_prompt="usr",
            disallowed_tools=frozenset(),
            gpu_ids=(0, 1),
            deadline=Deadline.after(60.0),
            gpu_lease=lease,
        )
    )

    assert lease.started is not None, "the subprocess must run inside the lease actor"
    assert str(lease.started["log_path"]).endswith("process.log")
    # Ray owns the visible devices — the caller env must not pin them.
    assert not (GPU_MASK_ENV_NAMES & lease.env.keys())
    # The logical count is still advertised for specialist tooling.
    assert lease.env.get("INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS") == "0,1"
    assert result.done_payload is not None
    assert result.exit_code == 0


class _RayActorError(Exception):
    """Stand-in for ``ray.exceptions.RayActorError``; ray is not a test dependency."""


class _InfeasibleGpuLease(_FakeGpuSpecialistLease):
    def start_async(self, launch) -> None:
        self.fault = RayInfeasibleError("ray_cluster_infeasible: cluster has 0 GPU(s), 1 requested")
        raise self.fault


class _DeadActorGpuLease(_FakeGpuSpecialistLease):
    def start_async(self, launch) -> None:
        self.started = {"cmd": launch.cmd}

    def poll_started(self) -> int | None:
        self.fault = _RayActorError("The actor died unexpectedly before finishing this task.")
        raise self.fault


@pytest.mark.parametrize(
    ("lease_cls", "message", "error"),
    [
        (
            _InfeasibleGpuLease,
            "specialist GPU actor submit failed: ",
            "failed to submit specialist GPU actor: "
            "RayInfeasibleError('ray_cluster_infeasible: cluster has 0 GPU(s), 1 requested')",
        ),
        (
            _DeadActorGpuLease,
            "specialist GPU actor start failed: ",
            "specialist GPU actor start failed: "
            "_RayActorError('The actor died unexpectedly before finishing this task.')",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_gpu_actor_that_cannot_launch_fails_the_specialist_with_its_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    lease_cls,
    message,
    error,
):
    workspace = tmp_path / "ws"
    lease = lease_cls(workspace)
    monkeypatch.setattr(
        subprocess_.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("gpu_lease path must not spawn locally"),
    )

    with caplog.at_level(logging.WARNING, logger=subprocess_.log.name):
        result = await SpecialistSubprocessDispatcher(
            config=SpecialistSubprocessConfig(poll_interval_seconds=0.05)
        ).run(
            SpecialistSpawn(
                task_id="t-gpu-fail",
                workspace=workspace,
                worktree=None,
                worktree_base=None,
                system_prompt="sys",
                user_prompt="usr",
                disallowed_tools=frozenset(),
                deadline=Deadline.after(60.0),
                gpu_lease=lease,
            )
        )

    assert (result.done_payload, result.exit_code, result.elapsed_seconds) == (None, None, 0.0)
    assert result.process_log_path == str(workspace / "process.log")
    assert result.error == error
    (record,) = [r for r in caplog.records if r.getMessage().startswith(message)]
    assert record.exc_info[1] is lease.fault


@pytest.mark.asyncio
async def test_run_clears_stale_wall_budget_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A reused task_id must not inherit a previous run's granted extension."""
    workspace = tmp_path / "ws"
    lease = _FakeGpuSpecialistLease(workspace)
    monkeypatch.setattr(
        subprocess_.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("gpu_lease path must not spawn locally"),
    )

    # A grant left over from a prior dispatch of the same task id.
    wall_budget.grant_wall_budget_extension("t-reused", 9999)
    assert wall_budget.wall_budget_extension("t-reused") == 9999.0

    cfg = SpecialistSubprocessConfig(poll_interval_seconds=0.05)
    disp = SpecialistSubprocessDispatcher(config=cfg)
    result = await disp.run(
        SpecialistSpawn(
            task_id="t-reused",
            workspace=workspace,
            worktree=None,
            worktree_base=None,
            system_prompt="sys",
            user_prompt="usr",
            disallowed_tools=frozenset(),
            deadline=Deadline.after(60.0),
            gpu_lease=lease,
        )
    )

    assert result.exit_code == 0
    # Cleared on entry and again when the run finished.
    assert wall_budget.wall_budget_extension("t-reused") == 0.0


def test_kill_on_ray_lease_process_delegates_to_actor():
    """_kill on a _RayLeaseProcess reaps via the actor, not killpg."""
    from hyperloom.orchestrator.specialists.subprocess_ import _RayLeaseProcess

    lease = _FakeGpuSpecialistLease(Path("/tmp"))
    lease.alive = True
    handle = _RayLeaseProcess(lease, pid=1234)
    assert handle.poll() is None  # alive
    SpecialistSubprocessDispatcher._kill(handle)
    assert lease.stopped is True
    # After reap the actor reports not-alive; poll latches the exit code.
    assert handle.poll() == 0


@pytest.mark.parametrize("raw", ["not-a-number", ""])
def test_ray_specialist_pending_deadline_invalid_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
):
    """A malformed scheduling-timeout override cannot make dispatch crash."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_SPECIALIST_SCHED_TIMEOUT_SEC", raw)
    assert subprocess_._ray_specialist_pending_deadline_sec() == 300.0


def test_ray_lease_process_dead_actor_without_exit_code_is_latched():
    """An unreachable Ray actor is a terminal failure, not an endless poll."""
    from hyperloom.orchestrator.actions.executors._ray_serving import _RAY_ACTOR_DIED_RC
    from hyperloom.orchestrator.specialists.subprocess_ import _RayLeaseProcess

    class _DeadLease:
        def is_alive(self):
            return False

        def exit_code(self):
            return None

    handle = _RayLeaseProcess(_DeadLease(), pid=1234)
    assert handle.poll() == _RAY_ACTOR_DIED_RC
    # The terminal value is latched; a second poll does not query the lease.
    handle._lease = None
    assert handle.poll() == _RAY_ACTOR_DIED_RC


def test_build_claude_cmd_includes_optional_flags_and_filters_emit_intent(tmp_path: Path):
    """Optional CLI wiring is composed once and must survive as valid argv."""
    workspace = tmp_path / "workspace"
    worktree = workspace / "worktree"
    framework = tmp_path / "framework"
    for path in (workspace, worktree, framework):
        path.mkdir(parents=True, exist_ok=True)
    system_prompt_file = workspace / "system_prompt.md"

    cfg = SpecialistSubprocessConfig(
        model="claude-test",
        mcp_config_path="/tmp/mcp.json",
        framework_source_roots=(str(framework), str(framework)),
    )
    cmd = SpecialistSubprocessDispatcher(cfg)._build_claude_cmd(
        system_prompt_file=system_prompt_file,
        system_prompt="SYSTEM",
        workspace=workspace,
        worktree=worktree,
        disallowed_tools=frozenset({"KillShell", "SlashCommand"}),
    )

    assert cmd[cmd.index("--model") + 1] == "claude-test"
    assert cmd[cmd.index("--system-prompt-file") + 1] == str(system_prompt_file)
    assert system_prompt_file.read_text() == "SYSTEM"
    assert "--allowedTools" not in cmd
    assert "-p" not in cmd
    deny_idx = cmd.index("--disallowedTools") + 1
    denied = set(cmd[deny_idx].split(","))
    assert "KillShell" in denied
    assert "SlashCommand" in denied
    assert cmd[cmd.index("--agents") + 1] == build_leaf_agents_json()
    assert cmd[cmd.index("--mcp-config") + 1] == "/tmp/mcp.json"
    add_dirs = [cmd[i + 1] for i, value in enumerate(cmd[:-1]) if value == "--add-dir"]
    # Worktree first, workspace second, then each distinct framework root.
    # integrate_patch is the only writer of source; the specialist gets neither.
    assert add_dirs == [str(worktree), str(workspace)]


@pytest.mark.asyncio
async def test_partial_progress_skips_invalid_payload_and_swallow_callback_error(tmp_path: Path):
    """Malformed checkpoints and telemetry failures never terminate a run."""
    disp = SpecialistSubprocessDispatcher(SpecialistSubprocessConfig())
    invalid = tmp_path / "invalid.partial.json"
    invalid.write_text("not json", encoding="utf-8")

    async def _must_not_run(*_args):
        raise AssertionError("invalid payload must not reach the callback")

    assert (
        await disp._publish_partial_progress(
            partial_files=(invalid,),
            since_mtime=0.0,
            elapsed=1.0,
            progress_cb=_must_not_run,
        )
        == 0.0
    )

    valid = tmp_path / "valid.partial.json"
    valid.write_text('{"summary": "still working"}', encoding="utf-8")

    async def _callback_fails(*_args):
        raise RuntimeError("telemetry sink unavailable")

    newest = await disp._publish_partial_progress(
        partial_files=(valid,),
        since_mtime=0.0,
        elapsed=2.0,
        progress_cb=_callback_fails,
    )
    assert newest == valid.stat().st_mtime


@pytest.mark.asyncio
async def test_partial_checkpoint_published_while_alive(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """A checkpoint written mid-run reaches the progress callback before exit."""
    bin_dir = tmp_path / "bin"
    fake_claude = make_fake_claude(bin_dir, behavior="partial_then_done")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    seen: list[tuple[dict, float]] = []

    async def _progress(payload, elapsed):
        seen.append((payload, elapsed))

    ctx = _make_runner_ctx("t-spec-progress")
    ctx.extra["specialist_progress_cb"] = _progress

    result = await runner.run(ctx)
    assert result.status == "succeeded"
    assert seen, "no progress checkpoint was published while the run was alive"
    payload, elapsed = seen[0]
    assert payload["summary"] == "fake claude subprocess output"
    assert elapsed >= 0.0


# ---- _collect_patches: git diff harvest ------------------------------------
def _make_git_worktree_pair(tmp_path: Path) -> tuple[Path, Path]:
    import subprocess as _sp

    base = tmp_path / "base"
    base.mkdir()
    (base / "mod.py").write_text("old = 1\n", encoding="utf-8")
    _sp.run(["git", "init", "-q", str(base)], check=True)
    _sp.run(["git", "-C", str(base), "add", "-A"], check=True)
    _sp.run(
        ["git", "-C", str(base), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "base"],
        check=True,
    )
    wt = tmp_path / "worktree"
    _sp.run(["git", "-C", str(base), "worktree", "add", "-b", "sp1", str(wt)], check=True)
    return base, wt


def test_collect_patches_harvests_git_diff_when_worktree_modified(tmp_path: Path):
    base, wt = _make_git_worktree_pair(tmp_path)
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    patches, roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)
    assert len(patches) == 1
    patch_text = Path(patches[0]).read_text(encoding="utf-8")
    assert "mod.py" in patch_text
    assert str(base) in roots[patches[0]]


def test_collect_patches_harvests_a_file_the_specialist_created(tmp_path: Path):
    """git diff cannot see an untracked path; a half-patch is worse than none."""
    base, wt = _make_git_worktree_pair(tmp_path)
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    (wt / "added.py").write_text("fresh = 3\n", encoding="utf-8")

    patches, _roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)

    patch_text = Path(patches[0]).read_text(encoding="utf-8")
    assert "mod.py" in patch_text
    assert "added.py" in patch_text


def test_collect_patches_keeps_its_own_output_out_of_the_harvest(tmp_path: Path):
    """The harvest lands in patches/, which must not become part of the diff."""
    base, wt = _make_git_worktree_pair(tmp_path)
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    (wt / "patches").mkdir()
    (wt / "patches" / "stray.patch").write_text("--- a/z\n+++ b/z\n", encoding="utf-8")

    patches, _roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)

    assert "stray.patch" not in Path(patches[0]).read_text(encoding="utf-8")


def test_collect_patches_falls_back_to_disk_scan_when_git_is_unusable(tmp_path: Path, monkeypatch):
    """A raising harvest would take the done-file and usage down with it."""
    import subprocess as _sp

    base, wt = _make_git_worktree_pair(tmp_path)
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    (wt / "patches").mkdir()
    (wt / "patches" / "manual.patch").write_text("--- a/foo.py\n+++ b/foo.py\n", encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise _sp.TimeoutExpired(cmd="git", timeout=120.0)

    monkeypatch.setattr(_sp, "run", _boom)

    patches, roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)

    assert any("manual.patch" in patch for patch in patches)
    assert roots == {}


def test_collect_patches_falls_back_to_disk_scan_when_no_changes(tmp_path: Path):
    base, wt = _make_git_worktree_pair(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "patches").mkdir()
    (ws / "patches" / "manual.patch").write_text("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n", encoding="utf-8")
    patches, roots = SpecialistSubprocessDispatcher._collect_patches(wt, ws, worktree_base=base)
    assert any("manual.patch" in p for p in patches)
    assert all(p not in roots for p in patches)


def test_collect_patches_no_worktree_falls_back_to_disk_scan(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "patches").mkdir()
    (ws / "patches" / "p.patch").write_text("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", encoding="utf-8")
    patches, roots = SpecialistSubprocessDispatcher._collect_patches(None, ws)
    assert len(patches) == 1
    assert roots == {}


def test_a_ray_actor_names_no_local_process_group_for_the_operator_log():
    """An actor's ids come from the node Ray placed it on, so they mean nothing here.

    Nothing reclaims a lane from this number -- no reaper probes it -- but it is
    printed to the operator who has to clear one by hand, so a number that names
    a process on a different host would send them to the wrong machine. None is
    the honest answer for an actor; a local specialist leads its own group and
    can say so.
    """
    local = subprocess_._local_tree_pgid(type("_P", (), {"pid": 4242})())
    actor = subprocess_._local_tree_pgid(subprocess_._RayLeaseProcess(object(), 4242))

    # A local specialist leads its own group, so its root pid IS the group id.
    assert local == 4242
    assert actor is None
    # Nothing to report is also the answer when the cleanup never spawned a root.
    assert subprocess_._local_tree_pgid(None) is None


async def _spawn_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu_ids: tuple[int, ...]) -> dict[str, str]:
    """The env a local specialist spawn is handed; the fake ``Popen`` refuses to start."""
    captured: dict[str, str] = {}

    def _popen(_cmd, *, env, **_kwargs):
        captured.update(env)
        raise OSError("spawn refused by test")

    monkeypatch.setattr(subprocess_.subprocess, "Popen", _popen)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    disp = SpecialistSubprocessDispatcher(config=SpecialistSubprocessConfig(poll_interval_seconds=0.01))
    result = await disp.run(
        SpecialistSpawn(
            task_id="t-env",
            workspace=tmp_path / "ws",
            worktree=None,
            worktree_base=None,
            system_prompt="sys",
            user_prompt="usr",
            disallowed_tools=frozenset(),
            gpu_ids=gpu_ids,
            deadline=Deadline.after(5.0),
        )
    )
    assert "spawn refused by test" in (result.error or "")
    return captured


async def test_cpu_specialist_env_hides_all_gpus_and_uses_private_caches(tmp_path, monkeypatch):
    env = await _spawn_env(tmp_path, monkeypatch, gpu_ids=())
    assert all(env[var] == "" for var in GPU_MASK_ENV_NAMES)
    cache_root = tmp_path / "ws" / ".cache"
    for var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "AITER_JIT_DIR", "INFERENCE_OPTIMIZER_AITER_JIT_DIR"):
        assert Path(env[var]).parent == cache_root


async def test_gpu_specialist_env_keeps_its_cards_and_shared_caches(tmp_path, monkeypatch):
    env = await _spawn_env(tmp_path, monkeypatch, gpu_ids=(0, 1))
    assert env["HIP_VISIBLE_DEVICES"] == env["ROCR_VISIBLE_DEVICES"] == "0,1"
    assert "TRITON_CACHE_DIR" not in env and "AITER_JIT_DIR" not in env


async def _run_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **spawn_kwargs: Any) -> tuple[Any, list[dict]]:
    """Run one local spawn against a ``Popen`` that records its kwargs and refuses to start."""
    popen_kwargs: list[dict] = []

    def _popen(_cmd, **kwargs):
        popen_kwargs.append(kwargs)
        raise OSError("spawn refused by test")

    monkeypatch.setattr(subprocess_.subprocess, "Popen", _popen)
    monkeypatch.setattr(launch_env, "missing_launch_capabilities", list)
    disp = SpecialistSubprocessDispatcher(config=SpecialistSubprocessConfig(poll_interval_seconds=0.01))
    spawn = SpecialistSpawn(
        task_id="t-user",
        workspace=tmp_path / "ws",
        worktree=None,
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        disallowed_tools=frozenset(),
        deadline=Deadline.after(5.0),
        **spawn_kwargs,
    )
    return await disp.run(spawn), popen_kwargs


async def test_specialist_user_unset_spawns_exactly_as_before(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_USER", raising=False)
    result, popen_kwargs = await _run_spawn(tmp_path, monkeypatch)
    assert "spawn refused by test" in (result.error or "")
    assert len(popen_kwargs) == 1
    assert not {"user", "group", "extra_groups", "umask"} & popen_kwargs[0].keys()


@pytest.mark.parametrize(
    ("user", "needle"),
    [("no-such-user-d16", "no such user"), ("root", "uid 0")],
)
async def test_specialist_user_that_cannot_be_used_fails_the_task_without_spawning(tmp_path, monkeypatch, user, needle):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", user)
    result, popen_kwargs = await _run_spawn(tmp_path, monkeypatch)
    assert popen_kwargs == []
    assert "specialist user refused" in (result.error or "")
    assert needle in (result.error or "")


async def test_specialist_user_is_refused_on_the_ray_gpu_actor_path(tmp_path, monkeypatch, restore_umask):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", "nobody")
    result, popen_kwargs = await _run_spawn(tmp_path, monkeypatch, gpu_lease=object())
    assert popen_kwargs == []
    assert "Ray GPU actor" in (result.error or "")


_DEMOTE_USER = "nobody"


@pytest.fixture
def restore_umask():
    """Naming a specialist user sets the app's umask; do not leak that into other tests."""
    saved = os.umask(0o022)
    yield
    os.umask(saved)


def _account_exists(name: str) -> bool:
    import pwd

    try:
        pwd.getpwnam(name)
    except KeyError:
        return False
    return True


_demote_skip = pytest.mark.skipif(
    os.geteuid() != 0 or not _account_exists(_DEMOTE_USER),
    reason=f"demoting needs root and an existing {_DEMOTE_USER!r} account; the test creates no users",
)

_DEMOTED_ROLE = """#!/bin/sh
id -u > proof_uid.txt
echo "$HOME" > proof_home.txt
echo edited >> src.py
cat {secret} > proof_secret.txt 2> proof_secret.err; echo $? > proof_secret.rc
git status --porcelain > proof_git.txt 2> proof_git.err; echo $? > proof_git.rc
echo '{{"gap_canonical_id": "gap.test.example", "domain": "serving_specialist", "proposal_set": [],
"patches_written": [], "summary": "demoted", "confidence": 0.5}}' > specialist_done.json
"""


@_demote_skip
async def test_demoted_specialist_writes_its_worktree_and_cannot_read_the_apps_files(monkeypatch, restore_umask):
    import pwd
    import shutil
    import tempfile

    entry = pwd.getpwnam(_DEMOTE_USER)
    base = Path(tempfile.mkdtemp(prefix="d16."))
    try:
        base.chmod(0o755)  # the demoted account must be able to traverse to the fake CLI and the checkout
        repo = base / "framework"
        init_git_repo(repo)
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))
        monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
        private_file = base / "app_private"
        private_file.write_text("private-marker", encoding="utf-8")
        private_file.chmod(0o600)
        role = base / "bin" / "claude"
        role.parent.mkdir()
        role.write_text(_DEMOTED_ROLE.format(secret=private_file), encoding="utf-8")
        role.chmod(0o755)
        session_dir = base / "session"
        session_dir.mkdir()
        session_dir.chmod(0o755)
        runner = SpecialistRunner(
            subprocess_config=SpecialistSubprocessConfig(
                claude_executable=str(role),
                model="",
                framework_source_roots=(str(repo),),
                poll_interval_seconds=0.2,
            ),
            session_dir=session_dir,
            default_max_turns=2,
        )

        result = await runner.run(_make_runner_ctx("t-demoted"))

        workspace = session_dir / "runs" / "specialist" / "t-demoted"
        worktree = workspace / "worktree"
        proof = {p.name: p.read_text(encoding="utf-8").strip() for p in worktree.glob("proof_*")}
        assert result.status == "succeeded", result.error
        assert proof["proof_uid.txt"] == str(entry.pw_uid)
        assert proof["proof_home.txt"] == entry.pw_dir
        assert "private-marker" not in proof["proof_secret.txt"]
        assert proof["proof_secret.rc"] != "0"
        assert "Permission denied" in proof["proof_secret.err"]
        # It edited its own worktree and git there still answers (no "dubious ownership").
        assert (worktree / "src.py").read_text(encoding="utf-8").endswith("edited\n")
        assert proof["proof_git.rc"] == "0", proof["proof_git.err"]
        # The app's prompt and log stay private to it.
        assert (workspace / "prompt.md").stat().st_mode & 0o077 == 0
        assert (workspace / "process.log").stat().st_mode & 0o077 == 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


@_demote_skip
def test_demote_shares_a_worktree_that_lives_outside_the_workspace(tmp_path, monkeypatch, restore_umask):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
    request = SpecialistSpawn(
        task_id="t-outside",
        workspace=tmp_path / "ws",
        worktree=tmp_path / "elsewhere",
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        disallowed_tools=frozenset(),
        deadline=Deadline.after(5.0),
    )
    request.workspace.mkdir()
    request.worktree.mkdir()
    (request.workspace / "prompt.md").write_text("p", encoding="utf-8")
    run_as = specialist_run_as()

    kwargs = prepare_demote(run_as, request.workspace, request.worktree, {}, (request.workspace / "prompt.md",))

    assert request.worktree.stat().st_gid == kwargs["group"]
    assert request.worktree.stat().st_mode & 0o2070 == 0o2070
    assert (request.workspace / "prompt.md").stat().st_mode & 0o077 == 0


@pytest.mark.skipif(not _account_exists(_DEMOTE_USER), reason=f"no {_DEMOTE_USER!r} account")
async def test_a_launch_the_ray_path_refuses_leaves_the_apps_umask_alone(tmp_path, monkeypatch, restore_umask):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
    await _run_spawn(tmp_path, monkeypatch, gpu_lease=object())
    assert os.umask(0o022) == 0o022


@pytest.mark.skipif(not _account_exists(_DEMOTE_USER), reason=f"no {_DEMOTE_USER!r} account")
async def test_a_launch_that_names_a_user_leaves_the_apps_umask_alone(tmp_path, monkeypatch, restore_umask):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
    await _run_spawn(tmp_path, monkeypatch)
    assert os.umask(0o022) == 0o022


async def test_a_launch_with_no_user_leaves_the_apps_umask_alone(tmp_path, monkeypatch, restore_umask):
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_USER", raising=False)
    await _run_spawn(tmp_path, monkeypatch)
    assert os.umask(0o022) == 0o022


# ---- review fixes: the app never follows a link the child planted ------------------------------
def test_harvest_through_a_symlinked_patches_dir_writes_nothing_outside(tmp_path: Path):
    base, wt = _make_git_worktree_pair(tmp_path)
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (wt / "patches").symlink_to(outside)
    patches, _roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)
    assert patches == []
    assert list(outside.iterdir()) == []


def test_harvest_over_a_symlinked_patch_name_does_not_truncate_its_target(tmp_path: Path):
    base, wt = _make_git_worktree_pair(tmp_path)
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    victim = tmp_path / "app_state.json"
    victim.write_text("precious", encoding="utf-8")
    (wt / "patches").mkdir()
    (wt / "patches" / "_worktree_diff.patch").symlink_to(victim)
    patches, _roots = SpecialistSubprocessDispatcher._collect_patches(wt, tmp_path / "ws", worktree_base=base)
    assert victim.read_text(encoding="utf-8") == "precious"
    assert len(patches) == 1 and "mod.py" in Path(patches[0]).read_text(encoding="utf-8")


def test_a_symlinked_patch_file_is_not_collected(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "patches").mkdir(parents=True)
    target = tmp_path / "secret.patch"
    target.write_text("--- a/x\n+++ b/x\n", encoding="utf-8")
    (ws / "patches" / "good.patch").write_text("--- a/y\n+++ b/y\n", encoding="utf-8")
    (ws / "patches" / "link.patch").symlink_to(target)
    patches, _roots = SpecialistSubprocessDispatcher._collect_patches(None, ws)
    assert [Path(p).name for p in patches] == ["good.patch"]


def test_a_symlinked_done_file_is_not_read(tmp_path: Path):
    app_only = tmp_path / "app_only.json"
    app_only.write_text('{"app_secret": "value"}', encoding="utf-8")
    done = tmp_path / "specialist_done.json"
    done.symlink_to(app_only)
    assert SpecialistSubprocessDispatcher._read_done(done) is None
    done.unlink()
    done.write_text('{"summary": "ok"}', encoding="utf-8")
    assert SpecialistSubprocessDispatcher._read_done(done) == {"summary": "ok"}


def test_the_codex_config_is_not_written_through_a_planted_link(tmp_path: Path):
    from hyperloom.orchestrator.specialists.agent_cli_config import _write_private_codex_config

    home = tmp_path / ".codex"
    home.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    (home / "config.toml.tmp").symlink_to(victim)
    _write_private_codex_config(codex_home=home, developer_instructions="d", mcp_lines=[])
    assert victim.read_text(encoding="utf-8") == "keep"
    assert "developer_instructions" in (home / "config.toml").read_text(encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(elsewhere)
    with pytest.raises(OSError):
        _write_private_codex_config(codex_home=linked, developer_instructions="d", mcp_lines=[])


# ---- review fixes: a demoted child needs the capability to be stopped, its MCP config and its git ----------
async def test_a_launch_without_the_capability_to_stop_the_child_is_refused(tmp_path, monkeypatch, restore_umask):
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
    monkeypatch.setattr(launch_env, "missing_launch_capabilities", lambda: ["CAP_KILL"])
    disp = SpecialistSubprocessDispatcher(config=SpecialistSubprocessConfig(poll_interval_seconds=0.01))
    spawn = SpecialistSpawn(
        task_id="t-nocap",
        workspace=tmp_path / "ws2",
        worktree=None,
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        disallowed_tools=frozenset(),
        deadline=Deadline.after(5.0),
    )
    refused = await disp.run(spawn)
    assert "specialist user refused" in (refused.error or "")
    assert "CAP_KILL" in (refused.error or "")
    assert os.umask(0o022) == 0o022


def test_the_capability_mask_names_what_is_missing():
    from hyperloom.common.env_safety import missing_capabilities

    assert missing_capabilities(0xFFFFFFFFFF) == []
    assert missing_capabilities(0xFFFFFFFFFF & ~(1 << 5)) == ["CAP_KILL"]
    assert missing_capabilities(0) == ["CAP_KILL", "CAP_SETGID", "CAP_SETUID"]


def test_the_mcp_config_is_staged_in_the_workspace_for_a_demoted_child(tmp_path: Path):
    from hyperloom.common.env_safety import RunAs

    run_as = RunAs(user="u", uid=1, gid=1, groups=(1,), home="/h")
    source = tmp_path / "runtime" / "specialist_mcp.json"
    source.parent.mkdir()
    source.write_text('{"mcpServers": {}}', encoding="utf-8")
    source.chmod(0o600)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cmd = ["claude", "--mcp-config", str(source), "--print"]
    launch_env.stage_mcp_config(run_as, workspace, cmd)
    staged = workspace / "specialist_mcp.json"
    assert cmd == ["claude", "--mcp-config", str(staged), "--print"]
    assert staged.read_text(encoding="utf-8") == '{"mcpServers": {}}'
    untouched = ["claude", "--mcp-config", str(source)]
    launch_env.stage_mcp_config(None, workspace, untouched)
    launch_env.stage_mcp_config(run_as, workspace, ["claude"])
    assert untouched == ["claude", "--mcp-config", str(source)]
    with pytest.raises(subprocess_.SpecialistAgentUnavailableError):
        launch_env.stage_mcp_config(run_as, workspace, ["claude", "--mcp-config", str(tmp_path / "missing.json")])


def test_a_forged_gitfile_is_refused_not_shared(tmp_path: Path):
    from hyperloom.common.env_safety import RunAs

    run_as = RunAs(user="u", uid=1, gid=1, groups=(1,), home="/h")
    target = tmp_path / "private"
    target.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {target}\n", encoding="utf-8")
    with pytest.raises(OSError):
        run_as.share_worktree_git(wt)
    (target / "gitdir").write_text(str(wt / ".git"), encoding="utf-8")  # points back, but is not under worktrees/
    with pytest.raises(OSError):
        run_as.share_worktree_git(wt)


_TWO_TASK_ROLE = """#!/bin/sh
id -u > proof_uid.txt
while [ $# -gt 0 ]; do
  if [ "$1" = --mcp-config ]; then cat "$2" > proof_mcp.txt 2> proof_mcp.err; echo $? > proof_mcp.rc; fi
  shift
done
echo edited >> src.py
git status --porcelain > proof_git_status.txt 2> proof_git_status.err; echo $? > proof_git_status.rc
git diff > proof_git_diff.txt 2> proof_git_diff.err; echo $? > proof_git_diff.rc
git log --oneline -1 > proof_git_log.txt 2> proof_git_log.err; echo $? > proof_git_log.rc
echo '{"gap_canonical_id": "gap.test.example", "domain": "serving_specialist", "proposal_set": [],
"patches_written": [], "summary": "demoted", "confidence": 0.5}' > specialist_done.json
"""


@_demote_skip
async def test_every_demoted_specialist_gets_git_and_its_mcp_config_with_the_app_umask_007(monkeypatch, restore_umask):
    import shutil
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="d16b."))
    try:
        base.chmod(0o755)
        repo = base / "framework"
        init_git_repo(repo)
        os.umask(0o007)  # what the first launch leaves behind: every later worktree is made under it
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))
        monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
        monkeypatch.setattr(launch_env, "missing_launch_capabilities", list)
        role = base / "bin" / "claude"
        role.parent.mkdir()
        role.parent.chmod(0o755)
        role.write_text(_TWO_TASK_ROLE, encoding="utf-8")
        role.chmod(0o755)
        session_dir = base / "session"
        (session_dir / "runtime").mkdir(parents=True)
        for parent in (session_dir, session_dir / "runs", session_dir / "runs" / "specialist"):
            parent.mkdir(exist_ok=True)
            parent.chmod(0o755)  # the account traverses these; the app never changes their modes
        mcp = session_dir / "runtime" / "specialist_mcp.json"
        mcp.write_text('{"mcpServers": {"pr_monitor": {"type": "http", "url": "http://x"}}}', encoding="utf-8")
        mcp.chmod(0o600)
        runner = SpecialistRunner(
            subprocess_config=SpecialistSubprocessConfig(
                claude_executable=str(role),
                model="",
                framework_source_roots=(str(repo),),
                poll_interval_seconds=0.2,
                mcp_config_path=str(mcp),
            ),
            session_dir=session_dir,
            default_max_turns=2,
        )
        for task_id in ("t-one", "t-two"):
            result = await runner.run(_make_runner_ctx(task_id))
            worktree = session_dir / "runs" / "specialist" / task_id / "worktree"
            proof = {p.name: p.read_text(encoding="utf-8").strip() for p in worktree.glob("proof_*")}
            assert result.status == "succeeded", result.error
            for tool in ("status", "diff", "log"):
                assert proof[f"proof_git_{tool}.rc"] == "0", (task_id, tool, proof[f"proof_git_{tool}.err"])
            assert proof["proof_mcp.rc"] == "0", (task_id, proof["proof_mcp.err"])
            assert "pr_monitor" in proof["proof_mcp.txt"]
        assert mcp.stat().st_mode & 0o077 == 0  # the original stays the app's
    finally:
        shutil.rmtree(base, ignore_errors=True)


@_demote_skip
def test_a_worktree_made_under_umask_007_is_readable_to_git_once_prepared(tmp_path, monkeypatch, restore_umask):
    import shutil
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="d16c."))
    try:
        base.chmod(0o755)
        repo = base / "framework"
        init_git_repo(repo)
        os.umask(0o007)
        worktree = base / "ws" / "worktree"
        worktree.parent.mkdir(parents=True)
        worktree.parent.chmod(0o755)
        # The shape the review reproduced: the app's own git, under its own umask, makes the worktree.
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "-b", "sp", str(worktree)], check=True)
        assert (worktree / ".git").stat().st_mode & 0o004 == 0
        monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
        run_as = specialist_run_as()
        kwargs = prepare_demote(run_as, worktree.parent, worktree, {}, ())
        for args in (["status", "--porcelain"], ["diff"]):
            done = subprocess.run(
                ["git", *args],
                cwd=worktree,
                capture_output=True,
                text=True,
                check=False,
                env={
                    "PATH": os.environ["PATH"],
                    "HOME": run_as.home,
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "safe.directory",
                    "GIT_CONFIG_VALUE_0": str(worktree),
                },
                **kwargs,
            )
            assert done.returncode == 0, (args, done.stderr)
        # Read-only: the child cannot write the admin dir it was shown.
        admin = Path((worktree / ".git").read_text(encoding="utf-8").split(":", 1)[1].strip())
        # ... nor rename or add siblings in the directory above it.
        for move in (f"mv {admin} {admin}.moved", f"touch {admin.parent}/planted"):
            blocked = subprocess.run(["/bin/sh", "-c", move], capture_output=True, check=False, **kwargs)
            assert blocked.returncode != 0, move
        for victim in (admin / "planted", admin / "HEAD"):
            blocked = subprocess.run(
                ["/bin/sh", "-c", f"echo x >> {victim}"], capture_output=True, check=False, **kwargs
            )
            assert blocked.returncode != 0, victim
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_the_done_file_the_integrator_reads_is_not_followed_through_a_link(tmp_path: Path):
    from hyperloom.orchestrator.actions.executors._patch_sources import _read_done_payload

    app_only = tmp_path / "app_only.json"
    app_only.write_text('{"app_secret": 1}', encoding="utf-8")
    (tmp_path / "specialist_done.json").symlink_to(app_only)
    assert _read_done_payload(tmp_path) is None


def test_the_round_archive_does_not_copy_a_file_a_child_linked_to(tmp_path: Path):
    from hyperloom.orchestrator.enablement.artifacts import _archive_copy

    class _Archive:
        recorded: list = []

        def record(self, role, dest):
            self.recorded.append((role, dest))

    secret = tmp_path / "secret"
    secret.write_text("s", encoding="utf-8")
    (tmp_path / "specialist_done.json").symlink_to(secret)
    dest = tmp_path / "out" / "specialist_done.json"
    _archive_copy(_Archive(), "r", tmp_path / "specialist_done.json", dest, child_written=True)
    assert not dest.exists() and _Archive.recorded == []


def test_a_claimed_patch_path_through_a_planted_directory_link_is_not_accepted(tmp_path: Path):
    from hyperloom.orchestrator.specialists.runner import _resolve_claimed_patch

    wt = tmp_path / "wt"
    (wt / "real").mkdir(parents=True)
    (wt / "real" / "a.patch").write_text("--- a/x\n+++ b/x\n", encoding="utf-8")
    (wt / "patches").symlink_to(wt / "real")
    assert _resolve_claimed_patch("real/a.patch", [wt]) == str(wt / "real" / "a.patch")
    assert _resolve_claimed_patch("patches/a.patch", [wt]) is None
    assert _resolve_claimed_patch(str(wt / "patches" / "a.patch"), [wt]) is None


def test_the_archive_does_not_copy_an_applied_patch_the_child_replaced_with_a_link(tmp_path: Path):
    from hyperloom.orchestrator.enablement.artifacts import snapshot_round

    secret = tmp_path / "app_secret"
    secret.write_text("secret", encoding="utf-8")
    patch = tmp_path / "001_fix.patch"
    patch.symlink_to(secret)
    archive = snapshot_round(
        tmp_path, {"specialist_task_id": "t1", "patches_applied": [str(patch)], "enablement_round": 1}
    )
    leaked = [
        p
        for p in tmp_path.rglob("*")
        if p.is_file() and not p.is_symlink() and p != secret and p.read_text(encoding="utf-8") == "secret"
    ]
    assert leaked == []
    assert archive is not None


# ---- review r2: the snapshot's objects stay readable, and no app write follows a child's link -------------
@_demote_skip
def test_a_snapshot_refreshed_between_two_demoted_tasks_stays_readable_to_git(monkeypatch, restore_umask):
    import shutil
    import tempfile

    from hyperloom.orchestrator.specialists.tree_snapshot import snapshot_tree

    base = Path(tempfile.mkdtemp(prefix="d16s."))
    try:
        base.chmod(0o755)
        tree = base / "installed"
        tree.mkdir()
        (tree / "mod.py").write_text("x = 1\n", encoding="utf-8")
        repo = base / "snap.git"
        os.umask(0o077)  # an app that runs under a strict umask must not hide the objects it commits
        monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
        run_as = specialist_run_as()
        git_env = {
            "PATH": os.environ["PATH"],
            "HOME": run_as.home,
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
        }
        for number in (1, 2):
            if number == 2:
                (tree / "mod.py").write_text("x = 2\n", encoding="utf-8")  # a kept integrate changed the tree
            assert snapshot_tree(tree, repo, refresh=True) == (repo, "")
            workspace = base / f"ws{number}"
            worktree, err = subprocess_._setup_worktree(repo, workspace / "worktree", f"task-{number}")
            assert worktree is not None, err
            kwargs = prepare_demote(run_as, workspace, worktree, {}, ())
            for args in (["status", "--porcelain"], ["log", "--oneline", "-1"], ["diff"]):
                done = subprocess.run(
                    ["git", *args],
                    cwd=worktree,
                    capture_output=True,
                    text=True,
                    check=False,
                    env={**git_env, "GIT_CONFIG_VALUE_0": str(worktree)},
                    **kwargs,
                )
                assert done.returncode == 0, (number, args, done.stderr)
    finally:
        shutil.rmtree(base, ignore_errors=True)


async def test_the_dispatcher_does_not_write_the_prompt_or_log_through_a_planted_link(tmp_path, monkeypatch):
    victim = tmp_path / "victim"
    victim.write_text("app only", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for name in ("prompt.md", "process.log", "process.log.redacting"):
        (workspace / name).symlink_to(victim)
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_USER", raising=False)
    await _run_spawn(tmp_path, monkeypatch)
    assert victim.read_text(encoding="utf-8") == "app only"
    assert not (workspace / "prompt.md").is_symlink()


def test_the_dsh_home_is_not_written_through_a_link_the_account_left(tmp_path):
    from hyperloom.orchestrator.specialists import dsh_launch

    victim_dir = tmp_path / "elsewhere"
    victim_dir.mkdir()
    victim_dir.chmod(0o755)
    (victim_dir / "AGENTS.md").write_text("app only", encoding="utf-8")
    home = tmp_path / "ws" / ".dsh"
    home.parent.mkdir()
    home.symlink_to(victim_dir)
    with pytest.raises(subprocess_.SpecialistAgentUnavailableError):
        dsh_launch._prepare_home(home, instructions="i", patch="p")
    assert (victim_dir / "AGENTS.md").read_text(encoding="utf-8") == "app only"
    assert victim_dir.stat().st_mode & 0o777 == 0o755
    real = tmp_path / "ws2" / ".dsh"
    real.mkdir(parents=True)
    (real / "AGENTS.md").symlink_to(victim_dir / "AGENTS.md")
    dsh_launch._prepare_home(real, instructions="i", patch="p")
    assert (victim_dir / "AGENTS.md").read_text(encoding="utf-8") == "app only"
    assert (real / "AGENTS.md").read_text(encoding="utf-8") == "i"


def test_dsh_profiles_are_not_removed_through_a_linked_home(tmp_path):
    from hyperloom.orchestrator.specialists import dsh_launch

    victim_dir = tmp_path / "elsewhere"
    (victim_dir / "profiles").mkdir(parents=True)
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / ".dsh").symlink_to(victim_dir)
    dsh_launch.remove_dsh_profiles(tmp_path / "ws")
    assert (victim_dir / "profiles").is_dir()


def test_the_dsh_trace_does_not_read_a_session_log_through_a_link(tmp_path):
    from hyperloom.inference_optimizer.trace.parse_dsh_session import parse_dsh_session_trace

    outside = tmp_path / "outside"
    (outside / "x").mkdir(parents=True)
    (outside / "x" / "session.v3.jsonl").write_text('{"type": "session"}\n', encoding="utf-8")
    home = tmp_path / ".dsh"
    (home / "sessions").mkdir(parents=True)
    (home / "sessions" / "a").symlink_to(outside)
    trace = parse_dsh_session_trace(home)
    assert "no session log" in trace.error


# ---- review r2: the app's own git never runs what a child planted in the worktree ----------------
_FSMONITOR_HOOK = "#!/bin/sh\necho ran > {marker}\nprintf ''\n"


def _plant_hostile_repo(where: Path, marker: Path) -> None:
    """Make ``where`` a repository whose config starts a program on every index read."""
    hook = where.parent / "hook.sh"
    hook.write_text(_FSMONITOR_HOOK.format(marker=marker), encoding="utf-8")
    hook.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(where)], check=True)
    subprocess.run(["git", "-C", str(where), "config", "core.fsmonitor", str(hook)], check=True)
    subprocess.run(["git", "-C", str(where), "config", "core.hooksPath", str(where.parent)], check=True)


def test_the_harvest_does_not_run_a_program_a_child_planted_in_a_replaced_dot_git(tmp_path: Path):
    base, wt = _make_git_worktree_pair(tmp_path)
    marker = tmp_path / "executed"
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    (wt / ".git").unlink()
    _plant_hostile_repo(wt, marker)  # `.git` becomes a directory with its own config
    subprocess.run(["git", "-C", str(wt), "add", "-A"], check=True, capture_output=True)
    marker.unlink(missing_ok=True)
    assert SpecialistSubprocessDispatcher._harvest_worktree_diff(wt) == ""
    assert worktree_head(wt) == ""
    assert not marker.exists()


@_demote_skip
def test_a_demoted_child_cannot_make_the_apps_git_run_its_fsmonitor(monkeypatch):
    import shutil
    import tempfile

    base = Path(tempfile.mkdtemp(prefix="d16g."))
    try:
        base.chmod(0o755)
        repo = base / "framework"
        init_git_repo(repo)
        workspace = base / "ws"
        worktree, err = subprocess_._setup_worktree(repo, workspace / "worktree", "task-g")
        assert worktree is not None, err
        monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", _DEMOTE_USER)
        run_as = specialist_run_as()
        kwargs = prepare_demote(run_as, workspace, worktree, {}, ())
        marker = base / "executed"
        marker.write_text("", encoding="utf-8")
        marker.unlink()
        hook = workspace / "hook.sh"
        hook.write_text(_FSMONITOR_HOOK.format(marker=marker), encoding="utf-8")
        hook.chmod(0o770)
        planted = f"""set -e
cd {worktree}
rm -f .git
git init -q .
git config core.fsmonitor {hook}
echo '* filter=evil diff=evil' > .gitattributes
echo changed >> mod.py
"""
        done = subprocess.run(
            ["/bin/sh", "-c", planted],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": os.environ["PATH"],
                "HOME": run_as.home,
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": "*",
            },
            **kwargs,
        )
        assert done.returncode == 0, done.stderr
        assert SpecialistSubprocessDispatcher._harvest_worktree_diff(worktree) == ""
        assert worktree_head(worktree) == ""
        assert not marker.exists()
        # The second shape: a gitfile that names an admin dir the child made itself, with a matching back pointer.
        forged = f"""set -e
cd {worktree}
rm -rf .git
mkdir -p {workspace}/fake/.git/worktrees/w
cd {workspace}/fake && git init -q . && git config core.fsmonitor {hook}
cd {worktree}
echo "{worktree}/.git" > {workspace}/fake/.git/worktrees/w/gitdir
echo "gitdir: {workspace}/fake/.git/worktrees/w" > .git
"""
        done = subprocess.run(
            ["/bin/sh", "-c", forged],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": os.environ["PATH"],
                "HOME": run_as.home,
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": "*",
            },
            **kwargs,
        )
        assert done.returncode == 0, done.stderr
        assert SpecialistSubprocessDispatcher._harvest_worktree_diff(worktree) == ""
        assert worktree_head(worktree) == ""
        assert not marker.exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_the_harvest_still_works_on_an_untouched_worktree(tmp_path: Path):
    base, wt = _make_git_worktree_pair(tmp_path)
    (wt / "mod.py").write_text("new = 2\n", encoding="utf-8")
    assert "+new = 2" in SpecialistSubprocessDispatcher._harvest_worktree_diff(wt)
    assert len(worktree_head(wt)) == 40
