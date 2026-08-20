# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SpecialistRunner subprocess + worktree tests.

Pins the production specialist dispatch: per-task git worktree, the
``claude --print --add-dir ...`` spawn, done.json + patch harvesting, and the
tool whitelist. Uses a hermetic fake ``claude`` shell script.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from .conftest import init_git_repo

from hyperloom.orchestrator.specialists.runner import (
    DEFAULT_SPECIALIST_TOOLS,
    SPECIALIST_TOOL_DENYLIST,
    SpecialistRunner,
)
from hyperloom.orchestrator.specialists import subprocess_
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _build_specialist_env,
    _pick_worktree_base,
    _setup_worktree,
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


def test_build_specialist_env_forwards_oauth_token_without_mirroring_it(monkeypatch):
    """A subscription-only parent must hand the token down untouched.

    Mirroring it into either API-key var would drop the child out of
    subscription mode and 401 it.
    """
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


def _make_fake_claude(
    bin_dir: Path,
    *,
    behavior: str,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write a fake ``claude`` executable simulating one of: done_only / done_with_patch / done_with_env / crash."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script_path = bin_dir / "claude"
    payload_json = json.dumps(
        payload
        or {
            "gap_canonical_id": "gap.test.example",
            "domain": "serving_specialist",
            "proposal_set": [
                {
                    "name": "fake_variant",
                    "extra_args": "--fake",
                    "extra_envs": {},
                    "reason": "fake",
                }
            ],
            "patches_written": [],
            "empty": False,
            "summary": "fake claude subprocess output",
            "confidence": 0.5,
        }
    )
    body = """#!/usr/bin/env bash
set -e
# Parse --add-dir paths (first is worktree, second is workspace).
ADD_DIRS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --add-dir) ADD_DIRS+=("$2"); shift 2 ;;
    *) shift ;;
  esac
done
WORKTREE="${ADD_DIRS[0]:-}"
WORKSPACE="${ADD_DIRS[1]:-}"
if [[ -n "$WORKTREE" && -f "$WORKTREE/prompt.md" ]]; then
  WORKSPACE="$WORKTREE"
fi
"""
    if behavior == "done_only":
        body += f"""
cat > "$WORKSPACE/specialist_done.json" <<'EOF'
{payload_json}
EOF
exit 0
"""
    elif behavior == "done_with_patch":
        patch_payload = json.dumps(
            {
                **(payload or {}),
                "gap_canonical_id": "gap.test.example",
                "domain": "serving_specialist",
                "proposal_set": [
                    {
                        "name": "patched_variant",
                        "extra_args": "",
                        "extra_envs": {},
                        "reason": "see patch",
                    }
                ],
                "patches_written": ["patches/001_test.patch"],
                "empty": False,
                "summary": "fake patch-authoring specialist",
                "confidence": 0.7,
            }
        )
        body += f"""
mkdir -p "$WORKTREE/patches"
cat > "$WORKTREE/patches/001_test.patch" <<'EOF'
diff --git a/dummy.txt b/dummy.txt
new file mode 100644
--- /dev/null
+++ b/dummy.txt
@@ -0,0 +1 @@
+pr-a2 patch
EOF
cat > "$WORKSPACE/specialist_done.json" <<'EOF'
{patch_payload}
EOF
exit 0
"""
    elif behavior == "done_with_env":
        body += """
cat > "$WORKSPACE/specialist_done.json" <<EOF
{
  "gap_canonical_id": "gap.test.example",
  "domain": "serving_specialist",
  "proposal_set": [],
  "patches_written": [],
  "empty": true,
  "summary": "env echo",
  "confidence": 0.0,
  "hip_visible": "$HIP_VISIBLE_DEVICES",
  "cuda_visible": "$CUDA_VISIBLE_DEVICES",
  "rocr_visible": "$ROCR_VISIBLE_DEVICES"
}
EOF
exit 0
"""
    elif behavior == "done_with_llm_env":
        # Echo the LLM-transport stability env for the dispatcher assertion.
        body += """
cat > "$WORKSPACE/specialist_done.json" <<EOF
{
  "gap_canonical_id": "gap.test.example",
  "domain": "serving_specialist",
  "proposal_set": [],
  "patches_written": [],
  "empty": true,
  "summary": "llm env echo",
  "confidence": 0.0,
  "api_timeout_ms": "$API_TIMEOUT_MS",
  "disable_autoupdater": "$DISABLE_AUTOUPDATER",
  "disable_nonessential": "$CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"
}
EOF
exit 0
"""
    elif behavior == "crash":
        body += "exit 3\n"
    elif behavior == "partial_then_crash":
        # Write only the partial checkpoint, then die before the final done.json.
        body += f"""
cat > "$WORKSPACE/specialist_done.partial.json" <<'EOF'
{payload_json}
EOF
exit 3
"""
    elif behavior == "partial_then_done":
        # Checkpoint first, wait for the reaper to see it, then exit normally.
        body += f"""
cat > "$WORKSPACE/specialist_done.partial.json" <<'EOF'
{payload_json}
EOF
sleep 1
cat > "$WORKSPACE/specialist_done.json" <<'EOF'
{payload_json}
EOF
exit 0
"""
    elif behavior == "hang":
        # Sleep past any wall budget without writing done.json.
        body += "sleep 600\n"
    else:
        raise ValueError(f"unknown behavior {behavior!r}")
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


@pytest.fixture
def fake_framework_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "framework"
    init_git_repo(repo)
    return repo


def _make_runner_ctx(task_id: str = "t-spec-1") -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="specialist",
        state="queued",
        params={
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.test.example",
            "max_turns": 2,
        },
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={})


def test_runner_requires_exactly_one_dispatch_mode():
    with pytest.raises(ValueError, match="exactly one"):
        SpecialistRunner()
    with pytest.raises(ValueError, match="mutually exclusive"):
        SpecialistRunner(
            backend_factory=lambda d: None,
            subprocess_config=SpecialistSubprocessConfig(),
        )


def test_runner_accepts_subprocess_config_only():
    runner = SpecialistRunner(
        subprocess_config=SpecialistSubprocessConfig(),
    )
    assert runner.subprocess_dispatcher is not None
    assert runner.backend_factory is None


def test_default_tools_include_write_capabilities():
    """Edit/Write/MultiEdit are lifted out of the denylist for worktree patch authoring."""
    for tool in ("Edit", "Write", "MultiEdit"):
        assert tool in DEFAULT_SPECIALIST_TOOLS
        assert tool not in SPECIALIST_TOOL_DENYLIST


def test_kb_write_tools_not_in_default_specialist_tools():
    """KB lifecycle stays Coordinator-owned; the specialist KB MCP was removed.

    Asserted by shape rather than by a fixed tool name, so re-introducing a KB
    MCP server under any name is caught (the old ``mcp__cortex_kb__*`` names no
    longer exist to assert against).
    """
    kb_mcp_tools = [t for t in DEFAULT_SPECIALIST_TOOLS if t.startswith("mcp__") and "kb" in t.lower()]
    assert kb_mcp_tools == [], f"specialist toolset exposes KB MCP tools: {kb_mcp_tools}"
    denylisted_kb_mcp = [t for t in SPECIALIST_TOOL_DENYLIST if t.startswith("mcp__") and "kb" in t.lower()]
    assert denylisted_kb_mcp == [], f"stale KB MCP entries in the denylist: {denylisted_kb_mcp}"


def test_task_allowed_tools_override_default_patch_tools():
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig())
    tools = runner._resolve_tools(["Read", "Grep", "Glob", "Write"])
    assert tools == ("Read", "Grep", "Glob", "Write")
    assert "Edit" not in tools
    assert "MultiEdit" not in tools
    assert "Bash" not in tools


def test_pick_worktree_base_picks_first_git_root(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    nonrepo = tmp_path / "not-a-repo"
    nonrepo.mkdir()
    base = _pick_worktree_base((str(nonrepo), str(fake_framework_repo)))
    assert base is not None
    assert base.samefile(fake_framework_repo)


def test_pick_worktree_base_returns_none_when_no_repo(tmp_path: Path):
    nonrepo = tmp_path / "not-a-repo"
    nonrepo.mkdir()
    base = _pick_worktree_base((str(nonrepo),))
    assert base is None


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
    fake_claude = _make_fake_claude(bin_dir, behavior="done_only")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=30.0,
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
    assert result.specialist_done["empty"] is False
    assert result.specialist_done["domain"] == "serving_specialist"
    workspace = session_dir / "runs" / "specialist" / "t-spec-done"
    assert (workspace / "specialist_done.json").exists()
    assert (workspace / "process.log").exists()
    assert (workspace / "worktree").is_dir()


@pytest.mark.asyncio
async def test_local_specialist_spawn_uses_devnull_stdin(
    tmp_path: Path,
    fake_framework_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The local specialist path must never inherit Coordinator stdin."""
    bin_dir = tmp_path / "bin"
    fake_claude = _make_fake_claude(bin_dir, behavior="done_only")
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
            per_turn_max_seconds=30.0,
            poll_interval_seconds=0.2,
        ),
        session_dir=session_dir,
        default_max_turns=2,
    )

    result = await runner.run(_make_runner_ctx("t-spec-devnull"))

    assert result.status == "succeeded"
    assert seen_stdin == [subprocess.DEVNULL]


@pytest.mark.asyncio
async def test_subprocess_path_injects_allocated_gpu_env(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    bin_dir = tmp_path / "bin"
    fake_claude = _make_fake_claude(bin_dir, behavior="done_with_env")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=30.0,
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
    """The dispatcher injects low-risk claude-code stability flags but does not
    set API_TIMEOUT_MS by default; liveness is governed by the process.log /
    heartbeat stale reaper."""
    # Ensure no inherited values mask the setdefault under test.
    for var in (
        "API_TIMEOUT_MS",
        "DISABLE_AUTOUPDATER",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    ):
        monkeypatch.delenv(var, raising=False)

    bin_dir = tmp_path / "bin"
    fake_claude = _make_fake_claude(bin_dir, behavior="done_with_llm_env")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=30.0,
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
    fake_claude = _make_fake_claude(bin_dir, behavior="done_only")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=30.0,
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
    fake_claude = _make_fake_claude(bin_dir, behavior="done_with_patch")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=30.0,
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
    fake_claude = _make_fake_claude(bin_dir, behavior="crash")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=15.0,
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
    assert result.specialist_done["empty"] is True
    assert "subprocess" in (result.error or "")


@pytest.mark.asyncio
async def test_subprocess_path_isolates_writes_to_worktree(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """Worktree patches must NOT appear in the base repo's working tree until ``integrate_patch`` applies them."""
    bin_dir = tmp_path / "bin"
    fake_claude = _make_fake_claude(bin_dir, behavior="done_with_patch")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=30.0,
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
    """A specialist that wrote only the partial (then died before the final
    done.json) surfaces the partial as a non-empty result."""
    bin_dir = tmp_path / "bin"
    fake_claude = _make_fake_claude(bin_dir, behavior="partial_then_crash")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=15.0,
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
    assert result.specialist_done["empty"] is False
    assert result.specialist_done.get("_recovered_from_partial") is True
    assert result.specialist_done["proposal_set"]


@pytest.mark.asyncio
async def test_wall_budget_overrides_legacy_max_seconds(
    tmp_path: Path,
    fake_framework_repo: Path,
):
    """A small Coordinator-injected ``wall_budget_sec`` must kill a hung
    specialist well before the legacy ``max_turns × per_turn`` ceiling (here
    2 × 15 = 30s)."""
    bin_dir = tmp_path / "bin"
    fake_claude = _make_fake_claude(bin_dir, behavior="hang")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=15.0,
        poll_interval_seconds=0.2,
    )
    runner = SpecialistRunner(
        subprocess_config=config,
        session_dir=session_dir,
        default_max_turns=2,
    )
    ctx = _make_runner_ctx("t-spec-budget")
    ctx.extra["wall_budget_sec"] = 1.0

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
    """A specialist that streams to process.log but never self-writes
    heartbeat.json must NOT be reaped as stale."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    process_log = workspace / "process.log"
    process_log.write_text("start\n", encoding="utf-8")
    heartbeat_file = workspace / "heartbeat.json"  # never written

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
        proc=proc,
        workspace=workspace,
        done_files=(),
        heartbeat_file=heartbeat_file,
        max_seconds=60.0,
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
    """With neither heartbeat.json nor process.log activity, the reaper
    still reaps a silent/hung subprocess as stale."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # No process.log, no heartbeat.json — total silence.
    heartbeat_file = workspace / "heartbeat.json"

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
        proc=proc,
        workspace=workspace,
        done_files=(),
        heartbeat_file=heartbeat_file,
        max_seconds=60.0,
        started=time.monotonic(),
    )
    assert outcome["stale_heartbeat"] is True, outcome
    assert killed["v"] is True


# ── extend_lease moves the live wall-clock deadline ──────────────────────────
@pytest.fixture
def _live_reaper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A reaper whose only stop condition is the hard wall-clock cap.

    Keeps process.log fresh so the staleness path never fires, and stubs
    ``_kill`` so the timeout never signals a real process group (the fake
    proc reuses this pytest process's pid).
    """
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
    subprocess_.clear_wall_budget_extension("task-base")

    started = time.monotonic()
    outcome = await disp._reap_loop(
        proc=proc,
        workspace=workspace,
        done_files=(),
        heartbeat_file=workspace / "heartbeat.json",
        max_seconds=0.3,
        started=time.monotonic(),
        task_id="task-base",
    )
    elapsed = time.monotonic() - started

    assert outcome["timed_out"] is True, outcome
    # Killed at ~0.3s. The bound is loose because only the direction matters:
    # a loaded CI box can stretch this, but it can never finish early.
    assert elapsed < 5.0, elapsed


@pytest.mark.asyncio
async def test_reap_loop_deadline_moves_when_extension_granted_mid_run(_live_reaper):
    """The regression this fix exists for.

    ``extend_lease`` used to push the task / lane / GPU leases out while the
    subprocess kept the ``max_seconds`` deadline computed once at spawn, so the
    specialist still died on schedule. The reaper must re-read the extension
    every poll.
    """
    disp, proc, workspace = _live_reaper
    subprocess_.clear_wall_budget_extension("task-live")

    started = time.monotonic()
    loop = asyncio.create_task(
        disp._reap_loop(
            proc=proc,
            workspace=workspace,
            done_files=(),
            heartbeat_file=workspace / "heartbeat.json",
            max_seconds=0.3,
            started=time.monotonic(),
            task_id="task-live",
        )
    )
    # Grant the extension while the run is still in flight, before the
    # original 0.3s cap would have fired.
    await asyncio.sleep(0.15)
    subprocess_.grant_wall_budget_extension("task-live", 0.6)
    # The reaper recomputes `max_seconds + wall_budget_extension(task_id)`
    # every poll, so this is the deadline it now enforces.
    assert subprocess_.wall_budget_extension("task-live") == 0.6
    outcome = await loop
    elapsed = time.monotonic() - started

    assert outcome["timed_out"] is True, outcome
    # Survived past the base cap — the load-independent half of the proof
    # (a slow box only ever pushes this later, never earlier).
    assert elapsed > 0.7, elapsed
    subprocess_.clear_wall_budget_extension("task-live")


@pytest.mark.asyncio
async def test_reap_loop_ignores_extension_for_a_different_task(_live_reaper):
    """Extensions are per-task; another task's grant must not leak across."""
    disp, proc, workspace = _live_reaper
    subprocess_.clear_wall_budget_extension("task-mine")
    subprocess_.grant_wall_budget_extension("task-other", 600)

    started = time.monotonic()
    outcome = await disp._reap_loop(
        proc=proc,
        workspace=workspace,
        done_files=(),
        heartbeat_file=workspace / "heartbeat.json",
        max_seconds=0.3,
        started=time.monotonic(),
        task_id="task-mine",
    )
    elapsed = time.monotonic() - started

    assert outcome["timed_out"] is True, outcome
    # The other task's 600s grant would have kept this alive far past any
    # plausible scheduling delay, so a bound this loose still proves isolation.
    assert elapsed < 30.0, elapsed
    subprocess_.clear_wall_budget_extension("task-other")


def test_wall_budget_extension_registry_guards():
    """Blank ids and non-positive grants are no-ops, not stored entries."""
    subprocess_.clear_wall_budget_extension("guard-task")
    # Non-positive extra_sec must not create an entry.
    assert subprocess_.grant_wall_budget_extension("guard-task", 0) == 0.0
    assert subprocess_.grant_wall_budget_extension("guard-task", -30) == 0.0
    assert subprocess_.wall_budget_extension("guard-task") == 0.0
    # Blank / whitespace task ids are ignored rather than keyed on "".
    assert subprocess_.grant_wall_budget_extension("", 600) == 0.0
    assert subprocess_.grant_wall_budget_extension("   ", 600) == 0.0
    assert subprocess_.wall_budget_extension("") == 0.0
    # Lookups are whitespace-insensitive so a padded id still finds its grant.
    subprocess_.grant_wall_budget_extension("  guard-task  ", 120)
    assert subprocess_.wall_budget_extension("guard-task") == 120.0
    # Clearing an unknown id is a no-op, not a KeyError.
    subprocess_.clear_wall_budget_extension("never-seen")
    subprocess_.clear_wall_budget_extension("guard-task")
    assert subprocess_.wall_budget_extension("guard-task") == 0.0


# ── P2/T4: needs_gpu specialist runs inside a GpuSpecialistLease actor ────────
class _FakeGpuSpecialistLease:
    """Fake GpuSpecialistLease: start_async() writes done.json + log, then 'exits'."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self.started: dict[str, Any] | None = None
        self.env: dict[str, str] | None = None
        self.alive = True
        self.stopped = False

    def start_async(
        self,
        cmd,
        *,
        env=None,
        cwd=None,
        log_path=None,
        env_mode="merge",
        stdin_path=None,
    ) -> None:
        # §3.3 non-blocking start: record + stage the done file, mark the pid
        # ready so poll_started() returns immediately on the next tick.
        self.started = {
            "cmd": cmd,
            "cwd": cwd,
            "log_path": log_path,
            "env_mode": env_mode,
            "stdin_path": stdin_path,
        }
        self.env = dict(env or {})
        Path(log_path).write_text("stream-json log line\n", encoding="utf-8")
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

    def stop(self) -> None:
        self.stopped = True
        self.alive = False

    def close(self) -> None:
        self.alive = False


@pytest.mark.asyncio
async def test_run_routes_through_gpu_lease_and_strips_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """With a gpu_lease, run() launches inside the actor (no local Popen) and
    strips *_VISIBLE_DEVICES so Ray owns the card assignment (P2/T4)."""
    workspace = tmp_path / "ws"
    lease = _FakeGpuSpecialistLease(workspace)

    # Any local Popen on the Ray path is a bug — make it explode.
    import hyperloom.orchestrator.specialists.subprocess_ as sp

    def _boom(*_a, **_k):
        raise AssertionError("local Popen must not run when a gpu_lease is set")

    monkeypatch.setattr(sp.subprocess, "Popen", _boom)
    # Pretend the parent has serving GPU visibility that must NOT leak through.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "6,7")

    cfg = SpecialistSubprocessConfig(poll_interval_seconds=0.05)
    disp = SpecialistSubprocessDispatcher(config=cfg)
    result = await disp.run(
        task_id="t-gpu",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        allowed_tools=(),
        max_turns=1,
        gpu_ids=(0, 1),
        wall_budget_sec=60.0,
        gpu_lease=lease,
    )

    assert lease.started is not None, "the subprocess must run inside the lease actor"
    assert str(lease.started["log_path"]).endswith("process.log")
    # Ray owns the visible devices — the caller env must not pin them.
    assert "ROCR_VISIBLE_DEVICES" not in lease.env
    assert "HIP_VISIBLE_DEVICES" not in lease.env
    assert "CUDA_VISIBLE_DEVICES" not in lease.env
    # The logical count is still advertised for specialist tooling.
    assert lease.env.get("INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS") == "0,1"
    assert result.done_payload is not None
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_run_clears_stale_wall_budget_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A reused task_id must not inherit a previous run's granted extension.

    The registry is keyed by task_id and lives for the process, so a grant left
    behind by an earlier dispatch would silently widen the next run's deadline.
    """
    workspace = tmp_path / "ws"
    lease = _FakeGpuSpecialistLease(workspace)
    monkeypatch.setattr(
        subprocess_.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("gpu_lease path must not spawn locally"),
    )

    # A grant left over from a prior dispatch of the same task id.
    subprocess_.grant_wall_budget_extension("t-reused", 9999)
    assert subprocess_.wall_budget_extension("t-reused") == 9999.0

    cfg = SpecialistSubprocessConfig(poll_interval_seconds=0.05)
    disp = SpecialistSubprocessDispatcher(config=cfg)
    result = await disp.run(
        task_id="t-reused",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        allowed_tools=(),
        max_turns=1,
        wall_budget_sec=60.0,
        gpu_lease=lease,
    )

    assert result.exit_code == 0
    # Cleared on entry and again when the run finished.
    assert subprocess_.wall_budget_extension("t-reused") == 0.0


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
    prompt = workspace / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")

    cfg = SpecialistSubprocessConfig(
        model="claude-test",
        mcp_config_path="/tmp/mcp.json",
        framework_source_roots=(str(framework), str(framework)),
        extra_claude_args=("--debug",),
        leaf_agents_json='{"researcher": {"description": "test"}}',
    )
    cmd = SpecialistSubprocessDispatcher(cfg)._build_claude_cmd(
        prompt_file=prompt,
        workspace=workspace,
        worktree=worktree,
        allowed_tools=("Read", "Task", "emit_intent"),
    )

    assert cmd[cmd.index("--model") + 1] == "claude-test"
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Task"
    assert "emit_intent" not in cmd
    assert cmd[cmd.index("--agents") + 1] == cfg.leaf_agents_json
    assert cmd[cmd.index("--mcp-config") + 1] == "/tmp/mcp.json"
    assert cmd[-1] == "--debug"
    add_dirs = [cmd[i + 1] for i, value in enumerate(cmd[:-1]) if value == "--add-dir"]
    # Worktree first, workspace second, then each distinct framework root.
    assert add_dirs == [str(worktree), str(workspace), str(framework)]


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
    fake_claude = _make_fake_claude(bin_dir, behavior="partial_then_done")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",
        framework_source_roots=(str(fake_framework_repo),),
        per_turn_max_seconds=15.0,
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
