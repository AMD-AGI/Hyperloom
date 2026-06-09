# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-A2 (Arbor-into-Hyperloom): SpecialistRunner subprocess + worktree.

Pins the production specialist dispatch: per-task git worktree, the
``claude --print --add-dir ...`` spawn, done.json + patch harvesting, and the
PR-A2 tool whitelist change. Uses a hermetic fake ``claude`` shell script.
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

from inference_optimizer.orchestrator.specialist_runner import (
    DEFAULT_SPECIALIST_TOOLS,
    SPECIALIST_TOOL_DENYLIST,
    SpecialistRunner,
)
from inference_optimizer.orchestrator.specialist_subprocess import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _pick_worktree_base,
    _setup_worktree,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# Fixtures
def _init_git_repo(path: Path) -> None:
    """Initialise a minimal git repo with one commit so ``git worktree add`` can branch off it."""
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "PR-A2 Test"
    env["GIT_AUTHOR_EMAIL"] = "pr-a2@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True, capture_output=True, env=env,
    )
    (path / "README.md").write_text("# pr-a2 test repo\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True, capture_output=True, env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        check=True, capture_output=True, env=env,
    )


def _make_fake_claude(
    bin_dir: Path, *, behavior: str, payload: dict[str, Any] | None = None,
) -> Path:
    """Write a fake ``claude`` executable simulating one of: done_only / done_with_patch / done_with_env / crash."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script_path = bin_dir / "claude"
    payload_json = json.dumps(payload or {
        "gap_canonical_id": "gap.test.example",
        "domain": "serving_specialist",
        "proposal_set": [{
            "name": "fake_variant", "extra_args": "--fake",
            "extra_envs": {}, "reason": "fake",
        }],
        "patches_written": [],
        "empty": False,
        "summary": "fake claude subprocess output",
        "confidence": 0.5,
    })
    body = f"""#!/usr/bin/env bash
set -e
# Parse --add-dir paths (first is worktree, second is workspace).
ADD_DIRS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --add-dir) ADD_DIRS+=("$2"); shift 2 ;;
    *) shift ;;
  esac
done
WORKTREE="${{ADD_DIRS[0]:-}}"
WORKSPACE="${{ADD_DIRS[1]:-}}"
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
        patch_payload = json.dumps({
            **(payload or {}),
            "gap_canonical_id": "gap.test.example",
            "domain": "serving_specialist",
            "proposal_set": [{
                "name": "patched_variant", "extra_args": "",
                "extra_envs": {}, "reason": "see patch",
            }],
            "patches_written": ["patches/001_test.patch"],
            "empty": False,
            "summary": "fake patch-authoring specialist",
            "confidence": 0.7,
        })
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
    elif behavior == "crash":
        body += "exit 3\n"
    else:
        raise ValueError(f"unknown behavior {behavior!r}")
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script_path


@pytest.fixture
def fake_framework_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "framework"
    _init_git_repo(repo)
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


# 1. Constructor invariants
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


# 2. Tool whitelist updates
def test_default_tools_include_write_capabilities():
    """PR-A2 lifted Edit/Write/MultiEdit out of the denylist for worktree patch authoring."""
    for tool in ("Edit", "Write", "MultiEdit"):
        assert tool in DEFAULT_SPECIALIST_TOOLS
        assert tool not in SPECIALIST_TOOL_DENYLIST


def test_kb_write_tools_remain_denied():
    """KB lifecycle stays Coordinator-owned (Inv-2 / Inv-6.1)."""
    for kb_tool in (
        "mcp__cortex_kb__propose_point",
    ):
        assert kb_tool in SPECIALIST_TOOL_DENYLIST


def test_task_allowed_tools_override_default_patch_tools():
    runner = SpecialistRunner(subprocess_config=SpecialistSubprocessConfig())
    tools = runner._resolve_tools(["Read", "Grep", "Glob", "Write"])
    assert tools == ("Read", "Grep", "Glob", "Write")
    assert "Edit" not in tools
    assert "MultiEdit" not in tools
    assert "Bash" not in tools


# 3. Worktree helpers
def test_pick_worktree_base_picks_first_git_root(
    tmp_path: Path, fake_framework_repo: Path,
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
    tmp_path: Path, fake_framework_repo: Path,
):
    workspace = tmp_path / "workspace"
    worktree, err = _setup_worktree(
        fake_framework_repo, workspace / "worktree", "specialist-test1",
    )
    assert err == "", err
    assert worktree is not None
    assert worktree.is_dir()
    cp = subprocess.run(
        ["git", "-C", str(fake_framework_repo), "branch", "--list",
         "specialist-test1"],
        capture_output=True, text=True, check=True,
    )
    assert "specialist-test1" in cp.stdout


# 4. End-to-end subprocess dispatch with the fake `claude` binary
@pytest.mark.asyncio
async def test_subprocess_path_harvests_done_file(
    tmp_path: Path, fake_framework_repo: Path,
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
async def test_subprocess_path_injects_allocated_gpu_env(
    tmp_path: Path, fake_framework_repo: Path,
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
async def test_readonly_research_scout_skips_worktree(
    tmp_path: Path, fake_framework_repo: Path,
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
    ctx.task.params.update({
        "domain": "research_scout_specialist",
        "gap_canonical_id": "gap.research_scout.round0",
        "readonly": True,
    })
    ctx.task.allowed_tools = ["Read", "Grep", "Glob", "Write"]

    result = await runner.run(ctx)

    assert result.status == "succeeded"
    workspace = session_dir / "runs" / "specialist" / "t-spec-scout"
    assert (workspace / "specialist_done.json").exists()
    assert not (workspace / "worktree").exists()


@pytest.mark.asyncio
async def test_subprocess_path_collects_patches(
    tmp_path: Path, fake_framework_repo: Path,
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
    tmp_path: Path, fake_framework_repo: Path,
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
    tmp_path: Path, fake_framework_repo: Path,
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
    heartbeat.json must NOT be reaped as stale (regression: 100% of
    specialists were killed mid-turn under gateway latency)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    process_log = workspace / "process.log"
    process_log.write_text("start\n", encoding="utf-8")
    heartbeat_file = workspace / "heartbeat.json"  # intentionally never written

    cfg = SpecialistSubprocessConfig(
        heartbeat_stale_seconds=1.0, poll_interval_seconds=0.2,
    )
    disp = SpecialistSubprocessDispatcher(config=cfg)
    proc = _FakeProc()

    async def _keep_streaming() -> None:
        # Touch process.log well past the 1.0s stale threshold, then let
        # the process "exit" cleanly so the reaper breaks on poll().
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
    await writer

    assert outcome["stale_heartbeat"] is False, outcome
    assert outcome["timed_out"] is False, outcome
    assert outcome["exit_code"] == 0


@pytest.mark.asyncio
async def test_reap_loop_kills_when_no_activity_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """With neither heartbeat.json nor process.log activity, the reaper
    still reaps a silent/hung subprocess as stale."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # No process.log, no heartbeat.json — total silence.
    heartbeat_file = workspace / "heartbeat.json"

    cfg = SpecialistSubprocessConfig(
        heartbeat_stale_seconds=0.5, poll_interval_seconds=0.2,
    )
    disp = SpecialistSubprocessDispatcher(config=cfg)
    proc = _FakeProc()  # stays alive; only staleness can stop it

    # Stub _kill so the reaper never signals a real process group.
    killed = {"v": False}

    def _fake_kill(p: Any) -> None:
        killed["v"] = True
        p.alive = False

    monkeypatch.setattr(
        SpecialistSubprocessDispatcher, "_kill", staticmethod(_fake_kill),
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
