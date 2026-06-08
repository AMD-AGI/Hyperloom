"""PR-A2 (Arbor-into-Hyperloom): SpecialistRunner subprocess + worktree.

These tests pin the production specialist dispatch shape:

* SpecialistRunner accepts a :class:`SpecialistSubprocessConfig` and routes
  every task through :class:`SpecialistSubprocessDispatcher`.
* The dispatcher provisions a per-task git worktree under
  ``runs/specialist/<task_id>/worktree/``, branched off the configured
  ``framework_source_roots`` HEAD.
* The dispatcher spawns ``claude --print --add-dir <worktree>
  --add-dir <workspace>`` and waits for ``specialist_done.json`` to
  appear OR the subprocess to exit.
* The runner harvests done.json + patches/, threads them into the
  shared finalize path, and returns a SpecialistRunResult whose
  ``specialist_done`` dict carries the patch list under
  ``patches_written``.
* Edit / Write / MultiEdit are no longer in the deny list (PR-A2);
  agents may author patches inside the worktree.

We use a fake ``claude`` binary (a tiny shell script written into
``tmp_path``) so the test is hermetic and fast — no real LLM call.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
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
    _pick_worktree_base,
    _setup_worktree,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _init_git_repo(path: Path) -> None:
    """Initialise a minimal git repo at ``path`` with one commit so
    ``git worktree add`` can branch off it."""
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
    """Write a fake ``claude`` executable that simulates one of:

    * ``done_only``: writes a ``specialist_done.json`` into the workspace
      (whatever directory is passed via ``--add-dir``) and exits 0.
    * ``done_with_patch``: writes done.json AND a patch under
      ``<worktree>/patches/001_test.patch`` referenced in done.patches_written.
    * ``crash``: exits non-zero without writing anything.
    """
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
    # Always parse args to find --add-dir paths so we know where to
    # write the done file. The first --add-dir is the worktree
    # (matches the dispatcher's order); the second is the workspace.
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
WORKSPACE="${ADD_DIRS[1]:-${ADD_DIRS[0]:-}}"
"""
    if behavior == "done_only":
        body += f"""
cat > "$WORKSPACE/specialist_done.json" <<'EOF'
{payload_json}
EOF
exit 0
"""
    elif behavior == "done_with_patch":
        # Build a payload that lists the patch.
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


# ---------------------------------------------------------------------------
# 1. Constructor invariants
# ---------------------------------------------------------------------------
def test_runner_requires_exactly_one_dispatch_mode():
    with pytest.raises(ValueError, match="exactly one"):
        SpecialistRunner()  # neither set
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


# ---------------------------------------------------------------------------
# 2. Tool whitelist updates
# ---------------------------------------------------------------------------
def test_default_tools_include_write_capabilities():
    """PR-A2 lifted Edit/Write/MultiEdit out of the denylist so
    specialists can author patches inside their worktree."""
    for tool in ("Edit", "Write", "MultiEdit"):
        assert tool in DEFAULT_SPECIALIST_TOOLS
        assert tool not in SPECIALIST_TOOL_DENYLIST


def test_kb_write_tools_remain_denied():
    """KB lifecycle stays Coordinator-owned — Inv-2 / Inv-6.1 cannot
    be relaxed by PR-A2."""
    for kb_tool in (
        "mcp__cortex_kb__propose_point",
    ):
        assert kb_tool in SPECIALIST_TOOL_DENYLIST


# ---------------------------------------------------------------------------
# 3. Worktree helpers
# ---------------------------------------------------------------------------
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
    # Branch should exist in the base repo.
    cp = subprocess.run(
        ["git", "-C", str(fake_framework_repo), "branch", "--list",
         "specialist-test1"],
        capture_output=True, text=True, check=True,
    )
    assert "specialist-test1" in cp.stdout


# ---------------------------------------------------------------------------
# 4. End-to-end subprocess dispatch with the fake `claude` binary
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_subprocess_path_harvests_done_file(
    tmp_path: Path, fake_framework_repo: Path,
):
    """Spawning the fake ``claude`` writes specialist_done.json into the
    workspace; the runner reads it, threads it through finalize, and
    returns a SpecialistRunResult with status=succeeded."""
    bin_dir = tmp_path / "bin"
    fake_claude = _make_fake_claude(bin_dir, behavior="done_only")
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    config = SpecialistSubprocessConfig(
        claude_executable=str(fake_claude),
        model="",  # empty → don't pass --model
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
    # Workspace + done.json + worktree exist on disk
    workspace = session_dir / "runs" / "specialist" / "t-spec-done"
    assert (workspace / "specialist_done.json").exists()
    assert (workspace / "process.log").exists()
    assert (workspace / "worktree").is_dir()


@pytest.mark.asyncio
async def test_subprocess_path_collects_patches(
    tmp_path: Path, fake_framework_repo: Path,
):
    """The agent writes both a done file AND a patch under
    worktree/patches/; the runner threads the patch path into
    specialist_done['patches_written']."""
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
    # Patch file is on disk inside the worktree.
    worktree = session_dir / "runs" / "specialist" / "t-spec-patch" / "worktree"
    assert (worktree / "patches" / "001_test.patch").exists()


@pytest.mark.asyncio
async def test_subprocess_crash_falls_back_to_empty_synthesised(
    tmp_path: Path, fake_framework_repo: Path,
):
    """When the fake binary exits non-zero without writing done.json,
    the runner synthesises an empty specialist_done and marks the run
    ``stale``-like (status=stale because backend_error set)."""
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
    # Crash with no done.json → empty_synthesised (because the
    # subprocess_error path includes exit_code:3).
    assert result.status in ("empty_synthesised", "stale")
    assert result.specialist_done["empty"] is True
    assert "subprocess" in (result.error or "")


@pytest.mark.asyncio
async def test_subprocess_path_isolates_writes_to_worktree(
    tmp_path: Path, fake_framework_repo: Path,
):
    """The worktree is a separate checkout — patches written there must
    NOT show up in the base repo's working tree until ``integrate_patch``
    explicitly applies them."""
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

    # The patch + dummy.txt live ONLY in the worktree, not in the base
    # repo's working tree.
    worktree = session_dir / "runs" / "specialist" / "t-spec-iso" / "worktree"
    assert (worktree / "patches" / "001_test.patch").exists()
    assert not (fake_framework_repo / "patches" / "001_test.patch").exists()
    assert not (fake_framework_repo / "dummy.txt").exists()
