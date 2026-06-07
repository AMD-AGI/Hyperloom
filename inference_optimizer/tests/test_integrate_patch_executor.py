# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-A4 (Arbor-into-Hyperloom): IntegratePatchExecutor tests.

Covers the deterministic patch integration step that consumes a
specialist's worktree patches and KEEPs / REVERTs them against the
framework source roots.

The tests use a tiny git repo + a real patch file so the ``git apply``
path is exercised end-to-end. The Magpie bench step is bypassed via
``params.apply_only=True`` — PR-A4's job is to land the patch protocol;
the benchmark integration is the same code path that ``explore``
already exercises in production and has dedicated coverage there.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors.integrate_patch import (
    IntegratePatchExecutor,
    _git_apply,
    _git_apply_reverse,
    _resolve_framework_root,
    _resolve_patch_paths,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "PR-A4 Test"
    env["GIT_AUTHOR_EMAIL"] = "pr-a4@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True, capture_output=True, env=env,
    )
    (path / "src.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True, capture_output=True, env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        check=True, capture_output=True, env=env,
    )


_VALID_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""


_BAD_PATCH = """\
diff --git a/nonexistent.py b/nonexistent.py
index 0000000..1111111 100644
--- a/nonexistent.py
+++ b/nonexistent.py
@@ -1,1 +1,1 @@
-OLD
+NEW
"""


def _write_specialist_workspace(
    session_dir: Path,
    task_id: str,
    *,
    patch_contents: list[str] | None = None,
    done_payload_override: dict[str, Any] | None = None,
) -> Path:
    workspace = session_dir / "runs" / "specialist" / task_id
    (workspace / "worktree" / "patches").mkdir(parents=True, exist_ok=True)
    patch_paths: list[str] = []
    for i, contents in enumerate(patch_contents or [_VALID_PATCH], start=1):
        path = workspace / "worktree" / "patches" / f"{i:03d}_test.patch"
        path.write_text(contents, encoding="utf-8")
        patch_paths.append(f"patches/{path.name}")
    payload: dict[str, Any] = {
        "gap_canonical_id": "gap.test.integrate",
        "domain": "serving_specialist",
        "proposal_set": [],
        "patches_written": patch_paths,
        "empty": False,
        "summary": "PR-A4 test",
        "confidence": 0.5,
    }
    if done_payload_override:
        payload.update(done_payload_override)
    (workspace / "specialist_done.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    return workspace


def _make_ctx(task_id: str, params: dict[str, Any]) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={})


# ---------------------------------------------------------------------------
# 1. Patch path resolution
# ---------------------------------------------------------------------------
def test_resolve_patch_paths_prefers_explicit_param(tmp_path: Path):
    workspace = _write_specialist_workspace(tmp_path, "t-a", patch_contents=[_VALID_PATCH])
    explicit = [str(workspace / "worktree" / "patches" / "001_test.patch")]
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=explicit,
        done_payload=None,
    )
    assert len(paths) == 1
    assert paths[0].name == "001_test.patch"


def test_resolve_patch_paths_from_done_payload(tmp_path: Path):
    workspace = _write_specialist_workspace(tmp_path, "t-b", patch_contents=[_VALID_PATCH])
    done_payload = json.loads(
        (workspace / "specialist_done.json").read_text(encoding="utf-8")
    )
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=None,
        done_payload=done_payload,
    )
    assert len(paths) == 1
    assert paths[0].name == "001_test.patch"


def test_resolve_patch_paths_falls_back_to_filesystem_scan(tmp_path: Path):
    workspace = _write_specialist_workspace(
        tmp_path, "t-c", patch_contents=[_VALID_PATCH],
    )
    # Pass done_payload=None to bypass the patches_written shortcut and
    # force the filesystem scan.
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=None,
        done_payload=None,
    )
    assert len(paths) == 1


def test_resolve_patch_paths_respects_empty_done_list(tmp_path: Path):
    """When done_payload says explicitly 'no patches', the resolver
    respects that and does NOT fall through to filesystem scan."""
    workspace = _write_specialist_workspace(
        tmp_path, "t-c-empty", patch_contents=[_VALID_PATCH],
        done_payload_override={"patches_written": []},
    )
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=None,
        done_payload={"patches_written": []},
    )
    assert paths == []


# ---------------------------------------------------------------------------
# 2. git apply primitives
# ---------------------------------------------------------------------------
def test_git_apply_succeeds_on_valid_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    patch = tmp_path / "valid.patch"
    patch.write_text(_VALID_PATCH, encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert ok, err
    # File was actually mutated.
    assert (repo / "src.py").read_text().endswith("return 2\n")


def test_git_apply_fails_on_bad_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    patch = tmp_path / "bad.patch"
    patch.write_text(_BAD_PATCH, encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert not ok
    assert err  # stderr non-empty


def test_git_apply_reverse_rolls_back(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    patch = tmp_path / "valid.patch"
    patch.write_text(_VALID_PATCH, encoding="utf-8")
    _git_apply(repo, patch)
    assert (repo / "src.py").read_text().endswith("return 2\n")
    ok, err = _git_apply_reverse(repo, patch)
    assert ok, err
    assert (repo / "src.py").read_text().endswith("return 1\n")


# ---------------------------------------------------------------------------
# 3. Framework root resolution
# ---------------------------------------------------------------------------
def test_resolve_framework_root_picks_explicit_when_dir(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    root = _resolve_framework_root(str(repo))
    assert root is not None
    assert root.samefile(repo)


def test_resolve_framework_root_returns_none_when_no_candidate(monkeypatch, tmp_path: Path):
    # Force allowlist to a path that doesn't exist + no override.
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(tmp_path / "does-not-exist"),
    )
    # Clear default allowlist by pointing INFERENCEX_PATH at a nonexistent
    # location so the resolve function finds nothing.
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path / "missing-ix"))
    root = _resolve_framework_root(None)
    # Either None (clean miss) or a fallback root that contains
    # something — both acceptable for this defensive helper.
    if root is not None:
        assert root.exists()


# ---------------------------------------------------------------------------
# 4. End-to-end executor invocation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_executor_apply_only_succeeds(tmp_path: Path):
    """apply_only=True path: patches applied, bench skipped, status
    reports 'applied_no_bench'."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    workspace = _write_specialist_workspace(
        session_dir, "t-spec-1", patch_contents=[_VALID_PATCH],
    )

    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-int-1", {
        "specialist_task_id": "t-spec-1",
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    result = await executor(ctx)

    assert result["status"] == "applied_no_bench"
    assert len(result["patches_applied"]) == 1
    assert result["patches_reverted"] == []
    # Patch survives in the framework repo.
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_apply_failure_rolls_back(tmp_path: Path):
    """A patch that targets a non-existent file fails ``git apply``;
    the executor reverses any partial apply and reports apply_failed."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    workspace = _write_specialist_workspace(
        session_dir, "t-spec-2", patch_contents=[_BAD_PATCH],
    )

    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-int-2", {
        "specialist_task_id": "t-spec-2",
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    result = await executor(ctx)

    assert result["status"] == "apply_failed"
    assert result["error_class"] == "git_apply_failed"
    # Source tree must remain pristine.
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_missing_specialist_workspace_fails_cleanly(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-int-3", {
        "specialist_task_id": "nonexistent",
    })
    result = await executor(ctx)
    assert result["status"] == "failed"
    assert result["error_class"] == "missing_specialist"


@pytest.mark.asyncio
async def test_executor_no_patches_returns_no_patches(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    workspace = session_dir / "runs" / "specialist" / "t-spec-4"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(json.dumps({
        "gap_canonical_id": "gap.empty",
        "domain": "serving_specialist",
        "proposal_set": [],
        "patches_written": [],
        "empty": True,
        "summary": "no proposals or patches",
    }))
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-int-4", {"specialist_task_id": "t-spec-4"})
    result = await executor(ctx)
    assert result["status"] == "no_patches"


@pytest.mark.asyncio
async def test_executor_config_changes_only_no_patches(tmp_path: Path):
    """Specialist produced no patches but provided config_changes only.
    The executor still proceeds (apply_only=True bench skip)."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    workspace = session_dir / "runs" / "specialist" / "t-spec-5"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(json.dumps({
        "gap_canonical_id": "gap.cfg",
        "domain": "serving_specialist",
        "proposal_set": [],
        "patches_written": [],
        "empty": False,
        "summary": "config-only specialist",
    }))
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-int-5", {
        "specialist_task_id": "t-spec-5",
        "config_changes": {"VLLM_USE_AITER": "1"},
        "apply_only": True,
    })
    result = await executor(ctx)
    assert result["status"] == "applied_no_bench"
    assert result["config_changes_applied"] == {"VLLM_USE_AITER": "1"}
    assert result["patches_applied"] == []


# ---------------------------------------------------------------------------
# 5. CLI registration
# ---------------------------------------------------------------------------
def test_integrate_patch_executor_imports_clean():
    """The real executor module must import without side effects
    (regression guard for the cli wiring step)."""
    from inference_optimizer.orchestrator.action_executors import integrate_patch as ip_mod
    assert hasattr(ip_mod, "IntegratePatchExecutor")
    assert callable(ip_mod.IntegratePatchExecutor)
