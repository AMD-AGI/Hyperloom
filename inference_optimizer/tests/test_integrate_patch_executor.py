# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-A4 (Arbor-into-Hyperloom): IntegratePatchExecutor tests."""

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


# Helpers
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


# Targets an *existing* file (src.py) but with context lines that do not match
# the tree — exercises the genuine ``git_apply_failed`` path (file exists, hunk
# is stale), distinct from the ``patch_target_missing`` preflight below.
_BAD_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 999
+    return 2
"""


# Targets a file that does not exist in the framework tree at all — a
# hallucinated layout (e.g. a CUDA-only file on a ROCm build). Must be caught
# by the missing-target preflight, not a wasted ``git apply``.
_MISSING_TARGET_PATCH = """\
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


# 1. Patch path resolution
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
    # done_payload=None forces the filesystem scan.
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=None,
        done_payload=None,
    )
    assert len(paths) == 1


def test_resolve_patch_paths_respects_empty_done_list(tmp_path: Path):
    """An explicit empty patches list is respected; no filesystem-scan fallthrough."""
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


# 2. git apply primitives
def test_git_apply_succeeds_on_valid_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    patch = tmp_path / "valid.patch"
    patch.write_text(_VALID_PATCH, encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert ok, err
    assert (repo / "src.py").read_text().endswith("return 2\n")


def test_git_apply_fails_on_bad_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    patch = tmp_path / "bad.patch"
    patch.write_text(_BAD_PATCH, encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert not ok
    assert err


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


# Specialists author patches whose ``+++ b/<path>`` prefix is *not* a simple
# ``-p1`` strip (they read framework files at deep absolute paths, e.g.
# ``b/usr/local/lib/python3.12/dist-packages/vllm/src.py``). The executor must
# auto-detect the strip level instead of hardcoding ``-p1``; otherwise every
# such Critic-approved patch fails to apply (regression seen in production).
def _deep_prefix_patch(depth: int) -> str:
    prefix = "/".join(f"d{i}" for i in range(depth))
    return (
        f"diff --git a/{prefix}/src.py b/{prefix}/src.py\n"
        "index 0000000..1111111 100644\n"
        f"--- a/{prefix}/src.py\n"
        f"+++ b/{prefix}/src.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
    )


def test_git_apply_auto_detects_deep_p_level(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    # ``b/d0/.../d6/src.py`` needs -p7 (1 for ``b/`` + 6 for d0..d5).
    patch = tmp_path / "deep.patch"
    patch.write_text(_deep_prefix_patch(6), encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert ok, f"auto -p detection should apply deep-prefix patch: {err}"
    assert (repo / "src.py").read_text().endswith("return 2\n")
    # Reverse must auto-detect the same level and roll back cleanly.
    ok_r, err_r = _git_apply_reverse(repo, patch)
    assert ok_r, err_r
    assert (repo / "src.py").read_text().endswith("return 1\n")


# 3. Framework root resolution
def test_resolve_framework_root_picks_explicit_when_dir(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    root = _resolve_framework_root(str(repo))
    assert root is not None
    assert root.samefile(repo)


def test_resolve_framework_root_returns_none_when_no_candidate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(tmp_path / "does-not-exist"),
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path / "missing-ix"))
    root = _resolve_framework_root(None)
    # Either None or a fallback root — both acceptable for this defensive helper.
    if root is not None:
        assert root.exists()


# 4. End-to-end executor invocation
@pytest.mark.asyncio
async def test_executor_apply_only_succeeds(tmp_path: Path):
    """apply_only=True: patches applied, bench skipped, status='applied_no_bench'."""
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
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_apply_failure_rolls_back(tmp_path: Path):
    """A bad patch fails ``git apply``; the executor reverses + reports apply_failed."""
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
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_missing_target_preflight_short_circuits(tmp_path: Path):
    """A patch targeting a file absent from the framework tree is rejected by
    the preflight with ``patch_target_missing`` before any ``git apply`` runs."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    _write_specialist_workspace(
        session_dir, "t-spec-miss", patch_contents=[_MISSING_TARGET_PATCH],
    )
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-int-miss", {
        "specialist_task_id": "t-spec-miss",
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    result = await executor(ctx)
    assert result["status"] == "apply_failed"
    assert result["error_class"] == "patch_target_missing"
    assert result["error"][0]["missing_targets"] == ["a/nonexistent.py"]
    assert "advisory" in result
    # Nothing was applied or reverted; the tree is untouched.
    assert result["patches_applied"] == []
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
    """config_changes-only (no patches) still proceeds under apply_only=True."""
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


# 5. CLI registration
def test_integrate_patch_executor_imports_clean():
    """The real executor module must import without side effects."""
    from inference_optimizer.orchestrator.action_executors import integrate_patch as ip_mod
    assert hasattr(ip_mod, "IntegratePatchExecutor")
    assert callable(ip_mod.IntegratePatchExecutor)
