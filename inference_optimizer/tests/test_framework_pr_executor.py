# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""FrameworkPrExecutor — Stage 2d coverage."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.framework_pr import (
    FrameworkPrExecutor,
    _candidate_slug,
    _fetch_diff_to_path,
    _materialize_pr_diff_via_worktree,
)
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    VariantResult,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# Helpers
def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "FRAMEWORK_PR Test"
    env["GIT_AUTHOR_EMAIL"] = "fw-pr@test.local"
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


def _make_candidate(
    *, repo: str = "sgl-project/sglang", pr_number: int = 1234,
    diff_url: str = "", title: str = "test PR",
) -> dict[str, Any]:
    return {
        "repo": repo,
        "pr_number": pr_number,
        "ref": f"PR:{pr_number}",
        "title": title,
        "diff_url": diff_url,
        "summary": "",
        "score": 0.5,
    }


def _make_ctx(task_id: str, params: dict[str, Any]) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="framework_pr",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={})


# 1. Pure helpers
def test_candidate_slug_prefers_repo_and_pr_number():
    cand = _make_candidate(repo="sgl-project/sglang", pr_number=1234)
    slug = _candidate_slug(cand)
    assert slug == "sgl-project-sglang-pr-1234"


def test_candidate_slug_falls_back_to_repo_plus_ref():
    cand = {"repo": "x/y", "ref": "branch:foo", "pr_number": ""}
    slug = _candidate_slug(cand)
    assert "x-y" in slug


def test_candidate_slug_handles_missing_fields():
    assert _candidate_slug({}) == "candidate"


def test_fetch_diff_to_path_succeeds_via_file_url(tmp_path: Path):
    src = tmp_path / "src.patch"
    src.write_text(_VALID_PATCH, encoding="utf-8")
    dest = tmp_path / "out" / "got.patch"
    ok, err = _fetch_diff_to_path(
        f"file://{src}", dest, timeout_sec=5.0,
    )
    assert ok, err
    assert dest.exists()
    assert dest.read_text() == _VALID_PATCH


def test_fetch_diff_to_path_fails_on_bad_url(tmp_path: Path):
    dest = tmp_path / "missing.patch"
    ok, err = _fetch_diff_to_path(
        f"file://{tmp_path / 'does-not-exist.patch'}", dest, timeout_sec=2.0,
    )
    assert not ok
    assert err


# 2. End-to-end executor (apply_only / explicit patches)
@pytest.mark.asyncio
async def test_executor_missing_candidate_fails_cleanly(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    executor = FrameworkPrExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-fp-1", {})
    result = await executor(ctx)
    assert result["status"] == "failed"
    assert result["error_class"] == "missing_param"


@pytest.mark.asyncio
async def test_executor_no_patch_when_no_source_at_all(tmp_path: Path):
    """No diff_url, no explicit patches, no head ref to check out → genuine no_patch."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = {  # source-less candidate
        "repo": "sgl-project/sglang",
        "pr_number": "",
        "ref": "",
        "title": "no source",
        "diff_url": "",
    }
    ctx = _make_ctx("t-fp-2", {
        "candidate": cand,
        "framework_source_root": str(repo),
    })
    result = await executor(ctx)
    assert result["status"] == "no_patch"
    assert result["candidate"] == cand


@pytest.mark.asyncio
async def test_executor_no_patch_when_explicit_patches_all_missing(tmp_path: Path):
    """Regression for P2.a: missing explicit patches must short-circuit to
    no_patch, never falling back to the candidate's diff_url."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()  # carries a real diff_url
    missing = [str(tmp_path / "nope-1.patch"), str(tmp_path / "nope-2.patch")]
    ctx = _make_ctx("t-fp-no-explicit", {
        "candidate": cand,
        "patches": missing,
        "framework_source_root": str(repo),
        "batch_id": "batch-001",
        "apply_only": True,
    })
    result = await executor(ctx)

    assert result["status"] == "no_patch"
    assert result["error_class"] == "explicit_patches_missing"
    assert result["patches_applied"] == []
    assert result["missing_patches"] == missing
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_apply_only_with_explicit_patch_succeeds(tmp_path: Path):
    """apply_only=True with an explicit patch: applies, skips bench, status='applied_no_bench'."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()
    ctx = _make_ctx("t-fp-3", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "batch_id": "batch-001",
        "apply_only": True,
    })
    result = await executor(ctx)

    assert result["status"] == "applied_no_bench"
    assert result["batch_id"] == "batch-001"
    assert len(result["patches_applied"]) == 1
    assert result["patches_reverted"] == []
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_apply_failure_rolls_back(tmp_path: Path):
    """A bad patch fails git apply; executor reports apply_failed and
    leaves the source tree pristine."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "bad.patch"
    patch_path.write_text(_BAD_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()
    ctx = _make_ctx("t-fp-4", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    result = await executor(ctx)

    assert result["status"] == "apply_failed"
    assert result["error_class"] == "git_apply_failed"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_no_framework_root_returns_apply_failed(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")
    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()

    with patch(
        "inference_optimizer.orchestrator.action_executors.framework_pr."
        "_resolve_framework_root",
        return_value=None,
    ):
        ctx = _make_ctx("t-fp-5", {
            "candidate": cand,
            "patches": [str(patch_path)],
            "apply_only": True,
        })
        result = await executor(ctx)

    assert result["status"] == "apply_failed"
    assert result["error_class"] == "no_framework_root"


@pytest.mark.asyncio
async def test_executor_fetch_failure_returns_fetch_failed(tmp_path: Path):
    """A diff_url fetch failure returns ``fetch_failed`` and never touches the root."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate(
        diff_url=f"file://{tmp_path / 'missing-diff.patch'}",
    )
    ctx = _make_ctx("t-fp-6", {
        "candidate": cand,
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    result = await executor(ctx)
    assert result["status"] == "fetch_failed"
    assert result["error_class"] == "diff_fetch_failed"
    assert (repo / "src.py").read_text().endswith("return 1\n")


# 3. KEEP / REVERT decision (mocked bench)
def _mk_variant_result(*, tput: float | None, status: str = "succeeded") -> VariantResult:
    return VariantResult(
        name="framework-pr-x",
        extra_server_args="",
        extra_envs={},
        status=status,
        output_throughput=tput,
        workspace="/tmp/x",
    )


@pytest.mark.asyncio
async def test_executor_keep_when_delta_above_threshold(tmp_path: Path):
    """Bench returns +10% tput → KEEP. Patch survives in the repo."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()

    async def fake_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 1100.0},
            {"accuracy_pass": None},
        )

    ctx = _make_ctx("t-fp-keep", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=fake_bench):
        result = await executor(ctx)

    assert result["status"] == "kept"
    assert result["delta_pct"] == pytest.approx(10.0, abs=1e-6)
    assert result["output_throughput"] == 1100.0
    assert len(result["patches_applied"]) == 1
    assert result["patches_reverted"] == []
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_keep_writes_kb_lessons(tmp_path: Path, monkeypatch):
    """A KEEP appends an 'integrated' record to lessons.jsonl for dedup."""
    import inference_optimizer.orchestrator.kb_writeback as kb_writeback

    kb_root = tmp_path / "kb" / "framework_optimization"
    monkeypatch.setattr(kb_writeback, "KB_ROOT", kb_root)

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()
    cand["pr_url"] = "https://github.com/sgl-project/sglang/pull/1234"

    async def fake_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 1100.0},
            {"accuracy_pass": None},
        )

    ctx = _make_ctx("t-fp-keep-kb", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=fake_bench):
        result = await executor(ctx)

    assert result["status"] == "kept"
    lessons = kb_root / "lessons.jsonl"
    assert lessons.exists()
    records = [
        json.loads(line)
        for line in lessons.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["outcome"] == "integrated"
    assert records[0]["pr_url"] == cand["pr_url"]
    assert records[0]["tps_delta_pct"] == pytest.approx(10.0, abs=1e-6)


@pytest.mark.asyncio
async def test_executor_revert_writes_kb_lessons(tmp_path: Path, monkeypatch):
    """A REVERT appends a 'reverted_smoke_fail' record for dedup."""
    import inference_optimizer.orchestrator.kb_writeback as kb_writeback

    kb_root = tmp_path / "kb" / "framework_optimization"
    monkeypatch.setattr(kb_writeback, "KB_ROOT", kb_root)

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()
    cand["pr_url"] = "https://github.com/sgl-project/sglang/pull/1234"

    async def fake_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 980.0},
            {"accuracy_pass": None},
        )

    ctx = _make_ctx("t-fp-revert-kb", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=fake_bench):
        result = await executor(ctx)

    assert result["status"] == "reverted"
    records = [
        json.loads(line)
        for line in (kb_root / "lessons.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["outcome"] == "reverted_smoke_fail"


@pytest.mark.asyncio
async def test_executor_revert_when_delta_below_threshold(tmp_path: Path):
    """Bench returns -2% tput → REVERT. Patch is rolled back."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()

    async def fake_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 980.0},
            {"accuracy_pass": None},
        )

    ctx = _make_ctx("t-fp-revert", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=fake_bench):
        result = await executor(ctx)

    assert result["status"] == "reverted"
    assert result["delta_pct"] == pytest.approx(-2.0, abs=1e-6)
    assert result["patches_applied"] == []
    assert len(result["patches_reverted"]) == 1
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_revert_on_accuracy_regression(tmp_path: Path):
    """Bench tput is positive but accuracy gate fails → REVERT."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()

    async def fake_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 1100.0},
            {"accuracy_pass": False},
        )

    ctx = _make_ctx("t-fp-acc", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=fake_bench):
        result = await executor(ctx)

    assert result["status"] == "reverted"
    assert result["accuracy_pass"] is False
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_bench_exception_triggers_revert(tmp_path: Path):
    """Unhandled bench failure → REVERT + error_class=bench_exception."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "p.patch"
    patch_path.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()

    async def boom(self, *, params, output_root, slug):  # noqa: ARG001
        raise RuntimeError("simulated bench crash")

    ctx = _make_ctx("t-fp-boom", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=boom):
        result = await executor(ctx)

    assert result["status"] == "reverted"
    assert result["error_class"] == "bench_exception"
    assert (repo / "src.py").read_text().endswith("return 1\n")


# 3b. Serial-KEEP integrity — REJECT must not clobber prior KEPT patches
_PATCH_B_ADDS_FILE = """\
diff --git a/new.py b/new.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+def g():
+    return 99
"""


@pytest.mark.asyncio
async def test_reject_after_keep_preserves_kept_changes(tmp_path: Path):
    """Regression for P1.c: B's REVERT must reset to A's KEEP commit, not baseline."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)

    patch_a = tmp_path / "a.patch"
    patch_a.write_text(_VALID_PATCH, encoding="utf-8")
    patch_b = tmp_path / "b.patch"
    patch_b.write_text(_PATCH_B_ADDS_FILE, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)

    async def keep_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 1100.0},
            {"accuracy_pass": None},
        )

    async def reject_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 980.0},
            {"accuracy_pass": None},
        )

    ctx_a = _make_ctx("t-fp-keep-a", {
        "candidate": _make_candidate(pr_number=101, title="A"),
        "patches": [str(patch_a)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=keep_bench):
        res_a = await executor(ctx_a)
    assert res_a["status"] == "kept", res_a
    assert res_a.get("keep_commit_sha"), "KEEP must record commit sha"
    assert (repo / "src.py").read_text().endswith("return 2\n")

    ctx_b = _make_ctx("t-fp-rej-b", {
        "candidate": _make_candidate(pr_number=102, title="B"),
        "patches": [str(patch_b)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=reject_bench):
        res_b = await executor(ctx_b)
    assert res_b["status"] == "reverted", res_b

    # B's added file is gone; A's KEPT change in src.py survives.
    assert not (repo / "new.py").exists()
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_apply_failure_after_keep_preserves_kept_changes(tmp_path: Path):
    """Companion: when B's apply fails, A's KEPT commit must still survive."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)

    patch_a = tmp_path / "a.patch"
    patch_a.write_text(_VALID_PATCH, encoding="utf-8")
    bad_patch = tmp_path / "bad.patch"
    bad_patch.write_text(_BAD_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)

    async def keep_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 1100.0},
            {"accuracy_pass": None},
        )

    ctx_a = _make_ctx("t-fp-keep-a2", {
        "candidate": _make_candidate(pr_number=201, title="A"),
        "patches": [str(patch_a)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=keep_bench):
        res_a = await executor(ctx_a)
    assert res_a["status"] == "kept"

    ctx_b = _make_ctx("t-fp-bad-b2", {
        "candidate": _make_candidate(pr_number=202, title="B"),
        "patches": [str(bad_patch)],
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    res_b = await executor(ctx_b)
    assert res_b["status"] == "apply_failed"

    assert (repo / "src.py").read_text().endswith("return 2\n")


# 3c. checkout-head (diff source) mode
def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "FRAMEWORK_PR Test"
    env["GIT_AUTHOR_EMAIL"] = "fw-pr@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    return env


def _init_repo_with_pr_branch(path: Path, *, pr_ref: str = "pr-head") -> str:
    """Init ``main`` plus a divergent ``pr_ref`` branch; returns the PR head sha."""
    env = _git_env()
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)],
                   check=True, capture_output=True, env=env)
    (path / "src.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."],
                   check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"],
                   check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "checkout", "-b", pr_ref],
                   check=True, capture_output=True, env=env)
    (path / "src.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "commit", "-am", "pr head"],
                   check=True, capture_output=True, env=env)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()
    # Back to main; point origin at ourselves.
    subprocess.run(["git", "-C", str(path), "checkout", "main"],
                   check=True, capture_output=True, env=env)
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", str(path)],
        check=True, capture_output=True, env=env,
    )
    return head


def test_materialize_pr_diff_via_worktree_extracts_net_diff(tmp_path: Path):
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo, pr_ref="pr-head")
    dest = tmp_path / "out" / "pr.patch"
    cand = {"repo": "x/y", "pr_number": 7, "ref": "pr-head",
            "head_sha": head_sha}
    ok, err = _materialize_pr_diff_via_worktree(
        repo, cand, dest, timeout_sec=60.0,
    )
    assert ok, err
    text = dest.read_text()
    assert "src.py" in text
    assert "return 2" in text
    assert not (dest.parent / "wt-x-y-pr-7").exists()
    assert (repo / "src.py").read_text().endswith("return 1\n")


def test_materialize_pr_diff_empty_when_no_ref(tmp_path: Path):
    repo = tmp_path / "framework"
    _init_repo_with_pr_branch(repo)
    dest = tmp_path / "pr.patch"
    ok, err = _materialize_pr_diff_via_worktree(
        repo, {"repo": "x/y"}, dest, timeout_sec=30.0,
    )
    assert not ok
    assert "cannot resolve PR head" in err or "head" in err.lower()


@pytest.mark.asyncio
async def test_executor_checkout_head_mode_applies_and_keeps(tmp_path: Path, monkeypatch):
    """apply_mode=checkout_head extracts the PR's net diff, applies, benches (+10%), KEEPs."""
    # Redirect the KB writeback root so the KEEP record doesn't leak into the real KB.
    import inference_optimizer.orchestrator.kb_writeback as kb_writeback
    monkeypatch.setattr(
        kb_writeback, "KB_ROOT", tmp_path / "kb" / "framework_optimization"
    )

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo, pr_ref="pr-head")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = {
        "repo": "sgl-project/sglang", "pr_number": 7, "ref": "pr-head",
        "head_sha": head_sha, "title": "checkout-head PR", "diff_url": "",
        "apply_mode": "checkout_head",
    }

    async def fake_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 1100.0},
            {"accuracy_pass": None},
        )

    ctx = _make_ctx("t-fp-checkout", {
        "candidate": cand,
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=fake_bench):
        result = await executor(ctx)

    assert result["status"] == "kept", result
    assert result["patch_source_mode"] == "checkout_head"
    assert result["delta_pct"] == pytest.approx(10.0, abs=1e-6)
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_no_diff_url_falls_back_to_checkout_head(tmp_path: Path):
    """No diff_url → the executor auto-selects checkout-head rather than no_patch."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo, pr_ref="pr-head")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = {
        "repo": "sgl-project/sglang", "pr_number": 7, "ref": "pr-head",
        "head_sha": head_sha, "title": "no diff_url", "diff_url": "",
    }
    ctx = _make_ctx("t-fp-nodiffurl", {
        "candidate": cand,
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    result = await executor(ctx)
    assert result["status"] == "applied_no_bench", result
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_keep_adds_new_file_pr(tmp_path: Path):
    """P2 regression: a PR that ADDS a new file must KEEP cleanly via ``git add -A``."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    patch_path = tmp_path / "add.patch"
    patch_path.write_text(_PATCH_B_ADDS_FILE, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = _make_candidate()

    async def fake_bench(self, *, params, output_root, slug):  # noqa: ARG001
        return (
            {"status": "succeeded", "output_throughput": 1100.0},
            {"accuracy_pass": None},
        )

    ctx = _make_ctx("t-fp-addfile", {
        "candidate": cand,
        "patches": [str(patch_path)],
        "framework_source_root": str(repo),
        "base_tput": 1000.0,
        "keep_threshold_pct": 1.0,
    })
    with patch.object(FrameworkPrExecutor, "_bench_candidate", new=fake_bench):
        result = await executor(ctx)

    assert result["status"] == "kept", result
    assert result.get("keep_commit_sha")
    assert (repo / "new.py").exists()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert status == "", f"worktree not clean after KEEP: {status!r}"


@pytest.mark.asyncio
async def test_executor_cross_repo_disables_checkout_head(tmp_path: Path):
    """P1 regression: a cross-repo candidate must fall back to diff_url, not checkout-head."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    _init_git_repo(repo)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin",
         "https://github.com/sgl-project/sglang.git"],
        check=True, capture_output=True,
    )
    diff_file = tmp_path / "served.patch"
    diff_file.write_text(_VALID_PATCH, encoding="utf-8")

    executor = FrameworkPrExecutor(session_dir=session_dir)
    cand = {
        "repo": "ROCm/vllm",  # cross-repo vs the sglang live origin
        "pr_number": 42,
        "ref": "pr-head",
        "title": "cross-repo PR",
        "diff_url": f"file://{diff_file}",
        "apply_mode": "checkout_head",  # explicitly requested
    }
    ctx = _make_ctx("t-fp-crossrepo", {
        "candidate": cand,
        "framework_source_root": str(repo),
        "apply_only": True,
    })
    result = await executor(ctx)

    assert result["status"] == "applied_no_bench", result
    assert result.get("patch_source_mode") == "diff_url"
    assert (repo / "src.py").read_text().endswith("return 2\n")


# 4. Registration / import surface
def test_framework_pr_executor_imports_clean():
    from inference_optimizer.orchestrator.action_executors import (
        framework_pr as fp_mod,
    )
    assert hasattr(fp_mod, "FrameworkPrExecutor")
    assert callable(fp_mod.FrameworkPrExecutor)


def test_framework_pr_meta_loads():
    """The action_registry can load actions/_meta/framework_pr.yaml."""
    from inference_optimizer.orchestrator.action_registry import ActionRegistry
    reg = ActionRegistry().load()
    fp = reg.get("framework_pr")
    assert fp is not None
    assert fp.name == "framework_pr"
    assert fp.family == "shallow"
    assert "Bash" in fp.allowed_tools
