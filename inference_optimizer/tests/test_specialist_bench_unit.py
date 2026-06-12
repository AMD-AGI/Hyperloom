# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the specialist micro-bench surface + worktree git helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import specialist_bench as sb


def _git(repo: Path, *args: str) -> None:
    """Run a git command in repo, raising on failure."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _init_repo(repo: Path) -> None:
    """Initialize a git repo with one committed file f.txt='a\\n'."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")


# ---- envelopes ----

def test_error_and_ok():
    e = sb._error("bad", x=1)
    assert e == {"ok": False, "reason": "bad", "x": 1}
    assert sb._ok() == {"ok": True}
    assert sb._ok({"a": 2}) == {"ok": True, "a": 2}


# ---- run_bench ----

async def test_run_bench_disabled(monkeypatch):
    monkeypatch.setattr(sb, "BENCH_TOOL_ENABLED", False)
    out = await sb.run_bench("kernel_gemm_timing", worktree=Path("/tmp"), call_id="c1")
    assert out["reason"] == "bench_tool_disabled"


async def test_run_bench_unknown_id(tmp_path):
    out = await sb.run_bench("nope", worktree=tmp_path, call_id="c1")
    assert out["reason"] == "unknown_bench_id"
    assert "kernel_gemm_timing" in out["allowed"]


async def test_run_bench_script_missing(tmp_path):
    out = await sb.run_bench(
        "kernel_gemm_timing", worktree=tmp_path, call_id="c1",
        bench_dir_root=tmp_path / "empty",
    )
    assert out["reason"] == "bench_script_missing"


async def test_run_bench_success(tmp_path):
    bench_root = tmp_path / "benches"
    bench_root.mkdir()
    (bench_root / "kernel_gemm_timing.sh").write_text(
        "#!/usr/bin/env bash\necho hello-bench\n", encoding="utf-8",
    )
    out = await sb.run_bench(
        "kernel_gemm_timing", worktree=tmp_path, call_id="c1",
        bench_dir_root=bench_root, params={"k": "v"},
    )
    assert out["ok"] is True
    assert out["exit_code"] == 0
    assert "hello-bench" in out["stdout_tail"]
    assert Path(out["output_dir"]).is_dir()


# ---- apply_patch_in_worktree ----

def test_apply_patch_empty():
    assert sb.apply_patch_in_worktree(Path("/tmp"), "")["reason"] == "empty_patch"


def test_apply_patch_worktree_missing(tmp_path):
    out = sb.apply_patch_in_worktree(tmp_path / "nope", "diff --git a/x b/x\n")
    assert out["reason"] == "worktree_missing"


def test_apply_patch_path_escape(tmp_path):
    _init_repo(tmp_path)
    patch = "--- a/../escape\n+++ b/../escape\n"
    out = sb.apply_patch_in_worktree(tmp_path, patch)
    assert out["reason"] == "patch_path_escapes_worktree"


def test_apply_patch_success(tmp_path):
    _init_repo(tmp_path)
    # Build a real patch via git, then revert so apply re-applies it.
    (tmp_path / "f.txt").write_text("b\n", encoding="utf-8")
    diff = subprocess.run(
        ["git", "-C", str(tmp_path), "diff"],
        capture_output=True, text=True, check=True,
    ).stdout
    _git(tmp_path, "checkout", "--", "f.txt")
    out = sb.apply_patch_in_worktree(tmp_path, diff)
    assert out["ok"] is True
    assert (tmp_path / "f.txt").read_text() == "b\n"


def test_apply_patch_rejected(tmp_path):
    _init_repo(tmp_path)
    bad = (
        "diff --git a/f.txt b/f.txt\n"
        "--- a/f.txt\n+++ b/f.txt\n"
        "@@ -1 +1 @@\n-does-not-match\n+new\n"
    )
    out = sb.apply_patch_in_worktree(tmp_path, bad)
    assert out["reason"] == "git_apply_rejected"


# ---- capture_worktree_cumulative_diff ----

def test_capture_diff_not_a_dir(tmp_path):
    assert sb.capture_worktree_cumulative_diff(tmp_path / "nope") is None


def test_capture_diff_clean_repo(tmp_path):
    _init_repo(tmp_path)
    assert sb.capture_worktree_cumulative_diff(tmp_path) == ""


def test_capture_diff_dirty_repo(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed\n", encoding="utf-8")
    diff = sb.capture_worktree_cumulative_diff(tmp_path)
    assert diff and "changed" in diff


def test_capture_diff_non_repo(tmp_path):
    # An existing dir that is not a git repo -> git fails -> None.
    assert sb.capture_worktree_cumulative_diff(tmp_path) is None


# ---- reset_worktree ----

def test_reset_worktree_not_a_dir(tmp_path):
    # Should not raise.
    sb.reset_worktree(tmp_path / "nope")


def test_reset_worktree_restores(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("dirty\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("x\n", encoding="utf-8")
    sb.reset_worktree(tmp_path)
    assert (tmp_path / "f.txt").read_text() == "a\n"
    assert not (tmp_path / "untracked.txt").exists()
