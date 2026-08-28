"""Unit tests for experience_integration helper gaps (git / probes / summary)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from kernelforge.knowledge import experience_integration as integ
import pytest


def _run(cmd: list[str], cwd: Path) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "t@e.com"], repo)
    _run(["git", "config", "user.name", "T"], repo)
    (repo / "kernel.py").write_text("old\n")
    _run(["git", "add", "kernel.py"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def test_git_head_returns_sha(tmp_path):
    repo = _init_repo(tmp_path)
    head = integ.git_head(str(repo))
    assert len(head) == 40


def test_git_head_empty_on_failure(tmp_path):
    assert integ.git_head(str(tmp_path / "not-a-repo")) == ""


def test_git_checkout_branch_empty_branch_noop(tmp_path):
    repo = _init_repo(tmp_path)
    assert integ.git_checkout_branch(str(repo), "") == ""


def test_git_checkout_branch_error_on_bad_workspace():
    out = integ.git_checkout_branch("/no/such/dir/xyz", "b")
    assert out != ""


def test_git_checkout_branch_reports_exception(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)

    def boom(*_a, **_k):
        raise OSError("git missing")

    monkeypatch.setattr(integ, "git", boom)
    out = integ.git_checkout_branch(str(repo), "b")
    assert out.startswith("checkout failed:")


def test_git_cumulative_diff_empty_without_base(tmp_path):
    repo = _init_repo(tmp_path)
    assert integ._git_cumulative_diff(str(repo), "") == ""


def test_git_cumulative_diff_returns_diff(tmp_path):
    repo = _init_repo(tmp_path)
    base = integ.git_head(str(repo))
    (repo / "kernel.py").write_text("new\n")
    _run(["git", "commit", "-am", "change"], repo)
    diff = integ._git_cumulative_diff(str(repo), base)
    assert "kernel.py" in diff
    assert "+new" in diff


def test_git_cumulative_diff_empty_on_exception(monkeypatch, tmp_path):
    def boom(*_a, **_k):
        raise OSError("git missing")

    monkeypatch.setattr(integ, "git", boom)
    assert integ._git_cumulative_diff(str(tmp_path), "base") == ""


def test_git_apply_check_and_exception(tmp_path):
    repo = _init_repo(tmp_path)
    good = "diff --git a/kernel.py b/kernel.py\n--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n"
    assert integ._git_apply(str(repo), good, check_only=True) is True
    # A workspace that is not there is not a patch that does not apply.
    with pytest.raises(OSError):
        integ._git_apply("/no/such/dir/xyz", good)


def test_git_commit_all_and_discard(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "kernel.py").write_text("changed\n")
    integ._git_commit_all(str(repo), "msg", allowed_paths={"kernel.py"})
    assert _run(["git", "log", "-1", "--pretty=%s"], repo) == "msg"

    (repo / "kernel.py").write_text("dirty\n")
    assert integ._git_discard_worktree(str(repo)) is True
    assert (repo / "kernel.py").read_text() == "changed\n"


def test_git_discard_removes_symlink_without_touching_target(tmp_path):
    repo = _init_repo(tmp_path)
    kernel = repo / "kernel.py"
    link = repo / "warm-link.py"
    before = integ._untracked_files(str(repo))
    link.symlink_to(kernel.name)

    assert integ._git_discard_worktree(str(repo), before) is True
    assert not link.exists()
    assert not link.is_symlink()
    assert kernel.read_text() == "old\n"


# --------------------------------------------------------------------------- #
# _bench_once / _correctness_once
# --------------------------------------------------------------------------- #
def test_bench_once_returns_complete_suite(monkeypatch):
    import kernelforge.mcp_server.tools.bench as bench

    async def fake(driver_script, driver_args):
        assert driver_args == []
        return {
            "success": True,
            "median_ms": 7.5,
            "case_times": {"case-1": 7.5},
        }

    monkeypatch.setattr(bench, "bench_wallclock", fake)
    assert integ._bench_once("drv.py") == {
        "success": True,
        "median_ms": 7.5,
        "case_times": {"case-1": 7.5},
    }


def test_bench_once_rejects_scalar_only_result(monkeypatch):
    import kernelforge.mcp_server.tools.bench as bench

    async def fake(driver_script, driver_args):
        return {"success": True, "median_ms": 7.5}

    monkeypatch.setattr(bench, "bench_wallclock", fake)
    assert integ._bench_once("drv.py") is None


def test_bench_once_none_on_exception(monkeypatch):
    import kernelforge.mcp_server.tools.bench as bench

    async def boom(**_k):
        raise RuntimeError("bench failed")

    monkeypatch.setattr(bench, "bench_wallclock", boom)
    assert integ._bench_once("drv.py") is None


def test_correctness_once_true(monkeypatch):
    import kernelforge.mcp_server.tools.test as test_mod

    async def fake(driver_script, driver_args, snr_threshold):
        assert driver_args == []
        return {"passed": True}

    monkeypatch.setattr(test_mod, "test_correctness", fake)
    assert integ._correctness_once("drv.py", 30.0) is True


def test_correctness_once_false_on_exception(monkeypatch):
    import kernelforge.mcp_server.tools.test as test_mod

    async def boom(**_k):
        raise RuntimeError("correctness failed")

    monkeypatch.setattr(test_mod, "test_correctness", boom)
    assert integ._correctness_once("drv.py", 30.0) is False


# --------------------------------------------------------------------------- #
# _cheap_summary
# --------------------------------------------------------------------------- #
class _Archive:
    def __init__(self, index):
        self._index = index

    def load_index(self):
        return self._index


def test_cheap_summary_none_archive():
    assert integ._cheap_summary(None) == {
        "category": "",
        "strategy": "",
        "recipe": "",
        "lessons": "",
    }


def test_cheap_summary_picks_best_mean_case_speedup_without_distilling_records():
    archive = _Archive(
        [
            {
                "decision": "KEEP",
                "wall_ms": 9.0,
                "mean_case_speedup": 3.0,
                "plan": "best mean plan",
            },
            {
                "decision": "KEEP",
                "wall_ms": 3.0,
                "mean_case_speedup": 2.0,
                "plan": "fast raw plan",
            },
            {"decision": "REVERT", "wall_ms": 1.0, "plan": "ignored"},
        ]
    )
    out = integ._cheap_summary(archive)
    assert out["strategy"] == "best mean plan"
    assert out["lessons"] == ""


def test_cheap_summary_survives_broken_archive():
    class _Bad:
        def load_index(self):
            raise RuntimeError("boom")

    out = integ._cheap_summary(_Bad())
    assert out == {"category": "", "strategy": "", "recipe": "", "lessons": ""}


# --------------------------------------------------------------------------- #
# kb_warmstart error path
# --------------------------------------------------------------------------- #
def test_kb_warmstart_reference_only_when_patch_empty(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)

    def fake_read(**_k):
        return [
            {
                "solution_slug": "s/prev",
                "speedup": 1.5,
                "patch_content": "",
                "strategy": "st",
                "recipe": "",
                "lessons": "",
                "match_mode": "exact",
            }
        ]

    monkeypatch.setattr("kernelforge.knowledge.experience_reader.read_top_solutions", fake_read)
    monkeypatch.setattr(
        integ,
        "_bench_once",
        lambda *_a, **_k: {
            "success": True,
            "median_ms": 10.0,
            "case_times": {"case": 10.0},
        },
    )

    warm = integ.kb_warmstart(
        config=object(), kernel=str(repo / "kernel.py"), driver="d.py", workspace_dir=str(repo), kernel_backend="triton"
    )
    assert warm["candidate"] is True
    assert warm["applied"] is False
    assert warm["keep_baseline_ms"] == 10.0


def test_write_experience_to_kb_does_not_synthesize_speedup(monkeypatch, tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n")
    captured = {}

    def reject_missing_speedup(**kwargs):
        captured.update(kwargs)
        return {"written": False, "reason": "missing_mean_case_speedup"}

    monkeypatch.setattr(
        "kernelforge.knowledge.experience_sink.write_run_experience",
        reject_missing_speedup,
    )
    monkeypatch.setattr(integ, "_git_cumulative_diff", lambda _w, _b: "diff")

    class _LR:
        experiment = type("E", (), {"experiment_id": "e"})()
        ic = type("IC", (), {"baseline_wall_ms": 8.0})()
        best_wall_ms = 4.0
        archive = None

    status = integ.write_experience_to_kb(
        config=object(),
        loop_runner=_LR(),
        workspace_dir=str(tmp_path),
        kernel=str(kernel),
        kernel_backend="triton",
        gpu_target="gfx942",
        base_sha="base",
    )
    assert status == {
        "written": False,
        "reason": "missing_mean_case_speedup",
    }
    assert captured["mean_case_speedup"] is None


def test_kb_warmstart_swallows_reader_error(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)

    def boom(**_k):
        raise RuntimeError("read blew up")

    monkeypatch.setattr("kernelforge.knowledge.experience_reader.read_top_solutions", boom)
    warm = integ.kb_warmstart(
        config=object(), kernel=str(repo / "kernel.py"), driver="d.py", workspace_dir=str(repo), kernel_backend="triton"
    )
    assert warm == {
        "candidate": False,
        "read_reason": "warm_start_error",
        "read_error": "RuntimeError: read blew up",
    }
