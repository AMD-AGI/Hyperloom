# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workspace a candidate is measured in: staged and racy edits, diffs that fail, measurements reused only for the exact candidate, and the loop's own output kept out of git."""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from dataclasses import replace

import pytest

from kernelforge.utils.git import GitError
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.tests.conftest import patch_iteration_name
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


def test_staged_candidate_is_detected_and_discarded(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)

    assert "return 2" in loop._working_tree_diff()

    loop._git_discard_worktree()

    assert loop._git("status", "--porcelain", "--untracked-files=no") == ""
    assert "return 1" in kernel.read_text()


def test_working_tree_diff_failure_is_not_treated_as_empty(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)

    def _unreadable_index(*_args, **_kwargs):
        raise GitError(128, ["git", "diff"], "", "fatal: unable to read index")

    patch_iteration_name(monkeypatch, "git", _unreadable_index)

    with pytest.raises(GitError, match="unable to read index"):
        loop._working_tree_diff()


def test_reuses_only_measurement_for_exact_candidate(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop._best_case_times = {"case": 1.0}
    loop.best_mean_case_speedup = 1.0
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    attempt_diff = loop._working_tree_diff()
    measurement = {
        "success": True,
        "measurement_count": 3,
        "measurements": [{}, {}, {}],
        "bench_repeat": loop.ic.bench_repeat,
        "candidate_diff_sha256": hashlib.sha256(attempt_diff.encode()).hexdigest(),
        "driver_sha256": loop._driver_sha256(),
        "baseline_case_times": {"case": 1.0},
        "best_mean_case_speedup": 1.0,
    }

    assert loop._can_reuse_insession_benchmark(
        measurement,
        attempt_diff=attempt_diff,
    )
    loop.ic = replace(loop.ic, commit_new_paths=["*.s"])
    assembly = workspace / "kernel.s"
    assembly.write_text("s_endpgm\n")
    assert not loop._can_reuse_insession_benchmark(measurement, attempt_diff=attempt_diff)
    assembly.unlink()
    measurement["candidate_diff_sha256"] = hashlib.sha256(b"").hexdigest()
    assert not loop._can_reuse_insession_benchmark(
        measurement,
        attempt_diff="",
    )
    measurement["candidate_diff_sha256"] = "0" * 64
    assert not loop._can_reuse_insession_benchmark(
        measurement,
        attempt_diff=attempt_diff,
    )


def test_candidate_snapshot_preserves_racy_git_index_detection(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["*.s"])
    kernel = workspace / "kernel.py"
    stamp = kernel.stat().st_mtime_ns - 2_000_000_000
    subprocess.run(["git", "config", "core.trustctime", "false"], cwd=workspace, check=True)
    os.utime(kernel, ns=(stamp, stamp))
    subprocess.run(["git", "update-index", "--refresh"], cwd=workspace, check=True)
    os.utime(workspace / ".git/index", ns=(stamp, stamp))
    kernel.write_text("def kernel():\n    return 2\n")
    os.utime(kernel, ns=(stamp, stamp))
    (workspace / "kernel.s").write_text("s_endpgm\n")
    patch, changed = loop._candidate_changes(loop._git("rev-parse", "HEAD"))
    assert set(changed) == {"kernel.py", "kernel.s"}
    assert "return 2" in patch


def test_the_loop_hides_its_output_folder_from_git(tmp_path, monkeypatch):
    """The workspace ignores nothing of its own, yet git lists nothing the loop wrote under forge_experiments."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    assert (workspace / "forge_experiments" / ".gitignore").read_text() == "*\n"
    assert any(path.name != ".gitignore" for path in (workspace / "forge_experiments").rglob("*") if path.is_file())
    listed = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "forge_experiments" not in listed.stdout
