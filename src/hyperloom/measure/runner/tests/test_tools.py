# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import errno
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hyperloom.measure.runner import tools
from hyperloom.measure.runner.tools import StagedSuite, ToolError, ensure_inference_suite, ensure_inferencex
from hyperloom.measure.tests.bundle_factory import STAGING, STAGING_MANIFEST, stage_into
from inference_suite import CLIENT_FILE, INFERENCEX_CLIENT_COMMIT


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "up"
    (repo / "benchmarks").mkdir(parents=True)
    (repo / "benchmarks" / "benchmark_lib.sh").write_text("# lib\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_the_commit_is_fetched_into_its_own_directory_and_reused(upstream, tmp_path, monkeypatch):
    repo, commit = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    first = ensure_inferencex(commit, tmp_path / "cache")
    (first / "marker").write_text("x")
    second = ensure_inferencex(commit, tmp_path / "cache")

    assert first == second == tmp_path / "cache" / f"InferenceX-{commit}"
    assert _git(first, "rev-parse", "HEAD") == commit
    assert (second / "marker").exists()  # reused, not fetched again


def test_a_hash_that_is_not_a_full_commit_is_refused(tmp_path):
    with pytest.raises(ToolError, match="not a full commit hash"):
        ensure_inferencex("main", tmp_path)


def test_a_tree_without_the_expected_files_is_refused_and_leaves_nothing(upstream, tmp_path, monkeypatch):
    repo, _ = upstream
    (repo / "benchmarks" / "benchmark_lib.sh").unlink()
    (repo / "other").write_text("x")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "two")
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    with pytest.raises(ToolError, match="not the expected tree"):
        ensure_inferencex(_git(repo, "rev-parse", "HEAD"), tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_a_tree_that_keeps_its_benchmark_library_below_inferencex_e2e_is_accepted(tmp_path, monkeypatch):
    repo = tmp_path / "up"
    (repo / "inferencex-e2e" / "benchmarks").mkdir(parents=True)
    (repo / "inferencex-e2e" / "benchmarks" / "benchmark_lib.sh").write_text("# lib\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    tree = ensure_inferencex(commit, tmp_path / "cache")

    assert (tree / "inferencex-e2e" / "benchmarks" / "benchmark_lib.sh").is_file()
    assert ensure_inferencex(commit, tmp_path / "cache") == tree


def test_a_tree_of_the_layout_inference_suite_stages_from_is_accepted(tmp_path, monkeypatch):
    client = tmp_path / "up" / "inferencex-e2e" / CLIENT_FILE
    client.parent.mkdir(parents=True)
    client.write_text("# client\n")
    repo = tmp_path / "up"
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    tree = ensure_inferencex(_git(repo, "rev-parse", "HEAD"), tmp_path / "cache")

    assert (tree / "inferencex-e2e" / CLIENT_FILE).is_file()


def test_an_unknown_commit_is_refused(upstream, tmp_path, monkeypatch):
    repo, _ = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    with pytest.raises(ToolError, match="git fetch failed"):
        ensure_inferencex("a" * 40, tmp_path / "cache")


class _Stage:
    """``python -m inference_suite stage`` as the runner calls it: lays a verifying staging into the tool cache it is
    given, or exits ``returncode`` without one."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode, self.argv = returncode, []

    def __call__(self, argv, **kwargs):
        self.argv.append(argv)
        if self.returncode == 0:
            cache = Path(argv[argv.index("--tool-cache") + 1])
            stage_into(cache.parents[1])
        return subprocess.CompletedProcess(argv, self.returncode, STAGING + "\n", "staging_missing: no wheel")


def _ensure(tmp_path):
    return ensure_inference_suite(tmp_path / "ix", tmp_path / "runner-cache", tmp_path / "wheels")


def _destination(tmp_path):
    return tmp_path / "runner-cache" / "inference-suite" / INFERENCEX_CLIENT_COMMIT


def test_a_staging_the_runner_cache_holds_intact_is_used_without_staging_again(tmp_path, monkeypatch):
    held = stage_into(tmp_path / "runner-cache")
    stage = _Stage()
    monkeypatch.setattr(tools.subprocess, "run", stage)

    assert _ensure(tmp_path) == StagedSuite(held, STAGING) and held == _destination(tmp_path)
    assert stage.argv == []


def test_a_missing_staging_is_staged_offline_from_the_checkout_into_the_directory_of_the_pin(tmp_path, monkeypatch):
    stage = _Stage()
    monkeypatch.setattr(tools.subprocess, "run", stage)

    staged = _ensure(tmp_path)

    destination = _destination(tmp_path)
    assert staged == StagedSuite(destination, STAGING)
    assert stage.argv == [
        [
            sys.executable,
            "-m",
            "inference_suite",
            "stage",
            "--source",
            str(tmp_path / "ix"),
            "--tool-cache",
            str(destination),
            "--wheels",
            str(tmp_path / "wheels"),
        ]
    ]


def test_a_damaged_staging_is_handed_to_stage_which_decides_and_the_runner_reports_its_own_manifest(
    tmp_path, monkeypatch
):
    held = stage_into(tmp_path / "runner-cache")
    (held / "staging_manifest.json").write_bytes(STAGING_MANIFEST.replace(b"inferencex", b"other"))
    stage = _Stage()
    monkeypatch.setattr(tools.subprocess, "run", stage)

    assert _ensure(tmp_path) == StagedSuite(held, STAGING) and len(stage.argv) == 1


def test_a_failed_stage_is_a_tool_error_naming_its_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", _Stage(returncode=2))

    with pytest.raises(ToolError, match="exited 2.*staging_missing: no wheel"):
        _ensure(tmp_path)


def test_two_runners_on_a_new_staging_stage_it_once_and_both_get_the_whole_tree(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    stage = _Stage()

    def slow_stage(argv, **kwargs):
        entered.set()
        assert release.wait(10)
        return stage(argv, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", slow_stage)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(_ensure, tmp_path)
        assert entered.wait(10)
        second = pool.submit(_ensure, tmp_path)
        time.sleep(0.2)
        assert not second.done()
        release.set()
        staged = {first.result(10), second.result(10)}

    assert staged == {StagedSuite(_destination(tmp_path), STAGING)} and len(stage.argv) == 1


def test_a_runner_cache_that_cannot_be_written_is_a_tool_error(tmp_path, monkeypatch):
    def read_only(self, *args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system", str(self))

    monkeypatch.setattr(tools.subprocess, "run", _Stage())
    monkeypatch.setattr(Path, "mkdir", read_only)

    with pytest.raises(ToolError, match="cannot be staged into"):
        _ensure(tmp_path)
