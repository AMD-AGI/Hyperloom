# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess

import pytest

from hyperloom.measure.runner.tools import ToolError, ensure_inferencex


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


def test_an_unknown_commit_is_refused(upstream, tmp_path, monkeypatch):
    repo, _ = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    with pytest.raises(ToolError, match="git fetch failed"):
        ensure_inferencex("a" * 40, tmp_path / "cache")
