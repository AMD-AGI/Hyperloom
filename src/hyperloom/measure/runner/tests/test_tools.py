# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import errno
import hashlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hyperloom.measure.runner import tools
from hyperloom.measure.runner.tools import ToolError, ensure_inference_suite, ensure_inferencex
from inference_suite import CLIENT_FILE


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


_MANIFEST = b'{"schema": "inference-suite-staging/1", "tools": {}}\n'
_STAGING = hashlib.sha256(_MANIFEST).hexdigest()


class _Stage:
    """``python -m inference_suite stage`` as the runner calls it: writes ``manifest`` into the tool cache it is given."""

    def __init__(self, manifest: bytes = _MANIFEST, returncode: int = 0) -> None:
        self.manifest, self.returncode, self.argv = manifest, returncode, []

    def __call__(self, argv, **kwargs):
        self.argv.append(argv)
        cache = argv[argv.index("--tool-cache") + 1]
        with open(f"{cache}/staging_manifest.json", "wb") as manifest:
            manifest.write(self.manifest)
        return subprocess.CompletedProcess(argv, self.returncode, "", "stage refused")


def test_a_staging_the_runner_cache_holds_is_used_without_staging_again(tmp_path, monkeypatch):
    held = tmp_path / "runner-cache" / "inference-suite" / _STAGING
    held.mkdir(parents=True)
    (held / "staging_manifest.json").write_bytes(_MANIFEST)
    stage = _Stage()
    monkeypatch.setattr(tools.subprocess, "run", stage)

    assert ensure_inference_suite(_STAGING, tmp_path / "ix", tmp_path / "runner-cache") == held
    assert stage.argv == []


def test_a_missing_staging_is_staged_from_the_inferencex_checkout_into_its_own_directory(tmp_path, monkeypatch):
    stage = _Stage()
    monkeypatch.setattr(tools.subprocess, "run", stage)

    staged = ensure_inference_suite(_STAGING, tmp_path / "ix", tmp_path / "runner-cache")

    assert staged == tmp_path / "runner-cache" / "inference-suite" / _STAGING
    assert stage.argv == [
        [
            sys.executable,
            "-m",
            "inference_suite",
            "stage",
            "--source",
            str(tmp_path / "ix"),
            "--tool-cache",
            str(staged),
        ]
    ]


@pytest.mark.parametrize("stage", [_Stage(manifest=b"{}\n"), _Stage(returncode=2)], ids=["other pins", "stage failed"])
def test_a_staging_that_is_not_the_sheets_is_removed_and_refused(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(tools.subprocess, "run", stage)

    with pytest.raises(ToolError, match=f"not the sheet's {_STAGING}"):
        ensure_inference_suite(_STAGING, tmp_path / "ix", tmp_path / "runner-cache")
    assert not (tmp_path / "runner-cache" / "inference-suite" / _STAGING).exists()


def test_a_staging_that_is_not_a_manifest_sha256_is_refused_before_anything_is_staged(tmp_path):
    with pytest.raises(ToolError, match="not a staging manifest sha256"):
        ensure_inference_suite("sha256:" + _STAGING, tmp_path / "ix", tmp_path / "runner-cache")
    assert not (tmp_path / "runner-cache").exists()


def test_two_runners_on_a_new_staging_stage_it_once_and_both_get_the_whole_tree(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    stage = _Stage()

    def slow_stage(argv, **kwargs):
        entered.set()
        assert release.wait(10)
        return stage(argv, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", slow_stage)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(ensure_inference_suite, _STAGING, tmp_path / "ix", tmp_path / "runner-cache")
        assert entered.wait(10)
        second = pool.submit(ensure_inference_suite, _STAGING, tmp_path / "ix", tmp_path / "runner-cache")
        time.sleep(0.2)
        assert not second.done()
        release.set()
        staged = {first.result(10), second.result(10)}

    assert staged == {tmp_path / "runner-cache" / "inference-suite" / _STAGING} and len(stage.argv) == 1
    assert (
        tmp_path / "runner-cache" / "inference-suite" / _STAGING / "staging_manifest.json"
    ).read_bytes() == _MANIFEST


def test_a_runner_cache_that_cannot_be_written_is_a_tool_error(tmp_path, monkeypatch):
    def read_only(self, *args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system", str(self))

    monkeypatch.setattr(tools.subprocess, "run", _Stage())
    monkeypatch.setattr(Path, "mkdir", read_only)

    with pytest.raises(ToolError, match="cannot be staged into"):
        ensure_inference_suite(_STAGING, tmp_path / "ix", tmp_path / "runner-cache")
