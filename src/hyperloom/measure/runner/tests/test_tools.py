# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest

from hyperloom.measure.runner import tools
from hyperloom.measure.runner.tools import (
    StagedSuite,
    StagingMismatch,
    ToolError,
    ensure_inference_suite,
    ensure_inferencex,
)
from hyperloom.measure.tests.bundle_factory import STAGING, STAGING_MANIFEST, lm_eval_at_its_pin, stage_into
from inference_suite import CLIENT_FILE, StagingError, staging_key
from inference_suite.contracts import PackageIdentity


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


@pytest.fixture
def identified(monkeypatch):
    """This interpreter's normalised manifest of a staging is :data:`STAGING` (lm-eval is not installed in CI)."""
    seen = []

    def identify(tool_cache):
        seen.append(tool_cache)
        return STAGING

    monkeypatch.setattr(tools, "normalised_manifest", identify)
    return seen


def _ensure(tmp_path, expected=STAGING):
    return ensure_inference_suite(tmp_path / "runner-cache", expected)


def test_the_published_staging_of_the_runners_pins_is_used_when_it_is_the_sheets(tmp_path, monkeypatch):
    held = stage_into(tmp_path / "runner-cache")
    monkeypatch.syspath_prepend(str(lm_eval_at_its_pin(tmp_path / "site")))

    assert _ensure(tmp_path) == StagedSuite(held, STAGING)
    assert held == tmp_path / "runner-cache" / "inference-suite" / staging_key()


def test_a_staging_that_is_other_software_than_the_sheets_is_refused_naming_each_package(tmp_path, identified):
    stage_into(tmp_path / "runner-cache")
    theirs = replace(STAGING, packages=(PackageIdentity("agentx-harness", "1.0.12", "d" * 64), *STAGING.packages[1:]))

    with pytest.raises(StagingMismatch, match=r"agentx-harness 1\.0\.13 .* here, 1\.0\.12 .* there") as refused:
        _ensure(tmp_path, theirs)
    assert "lm-eval" not in str(refused.value)


@pytest.mark.parametrize("held", ["none", "private", "damaged"])
def test_no_published_staging_that_verifies_is_a_tool_error_and_nothing_is_staged_here(tmp_path, identified, held):
    runner_cache = tmp_path / "runner-cache"
    if held != "none":
        published = stage_into(runner_cache)
    if held == "private":
        published.rename(published.with_name(f".staging-{staging_key()}-another-host"))
    if held == "damaged":
        (published / "staging_manifest.json").write_bytes(STAGING_MANIFEST.replace(b"inferencex", b"other"))

    with pytest.raises(ToolError, match="no published staging .* that verifies; cache_build publishes it"):
        _ensure(tmp_path)
    assert identified == []
    assert not (runner_cache / "inference-suite" / staging_key()).exists() or held == "damaged"


def test_a_staging_this_interpreter_cannot_identify_is_a_tool_error(tmp_path, monkeypatch):
    stage_into(tmp_path / "runner-cache")

    def unidentified(tool_cache):
        raise StagingError("lm_eval None, not lm_eval[api]==0.4.13")

    monkeypatch.setattr(tools, "normalised_manifest", unidentified)

    with pytest.raises(ToolError, match="cannot be identified: lm_eval None"):
        _ensure(tmp_path)
