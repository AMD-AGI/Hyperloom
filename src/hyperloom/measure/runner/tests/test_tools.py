# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import replace
from pathlib import Path

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


#: The git the tests set their repositories up with: the one on ``PATH`` at import, never the ownership-checking fake
#: (:data:`_OWNERSHIP_CHECKING_GIT`), which only the code under test runs.
_SETUP_GIT = shutil.which("git") or "git"


def _git(path, *args):
    return subprocess.run(
        [_SETUP_GIT, "-C", str(path), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


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

    first = ensure_inferencex(commit, tmp_path / "cache", tmp_path / "store")
    (first / "marker").write_text("x")
    second = ensure_inferencex(commit, tmp_path / "cache", tmp_path / "store")

    assert first == second == tmp_path / "cache" / f"InferenceX-{commit}"
    assert _git(first, "rev-parse", "HEAD") == commit
    assert (second / "marker").exists()  # reused, not fetched again


def test_a_hash_that_is_not_a_full_commit_is_refused(tmp_path):
    with pytest.raises(ToolError, match="not a full commit hash"):
        ensure_inferencex("main", tmp_path, tmp_path / "store")


def test_a_tree_without_the_expected_files_is_refused_and_leaves_nothing(upstream, tmp_path, monkeypatch):
    repo, _ = upstream
    (repo / "benchmarks" / "benchmark_lib.sh").unlink()
    (repo / "other").write_text("x")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "two")
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    with pytest.raises(ToolError, match="not the expected tree"):
        ensure_inferencex(_git(repo, "rev-parse", "HEAD"), tmp_path / "cache", tmp_path / "store")
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

    tree = ensure_inferencex(commit, tmp_path / "cache", tmp_path / "store")

    assert (tree / "inferencex-e2e" / "benchmarks" / "benchmark_lib.sh").is_file()
    assert ensure_inferencex(commit, tmp_path / "cache", tmp_path / "store") == tree


def test_a_tree_of_the_layout_inference_suite_stages_from_is_accepted(tmp_path, monkeypatch):
    client = tmp_path / "up" / "inferencex-e2e" / CLIENT_FILE
    client.parent.mkdir(parents=True)
    client.write_text("# client\n")
    repo = tmp_path / "up"
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    tree = ensure_inferencex(_git(repo, "rev-parse", "HEAD"), tmp_path / "cache", tmp_path / "store")

    assert (tree / "inferencex-e2e" / CLIENT_FILE).is_file()


def test_an_unknown_commit_is_refused(upstream, tmp_path, monkeypatch):
    repo, _ = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    with pytest.raises(ToolError, match="git fetch failed"):
        ensure_inferencex("a" * 40, tmp_path / "cache", tmp_path / "store")


#: A git that refuses a fetch from a repository on this machine unless its global configuration (or the system's, which
#: the test leaves alone) trusts it, as git before 2.38 does with one another account owns: ``-c`` and
#: ``GIT_CONFIG_COUNT`` do not count there. Like git, it judges the repository where it resolves to, so a link
#: trusted as itself is refused. The subcommand is the first argument after git's global options (``-c`` and ``-C`` take
#: the next argument as their value), so a fetch the host runs with ``-c`` options before it is checked too.
_OWNERSHIP_CHECKING_GIT = """#!/bin/sh
sub=
value=
for arg; do
  if [ -n "$value" ]; then value=; continue; fi
  case "$arg" in
    -c|-C) value=1;;
    -*) ;;
    *) sub="$arg"; break;;
  esac
done
if [ "$sub" = fetch ]; then
  for arg; do case "$arg" in /*|file://*) repo="${arg#file://}";; esac; done
  if [ -n "$repo" ]; then
    real=$(cd "$repo" && pwd -P) || exit 128
  fi
  if [ -n "$repo" ] && ! "%(git)s" config --global --get-all safe.directory | grep -qxF "$real"; then
    echo "fatal: detected dubious ownership in repository at '$real'" >&2
    exit 128
  fi
fi
exec "%(git)s" "$@"
"""


@pytest.fixture
def ownership_checking_git(tmp_path, monkeypatch):
    real = shutil.which("git")
    assert real is not None
    bin_dir = tmp_path / "ownership-bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(_OWNERSHIP_CHECKING_GIT % {"git": real})
    (bin_dir / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)


@pytest.mark.parametrize("prefix", ["", "file://"])
@pytest.mark.usefixtures("ownership_checking_git")
def test_a_local_repository_is_trusted_for_its_fetch_alone(upstream, tmp_path, monkeypatch, prefix):
    repo, commit = upstream
    monkeypatch.setenv("INFERENCEX_REPO", f"{prefix}{repo}")

    tree = ensure_inferencex(commit, tmp_path / "cache", tmp_path / "store")

    assert _git(tree, "rev-parse", "HEAD") == commit
    # nothing that runs after the fetch trusts it: no trust left in the checkout, none in the runner's home
    assert not (tree / ".git" / tools._TRUST_CONFIG).exists()
    assert not (tmp_path / "home").exists()


@pytest.mark.usefixtures("ownership_checking_git")
def test_a_local_repository_reached_through_a_link_is_fetched_from_and_trusted_where_it_resolves_to(
    upstream, tmp_path, monkeypatch
):
    repo, commit = upstream
    link = tmp_path / "mirror"
    link.symlink_to(repo)
    monkeypatch.setenv("INFERENCEX_REPO", str(link))

    assert _git(ensure_inferencex(commit, tmp_path / "cache", tmp_path / "store"), "rev-parse", "HEAD") == commit


def test_a_repository_another_account_owns_is_fetched_by_a_git_that_checks_its_owner(upstream, tmp_path, monkeypatch):
    if os.geteuid() != 0:
        pytest.skip("making a repository another account's needs root")
    repo, commit = upstream
    for path in (repo, *repo.rglob("*")):
        os.lchown(path, 65534, 65534)
    probe = tmp_path / "probe"
    _git(tmp_path, "init", "-q", str(probe))
    plain = subprocess.run(["git", "-C", str(probe), "fetch", "-q", "--depth", "1", str(repo), commit], check=False)
    if plain.returncode == 0:
        pytest.skip("this git fetches from a repository another account owns without asking safe.directory")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    assert _git(ensure_inferencex(commit, tmp_path / "cache", tmp_path / "store"), "rev-parse", "HEAD") == commit


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


def test_a_published_staging_is_refused_when_this_interpreters_lm_eval_is_not_at_its_pin(tmp_path, monkeypatch):
    stage_into(tmp_path / "runner-cache")
    info = tmp_path / "site" / "lm_eval-0.4.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: lm_eval\nVersion: 0.4.1\n")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))

    with pytest.raises(ToolError, match=r"cannot be identified: .* lm_eval 0\.4\.1, not lm_eval\[api\]==0\.4\.13"):
        _ensure(tmp_path)


def _filled_store(tmp_path, repo, commit):
    """The runner cache's InferenceX store holding ``commit``, as the host's fill leaves it."""
    store = tmp_path / "runner-cache" / tools.TOOL_CHECKOUTS_DIR / tools.INFERENCEX_STORE
    store.mkdir(parents=True)
    _git(store, "init", "-q", "--bare")
    _git(store, "fetch", "-q", "--depth", "1", str(repo), f"{commit}:refs/tools/{commit}")
    return store


def _snapshot(root):
    """Every path below ``root`` with its mode, mtime and bytes: what a write would change."""
    seen = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        seen[str(path.relative_to(root))] = (st.st_mode, st.st_mtime_ns, path.read_bytes() if path.is_file() else None)
    return seen


def _read_only(root):
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


def _writable(root):
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)


def test_a_tree_an_earlier_test_changed_is_not_what_a_later_test_runs(upstream, tmp_path, monkeypatch):
    repo, commit = upstream
    store = _filled_store(tmp_path, repo, commit)
    monkeypatch.setenv("INFERENCEX_REPO", str(tmp_path / "offline"))

    first = ensure_inferencex(commit, tmp_path / "scratch-1", store)
    (first / "benchmarks" / "benchmark_lib.sh").write_text("tampered\n")
    second = ensure_inferencex(commit, tmp_path / "scratch-2", store)

    assert second.is_relative_to(tmp_path / "scratch-2"), "each test runs its own tree"
    assert (second / "benchmarks" / "benchmark_lib.sh").read_text() == "# lib\n"


def test_a_read_only_store_holding_the_commit_serves_it_and_is_never_written(upstream, tmp_path, monkeypatch, caplog):
    repo, commit = upstream
    store = _filled_store(tmp_path, repo, commit)
    cache = tmp_path / "runner-cache"
    _read_only(cache)
    before = _snapshot(cache)
    monkeypatch.setenv("INFERENCEX_REPO", str(tmp_path / "offline"))

    try:
        with caplog.at_level(logging.WARNING, logger=tools.__name__):
            tree = ensure_inferencex(commit, tmp_path / "scratch", store)
        after = _snapshot(cache)
    finally:
        _writable(cache)

    assert _git(tree, "rev-parse", "HEAD") == commit, "fetched from the store: INFERENCEX_REPO does not exist"
    assert tree.is_relative_to(tmp_path / "scratch")
    assert after == before, "the runner cache is read, never written"
    assert not (tree / ".git" / "objects" / "info" / "alternates").exists(), "the tree does not borrow the store"
    assert caplog.records == []


def test_a_store_without_the_commit_falls_back_to_inferencex_repo_and_stays_unchanged(
    upstream, tmp_path, monkeypatch, caplog
):
    repo, old = upstream
    store = _filled_store(tmp_path, repo, old)
    (repo / "benchmarks" / "benchmark_lib.sh").write_text("# lib 2\n")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")
    commit = _git(repo, "rev-parse", "HEAD")
    cache = tmp_path / "runner-cache"
    _read_only(cache)
    before = _snapshot(cache)
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    try:
        with caplog.at_level(logging.WARNING, logger=tools.__name__):
            tree = ensure_inferencex(commit, tmp_path / "scratch", store)
        after = _snapshot(cache)
    finally:
        _writable(cache)

    assert _git(tree, "rev-parse", "HEAD") == commit
    assert (tree / "benchmarks" / "benchmark_lib.sh").read_text() == "# lib 2\n"
    assert after == before, "the commit is not added to the store"
    (record,) = caplog.records
    assert commit in record.getMessage() and str(store) in record.getMessage()


def test_without_a_store_the_commit_is_fetched_into_scratch_and_nothing_is_made_in_the_cache(
    upstream, tmp_path, monkeypatch, caplog
):
    repo, commit = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    cache = tmp_path / "runner-cache"
    cache.mkdir()
    _read_only(cache)
    store = cache / tools.TOOL_CHECKOUTS_DIR / tools.INFERENCEX_STORE

    try:
        with caplog.at_level(logging.WARNING, logger=tools.__name__):
            tree = ensure_inferencex(commit, tmp_path / "scratch", store)
        made = list(cache.iterdir())
    finally:
        _writable(cache)

    assert _git(tree, "rev-parse", "HEAD") == commit
    assert made == []
    assert caplog.records == [], "no store is the normal case, not a warning"


def test_an_object_changed_in_the_store_is_refused_with_a_warning_and_the_commit_fetched_again(
    upstream, tmp_path, monkeypatch, caplog
):
    repo, commit = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    store = _filled_store(tmp_path, repo, commit)
    blob = _git(repo, "rev-parse", "HEAD:benchmarks/benchmark_lib.sh")
    loose = store / "objects" / blob[:2] / blob[2:]
    loose.chmod(0o644)
    loose.write_bytes(zlib.compress(b"blob 9\x00tampered\n"))

    with caplog.at_level(logging.WARNING, logger=tools.__name__):
        second = ensure_inferencex(commit, tmp_path / "scratch-2", store)

    assert (second / "benchmarks" / "benchmark_lib.sh").read_text() == "# lib\n"
    (record,) = caplog.records
    assert commit in record.getMessage() and str(store) in record.getMessage()


def test_the_host_fills_the_store_with_the_commit_and_a_test_then_reads_it_read_only_with_no_network(
    upstream, tmp_path, monkeypatch, caplog
):
    repo, commit = upstream
    cache = tmp_path / "runner-cache"
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    store = tools.fill_inferencex_store(cache, commit)

    assert store == cache / tools.TOOL_CHECKOUTS_DIR / tools.INFERENCEX_STORE
    assert _git(store, "rev-parse", "--is-bare-repository") == "true"
    assert _git(store, "rev-parse", f"refs/tools/{commit}") == commit
    assert store.stat().st_mode & 0o005 == 0o005, "the sandbox's account reads it"
    assert [path.name for path in store.parent.iterdir()] == [tools.INFERENCEX_STORE], "no work directory is left"
    filled = _snapshot(cache)
    monkeypatch.setenv("INFERENCEX_REPO", str(tmp_path / "offline"))
    assert tools.fill_inferencex_store(cache, commit) == store, "a store that holds the commit fetches nothing"
    assert _snapshot(cache) == filled

    _read_only(cache)
    before = _snapshot(cache)
    try:
        with caplog.at_level(logging.WARNING, logger=tools.__name__):
            tree = ensure_inferencex(commit, tmp_path / "scratch", store)
        after = _snapshot(cache)
    finally:
        _writable(cache)

    assert _git(tree, "rev-parse", "HEAD") == commit, "served by the store: INFERENCEX_REPO does not exist"
    assert after == before
    assert caplog.records == []


def test_the_host_fills_a_runner_cache_named_by_a_relative_path(upstream, tmp_path, monkeypatch):
    repo, commit = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    monkeypatch.chdir(tmp_path)

    store = tools.fill_inferencex_store(Path("runner-cache"), commit)

    assert store == tmp_path / "runner-cache" / tools.TOOL_CHECKOUTS_DIR / tools.INFERENCEX_STORE
    assert _git(store, "rev-parse", f"refs/tools/{commit}") == commit


def test_the_host_adds_a_new_pin_to_a_store_that_holds_an_older_one(upstream, tmp_path, monkeypatch):
    repo, old = upstream
    cache = tmp_path / "runner-cache"
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    store = tools.fill_inferencex_store(cache, old)
    (repo / "benchmarks" / "benchmark_lib.sh").write_text("# lib 2\n")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")
    new = _git(repo, "rev-parse", "HEAD")

    assert tools.fill_inferencex_store(cache, new) == store

    assert _git(store, "rev-parse", f"refs/tools/{old}", f"refs/tools/{new}").split() == [old, new]


def test_a_commit_the_host_cannot_fetch_is_a_tool_error_and_a_bad_hash_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEX_REPO", str(tmp_path / "offline"))

    with pytest.raises(ToolError, match="was not fetched into"):
        tools.fill_inferencex_store(tmp_path / "runner-cache", "a" * 40)
    with pytest.raises(ToolError, match="not a full commit hash"):
        tools.fill_inferencex_store(tmp_path / "runner-cache", "main")


@pytest.mark.usefixtures("ownership_checking_git")
def test_a_store_below_a_path_with_a_space_is_trusted_and_served(upstream, tmp_path, monkeypatch, caplog):
    repo, commit = upstream
    store = tmp_path / "runner cache" / tools.TOOL_CHECKOUTS_DIR / tools.INFERENCEX_STORE
    store.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "--bare", str(store)], check=True)
    _git(
        tmp_path, f"--git-dir={store}", "fetch", "-q", "--depth", "1", f"file://{repo}", f"{commit}:refs/tools/{commit}"
    )
    monkeypatch.setenv("INFERENCEX_REPO", str(tmp_path / "offline"))

    with caplog.at_level(logging.WARNING, logger=tools.__name__):
        tree = ensure_inferencex(commit, tmp_path / "scratch", store)

    assert _git(tree, "rev-parse", "HEAD") == commit
    assert caplog.records == []


@pytest.mark.usefixtures("ownership_checking_git")
def test_a_store_fetch_without_its_trust_entry_falls_back_with_a_warning_naming_the_source(
    upstream, tmp_path, monkeypatch, caplog
):
    repo, commit = upstream
    store = _filled_store(tmp_path, repo, commit)
    real = tools._trusting
    untrusted = os.path.realpath(store)

    def trusting_all_but_the_store(directories, config, cwd):
        return real([directory for directory in directories if directory != untrusted], config, cwd)

    monkeypatch.setattr(tools, "_trusting", trusting_all_but_the_store)
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))

    with caplog.at_level(logging.WARNING, logger=tools.__name__):
        tree = ensure_inferencex(commit, tmp_path / "scratch", store)

    assert _git(tree, "rev-parse", "HEAD") == commit
    (record,) = caplog.records
    assert "dubious ownership" in record.getMessage()
    assert f"so it is fetched from {repo}" in record.getMessage()


def test_without_inferencex_repo_the_warning_names_upstream(upstream, tmp_path, monkeypatch, caplog):
    repo, old = upstream
    store = _filled_store(tmp_path, repo, old)
    monkeypatch.delenv("INFERENCEX_REPO", raising=False)
    monkeypatch.setattr(tools, "INFERENCEX_REPO", str(repo))  # stands in for upstream

    with caplog.at_level(logging.WARNING, logger=tools.__name__):
        tools.ensure_inferencex(old, tmp_path / "scratch", tmp_path / "no-store")
        assert caplog.records == []
        with pytest.raises(ToolError):
            tools.ensure_inferencex("b" * 40, tmp_path / "scratch-2", store)

    assert f"so it is fetched from {repo}" in caplog.records[0].getMessage()
    assert "INFERENCEX_REPO" not in caplog.records[0].getMessage()


@pytest.mark.usefixtures("ownership_checking_git")
def test_the_host_fetches_from_a_local_mirror_trusted_for_the_fill_alone(upstream, tmp_path, monkeypatch):
    repo, commit = upstream
    monkeypatch.setenv("INFERENCEX_REPO", f"file://{repo}")

    store = tools.fill_inferencex_store(tmp_path / "runner-cache", commit)

    assert _git(store, "rev-parse", f"refs/tools/{commit}") == commit
    assert not (tmp_path / "home").exists(), "nothing is trusted beyond the fill"


@pytest.mark.usefixtures("ownership_checking_git")
def test_the_ownership_checking_git_checks_a_fetch_after_global_options(upstream, tmp_path):
    repo, commit = upstream
    store = tmp_path / "store.git"
    subprocess.run(["git", "init", "-q", "--bare", str(store)], check=True)

    done = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            f"--git-dir={store}",
            "fetch",
            "-q",
            "--depth",
            "1",
            str(repo),
            commit,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": str(tmp_path / "empty.gitconfig")},
    )

    assert done.returncode == 128 and "dubious ownership" in done.stderr


def test_tools_is_made_group_shared_below_a_setgid_runner_cache(upstream, tmp_path, monkeypatch):
    repo, commit = upstream
    cache = tmp_path / "runner-cache"
    cache.mkdir()
    cache.chmod(0o2775)
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    old = os.umask(0o022)
    try:
        store = tools.fill_inferencex_store(cache, commit)
    finally:
        os.umask(old)

    assert store.parent.stat().st_mode & 0o7777 == 0o2775, "a builder of another account of the group writes it"


#: Two fills of the store run as separate processes, each starting its fill at the same moment (``argv[3]``, once both
#: have imported); the source pauses a second in the middle of each pack, while the receiving fetch holds the store's
#: ``shallow.lock``, so the two fetches overlap as two fetches of a real InferenceX do.
_FILL = "import sys, time; from pathlib import Path; from hyperloom.measure.runner import tools; " + (
    "time.sleep(max(0.0, float(sys.argv[3]) - time.time())); tools.fill_inferencex_store(Path(sys.argv[1]), sys.argv[2])"
)


def _slow_source(tmp_path, pause=1):
    """The global git configuration of a fill whose source sends a pack in two parts ``pause`` seconds apart: the
    host's own configuration, which the fill includes, makes ``git upload-pack`` run ``git pack-objects`` through a
    hook that passes its first bytes on at once and the rest ``pause`` seconds later."""
    config = tmp_path / f"slow-{pause}.gitconfig"
    hook = f"sh -c '\\\"$@\\\" | {{ dd bs=1 count=32 2>/dev/null; sleep {pause}; cat; }}' hook"
    config.write_text(f'[uploadpack]\n\tpackObjectsHook = "{hook}"\n')
    return config


def _fill_at_once(cache, commits, env):
    import hyperloom
    from inference_suite import __file__ as inference_suite_file

    paths = {os.path.dirname(os.path.dirname(path)) for path in (hyperloom.__file__, inference_suite_file)}
    env = {**env, "PYTHONPATH": os.pathsep.join([*paths, env.get("PYTHONPATH", "")])}
    start = str(time.time() + 3)
    fills = [
        subprocess.Popen(
            [sys.executable, "-c", _FILL, str(cache), commit, start],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for commit in commits
    ]
    return [(fill.wait(timeout=120), fill.stderr.read()) for fill in fills]


def _second_commit(repo):
    (repo / "benchmarks" / "benchmark_lib.sh").write_text("# lib 2\n")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("case", ["one pin on an empty cache", "two pins on an empty cache", "a new pin beside an old"])
def test_two_fills_at_once_both_succeed(upstream, tmp_path, monkeypatch, case):
    repo, old = upstream
    new = _second_commit(repo)
    env = {**os.environ, "INFERENCEX_REPO": str(repo), "GIT_CONFIG_GLOBAL": str(_slow_source(tmp_path))}
    commits = {"one pin on an empty cache": [old, old], "two pins on an empty cache": [old, new]}.get(case, [new, new])
    for round_ in range(3):
        cache = tmp_path / f"runner-cache-{round_}"
        if case == "a new pin beside an old":
            monkeypatch.setenv("INFERENCEX_REPO", str(repo))
            tools.fill_inferencex_store(cache, old)

        results = _fill_at_once(cache, commits, env)

        assert [code for code, _ in results] == [0, 0], results
        store = cache / tools.TOOL_CHECKOUTS_DIR / tools.INFERENCEX_STORE
        for commit in {*commits}:
            assert _git(store, "rev-parse", f"refs/tools/{commit}") == commit
        assert not (store / "shallow.lock").exists()


def test_a_lock_a_killed_fill_left_is_removed_once_stale_and_the_fill_succeeds(upstream, tmp_path, monkeypatch, caplog):
    repo, old = upstream
    cache = tmp_path / "runner-cache"
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    store = tools.fill_inferencex_store(cache, old)
    new = _second_commit(repo)
    lock = store / "shallow.lock"
    lock.write_text("")
    stale = time.time() - tools._STALE_LOCK_SEC - 60
    os.utime(lock, (stale, stale))

    with caplog.at_level(logging.WARNING, logger=tools.__name__):
        assert tools.fill_inferencex_store(cache, new) == store

    assert _git(store, "rev-parse", f"refs/tools/{new}") == new
    assert not lock.exists() and sorted(store.glob("shallow.lock.*")) == []
    assert [record.getMessage() for record in caplog.records] == [f"removed {lock}, which a killed fill left"]


def test_a_fresh_lock_is_waited_for_and_the_fill_succeeds_once_it_is_released(upstream, tmp_path, monkeypatch):
    repo, old = upstream
    cache = tmp_path / "runner-cache"
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    store = tools.fill_inferencex_store(cache, old)
    new = _second_commit(repo)
    lock = store / "shallow.lock"
    lock.write_text("")
    monkeypatch.setattr(tools, "_LOCK_PAUSE_SEC", 0.2)
    released = threading.Timer(1.0, lock.unlink)
    released.start()
    try:
        assert tools.fill_inferencex_store(cache, new) == store
    finally:
        released.join()

    assert _git(store, "rev-parse", f"refs/tools/{new}") == new


def test_a_fresh_lock_held_past_the_wait_is_a_tool_error_and_left_in_place(upstream, tmp_path, monkeypatch):
    repo, old = upstream
    cache = tmp_path / "runner-cache"
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    store = tools.fill_inferencex_store(cache, old)
    new = _second_commit(repo)
    lock = store / "shallow.lock"
    lock.write_text("")
    monkeypatch.setattr(tools, "_LOCK_WAIT_SEC", 1.0)
    monkeypatch.setattr(tools, "_LOCK_PAUSE_SEC", 0.2)

    with pytest.raises(ToolError, match="another fill held a lock of .* for 1.0 s"):
        tools.fill_inferencex_store(cache, new)

    assert lock.exists(), "a lock another fill may still hold is never removed while it is fresh"


def test_a_fill_whose_fetch_times_out_fails_at_once_and_leaves_no_lock(upstream, tmp_path, monkeypatch, caplog):
    """The source stalls in the middle of the pack, while the fill's own fetch holds ``shallow.lock``: the fetch is
    stopped at its timeout, removes the lock itself, and the fill fails then, never waiting on that lock as another
    fill's; the next fill fetches at once."""
    repo, old = upstream
    cache = tmp_path / "runner-cache"
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    store = tools.fill_inferencex_store(cache, old)
    new = _second_commit(repo)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(_slow_source(tmp_path, pause=60)))
    monkeypatch.setattr(tools, "_FETCH_TIMEOUT_SEC", 3)
    lock = store / "shallow.lock"
    seen = []
    watching = threading.Event()

    def watch():
        while not watching.wait(0.05):
            if lock.exists():
                seen.append(lock)

    watcher = threading.Thread(target=watch)
    watcher.start()
    started = time.monotonic()
    try:
        with caplog.at_level(logging.INFO, logger=tools.__name__), pytest.raises(ToolError) as failed:
            tools.fill_inferencex_store(cache, new)
    finally:
        watching.set()
        watcher.join()

    assert seen, "the fill's own fetch held the lock while the source stalled"
    assert "still running after 3 s and was stopped" in str(failed.value)
    assert "another fill" not in str(failed.value)
    assert time.monotonic() - started < 3 + tools._STOP_GRACE_SEC
    assert not lock.exists(), "the stopped fetch removed the lock it took"
    assert [record.getMessage() for record in caplog.records] == []

    monkeypatch.delenv("GIT_CONFIG_GLOBAL")
    with caplog.at_level(logging.INFO, logger=tools.__name__):
        assert tools.fill_inferencex_store(cache, new) == store
    assert _git(store, "rev-parse", f"refs/tools/{new}") == new
    assert [record.getMessage() for record in caplog.records] == []


def test_a_fetch_failing_on_no_lock_raises_at_once_and_leaves_a_lock_that_is_there(upstream, tmp_path, monkeypatch):
    """A lock file that is merely there, which a concurrent fill may hold, does not make another failure a wait: only
    git's own message says a lock is held. The lock is the other fill's to remove and stays."""
    repo, old = upstream
    cache = tmp_path / "runner-cache"
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    store = tools.fill_inferencex_store(cache, old)
    new = _second_commit(repo)
    lock = store / "shallow.lock"
    lock.write_text("")
    real = tools._git
    fetches = []

    def failing_without_a_lock(*args, **kwargs):
        if "fetch" in args:
            fetches.append(args)
            raise ToolError("git -c failed: fatal: early EOF")
        return real(*args, **kwargs)

    monkeypatch.setattr(tools, "_git", failing_without_a_lock)

    with pytest.raises(ToolError, match="early EOF") as failed:
        tools.fill_inferencex_store(cache, new)

    assert len(fetches) == 1 and "another fill" not in str(failed.value)
    assert lock.exists()


@pytest.mark.parametrize(
    "message",
    [
        "fatal: Unable to create '/s/shallow.lock': File exists.",
        "Another git process seems to be running in this repository, or the lock file may be stale",
        "may have crashed in this repository earlier:\nremove the file manually to continue.",
        "fatal: shallow file has changed since we read it",
    ],
)
def test_a_lock_already_released_when_the_fetch_failed_on_it_is_tried_again(upstream, tmp_path, monkeypatch, message):
    repo, commit = upstream
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    monkeypatch.setattr(tools, "_LOCK_PAUSE_SEC", 0.1)
    real = tools._git
    failed = []

    def failing_once_on_a_lock(*args, **kwargs):
        if "fetch" in args and not failed:
            failed.append(args)
            raise ToolError(f"git -c failed: {message}")
        return real(*args, **kwargs)

    monkeypatch.setattr(tools, "_git", failing_once_on_a_lock)

    store = tools.fill_inferencex_store(tmp_path / "runner-cache", commit)

    assert failed and _git(store, "rev-parse", f"refs/tools/{commit}") == commit
