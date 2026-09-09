# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import hyperloom.orchestrator.kernel.campaign_baseline as campaign_baseline
from hyperloom.orchestrator.kernel.campaign_baseline import (
    RepoBaseline,
    campaign_repositories,
    reclaim_campaign_repositories,
    seal_campaign_baseline,
    session_branch_name,
)
from kernelforge.kernel_rewrite_controller.worktree import (
    CAMPAIGN_BRANCH_PREFIX,
    FORGE_LOOP_OUTPUT_DIRNAME,
)

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "baseline-test",
    "GIT_AUTHOR_EMAIL": "baseline-test@local",
    "GIT_COMMITTER_NAME": "baseline-test",
    "GIT_COMMITTER_EMAIL": "baseline-test@local",
}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path, name: str = "framework") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init")
    (repo / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "upstream")
    return repo


def _state(repo: Path) -> SimpleNamespace:
    return SimpleNamespace(framework_repo_path=str(repo))


@pytest.fixture(autouse=True)
def _no_runtime_discovery(monkeypatch: pytest.MonkeyPatch):
    """Keep these tests off whatever framework the host has installed.

    ``seal_campaign_baseline`` commits, and it finds its repositories from the
    interpreter. Left alone it would reach a real sglang or aiter checkout.

    Stubbed at ``find_spec`` rather than at ``_package_repository`` so the
    resolver itself still runs for the tests that are about it.
    """
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)


def test_a_clean_repository_is_pinned_without_a_new_commit(tmp_path: Path) -> None:
    """Nothing to seal is not a reason to add a commit nobody asked for."""
    repo = _repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    pins = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)

    assert pins[str(repo)].commit == head.lower()
    assert _git(repo, "rev-parse", "HEAD") == head


def test_the_serving_tree_is_sealed_into_the_pinned_commit(tmp_path: Path) -> None:
    """The campaign's base has to be the code the server is actually running."""
    repo = _repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")

    pins = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)

    sealed = pins[str(repo)].commit
    assert sealed != head.lower()
    assert _git(repo, "rev-parse", "HEAD").lower() == sealed
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert _git(repo, "show", f"{sealed}:kernel.py") == "VALUE = 2"
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == session_branch_name("s1", 0)


def test_untracked_files_are_left_out_of_the_seal(tmp_path: Path) -> None:
    """Tuned tables and JIT caches are in no patch and belong in no commit."""
    repo = _repo(tmp_path)
    (repo / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "tuned_gemm.csv").write_text("shape,config\n", encoding="utf-8")

    seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)

    assert _git(repo, "status", "--porcelain") == "?? tuned_gemm.csv"


def test_a_detached_head_is_sealed_onto_the_session_branch(tmp_path: Path) -> None:
    """The framework checkouts ride a detached upstream commit, not a branch."""
    repo = _repo(tmp_path)
    _git(repo, "checkout", "--detach", "HEAD")
    (repo / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")

    pins = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)

    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == session_branch_name("s1", 0)
    assert _git(repo, "rev-parse", "HEAD").lower() == pins[str(repo)].commit


def test_a_second_entry_seals_on_top_of_the_first(tmp_path: Path) -> None:
    """A macro cycle can enter KERNEL twice, and the branch is already there."""
    repo = _repo(tmp_path)
    (repo / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)[str(repo)].commit
    (repo / "kernel.py").write_text("VALUE = 3\n", encoding="utf-8")

    second = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)[str(repo)].commit

    assert second != first
    assert _git(repo, "show", f"{second}:kernel.py") == "VALUE = 3"
    assert _git(repo, "rev-parse", f"{second}^").lower() == first


def test_a_repository_that_cannot_be_sealed_does_not_stop_the_others(tmp_path: Path) -> None:
    """One unusable tree costs its own operators, never the whole phase."""
    good = _repo(tmp_path, "good")
    (good / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    broken = tmp_path / "broken"
    (broken / ".git").mkdir(parents=True)
    state = SimpleNamespace(framework_repo_path=os.pathsep.join([str(good)]))
    os.environ["INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS"] = str(broken)
    try:
        pins = seal_campaign_baseline(state, session_id="s1", macro_cycle=0)
    finally:
        os.environ.pop("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", None)

    assert str(good) in pins
    assert str(broken) not in pins


def test_the_framework_being_served_is_found_without_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """All three configured sources were empty in the GLM-5.2 session."""
    repo = _repo(tmp_path, "sglang-checkout")
    package = repo / "python" / "sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        campaign_baseline,
        "_package_repository",
        lambda name: repo.resolve() if name == "sglang" else None,
    )

    roots = campaign_repositories(SimpleNamespace(framework_repo_path=""))

    assert roots == (repo.resolve(),)


def test_a_configured_root_is_added_to_what_the_runtime_found(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An operator pointing at a fourth checkout is honoured, not overridden."""
    served = _repo(tmp_path, "served")
    extra = _repo(tmp_path, "extra")
    monkeypatch.setattr(
        campaign_baseline,
        "_package_repository",
        lambda name: served.resolve() if name == "aiter" else None,
    )

    roots = campaign_repositories(SimpleNamespace(framework_repo_path=str(extra)))

    assert set(roots) == {served.resolve(), extra.resolve()}


def test_a_wheel_installed_framework_is_not_a_repository_to_seal(monkeypatch) -> None:
    """A wheel carries no source to rewrite, so it has no base to pin."""
    from types import SimpleNamespace as Spec

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: Spec(origin="/opt/venv/lib/python3.10/site-packages/vllm/__init__.py"),
    )

    assert campaign_baseline._package_repository("vllm") is None


def test_a_repository_a_killed_controller_left_on_a_branch_is_reclaimed(tmp_path: Path) -> None:
    """Integration refuses a HEAD that is not the base commit it was promised."""
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").lower()
    _git(repo, "checkout", "-b", f"{CAMPAIGN_BRANCH_PREFIX}abandoned")
    (repo / "kernel.py").write_text("half-finished\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "campaign work")
    (repo / FORGE_LOOP_OUTPUT_DIRNAME).mkdir()

    reclaimed = reclaim_campaign_repositories({str(repo): RepoBaseline(commit=base)})

    assert str(repo) in reclaimed
    assert _git(repo, "rev-parse", "HEAD").lower() == base
    assert (repo / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (repo / FORGE_LOOP_OUTPUT_DIRNAME).exists()


def test_a_repository_the_controller_returned_cleanly_is_left_alone(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").lower()
    branch_before = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")

    assert reclaim_campaign_repositories({str(repo): RepoBaseline(commit=base)}) == {}
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == branch_before


def test_a_configured_file_resolves_to_the_repository_holding_it(tmp_path: Path) -> None:
    """A launch recipe or a source file names its repository just as well."""
    repo = _repo(tmp_path)
    inside = repo / "nested" / "config.yaml"
    inside.parent.mkdir()
    inside.write_text("{}\n", encoding="utf-8")

    roots = campaign_repositories(SimpleNamespace(framework_repo_path=str(inside)))

    assert roots == (repo.resolve(),)


def test_a_configured_path_in_no_repository_is_dropped(tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()

    assert campaign_repositories(SimpleNamespace(framework_repo_path=str(loose))) == ()


def test_a_package_that_cannot_be_imported_names_no_repository(monkeypatch) -> None:
    """A broken install is not a repository, and must not raise on the way out."""

    def _raise(_name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib.util, "find_spec", _raise)
    assert campaign_baseline._package_repository("sglang") is None

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert campaign_baseline._package_repository("sglang") is None


def test_a_namespace_package_with_no_origin_names_no_repository(monkeypatch) -> None:
    from types import SimpleNamespace as Spec

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: Spec(origin=None))

    assert campaign_baseline._package_repository("sglang") is None


def test_reclaiming_a_repository_that_is_gone_is_reported_not_raised(tmp_path: Path) -> None:
    """One unusable tree must not stop the repositories beside it."""
    good = _repo(tmp_path, "good")
    base = _git(good, "rev-parse", "HEAD").lower()
    _git(good, "checkout", "-b", f"{CAMPAIGN_BRANCH_PREFIX}abandoned")

    reclaimed = reclaim_campaign_repositories(
        {
            str(tmp_path / "absent"): RepoBaseline(commit="b" * 40),
            str(good): RepoBaseline(commit=base),
        },
    )

    assert str(good) in reclaimed
    assert str(tmp_path / "absent") not in reclaimed
