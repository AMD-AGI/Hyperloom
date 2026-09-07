# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from hyperloom.orchestrator.kernel.campaign_baseline import (
    seal_campaign_baseline,
    session_branch_name,
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


def test_a_clean_repository_is_pinned_without_a_new_commit(tmp_path: Path) -> None:
    """Nothing to seal is not a reason to add a commit nobody asked for."""
    repo = _repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    pins = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)

    assert pins == {str(repo): head.lower()}
    assert _git(repo, "rev-parse", "HEAD") == head


def test_the_serving_tree_is_sealed_into_the_pinned_commit(tmp_path: Path) -> None:
    """The campaign's base has to be the code the server is actually running."""
    repo = _repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")

    pins = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)

    sealed = pins[str(repo)]
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
    assert _git(repo, "rev-parse", "HEAD").lower() == pins[str(repo)]


def test_a_second_entry_seals_on_top_of_the_first(tmp_path: Path) -> None:
    """A macro cycle can enter KERNEL twice, and the branch is already there."""
    repo = _repo(tmp_path)
    (repo / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)[str(repo)]
    (repo / "kernel.py").write_text("VALUE = 3\n", encoding="utf-8")

    second = seal_campaign_baseline(_state(repo), session_id="s1", macro_cycle=0)[str(repo)]

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
