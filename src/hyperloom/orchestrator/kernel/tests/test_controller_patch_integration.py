# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name
from hyperloom.orchestrator.actions.executors._patch_snapshot import _git_commit_kept
from hyperloom.orchestrator.kernel import controller_patch_integration as integration
from hyperloom.orchestrator.kernel.controller_patch_integration import (
    integrate_controller_patches,
)
from hyperloom.orchestrator.state.shared_state import SharedState

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "integration-test",
    "GIT_AUTHOR_EMAIL": "integration-test@local",
    "GIT_COMMITTER_NAME": "integration-test",
    "GIT_COMMITTER_EMAIL": "integration-test@local",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "first.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "second.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _named_repo(parent: Path, name: str, filename: str) -> tuple[Path, str]:
    repo = parent / name
    repo.mkdir()
    _git(repo, "init")
    (repo / filename).write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", f"{name} baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _patch(repo: Path, relative: str, content: str) -> str:
    path = repo / relative
    original = path.read_text(encoding="utf-8")
    path.write_text(content, encoding="utf-8")
    patch = _git(repo, "diff", "--binary", "--", relative)
    path.write_text(original, encoding="utf-8")
    return patch + "\n"


def _publish(
    patches_root: Path,
    repo: Path,
    base_commit: str,
    *,
    kernel_name: str,
    kernel_path: str,
    patch: str,
) -> Path:
    operator_id = f"kernel:forge-loop:{kernel_name}:standalone:unknown:triton:mi355x"
    patch_dir = patches_root / operator_directory_name(operator_id)
    patch_dir.mkdir(parents=True)
    (patch_dir / "change.patch").write_text(patch, encoding="utf-8")
    (patch_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (patch_dir / "publication.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "operator_id": operator_id,
                "identity": {
                    "producer": "forge-loop",
                    "kernel_name": kernel_name,
                    "framework": "standalone",
                    "framework_version": "unknown",
                    "backend": "triton",
                    "gpu": "mi355x",
                },
                "base_commit": base_commit,
                "best_commit": "b" * 40,
                "repo_root": str(repo),
                "kernel_path": kernel_path,
                "operator_name": kernel_name,
                "micro_validated": True,
                "manifest": {"changed_files": [kernel_path]},
            }
        ),
        encoding="utf-8",
    )
    return patch_dir


def _state(session_dir: Path, repo: Path) -> SharedState:
    state = SharedState(
        baseline_tput=100.0,
        current_best={"action": "baseline", "tput": 100.0},
        framework_repo_path=str(repo),
    )
    state.save(session_dir)
    return state


@pytest.mark.asyncio
async def test_multiple_patches_are_kept_and_committed_one_by_one(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="first",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        repo,
        base,
        kernel_name="second",
        kernel_path="second.py",
        patch=_patch(repo, "second.py", "VALUE = 3\n"),
    )
    seen: list[str] = []

    async def _validate(publication):
        seen.append(publication.identity["kernel_name"])
        return {
            "decision": "KEEP",
            "new_tput": 110.0 + len(seen),
            "gain_pct": 10.0 + len(seen),
        }

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    state = _state(session_dir, repo)
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=state,
        validator=_validate,
    )

    assert seen == ["first", "second"]
    assert summary.kept_count == 2
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (repo / "second.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    assert int(_git(repo, "rev-list", "--count", "HEAD")) == 3
    assert len(state.optimization_stack) == 2
    assert state.current_best["tput"] == 112.0


@pytest.mark.asyncio
async def test_conflicting_patch_is_skipped_without_reverting_prior_keep(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="a_first",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        repo,
        base,
        kernel_name="b_conflict",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 99\n"),
    )

    async def _keep(_publication):
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_keep,
    )

    assert [result.status for result in summary.results] == [
        "kept",
        "reverted_apply_conflict",
    ]
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert int(_git(repo, "rev-list", "--count", "HEAD")) == 2


@pytest.mark.asyncio
async def test_e2e_failure_reverts_only_current_patch_and_continues(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="first",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        repo,
        base,
        kernel_name="second",
        kernel_path="second.py",
        patch=_patch(repo, "second.py", "VALUE = 3\n"),
    )

    async def _validate(publication):
        if publication.identity["kernel_name"] == "second":
            return {"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.kept_count == 1
    assert summary.reverted_count == 1
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (repo / "second.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert int(_git(repo, "rev-list", "--count", "HEAD")) == 2


@pytest.mark.asyncio
async def test_a_revert_leaves_the_operators_untracked_files_alone(tmp_path: Path) -> None:
    """A failed patch must not take the operator's own files with it.

    Admission asks ``git status --untracked-files=no``, so an untracked file is
    admitted rather than refused -- which means a tree-wide ``git clean`` on the
    revert path would delete work this integration never looked at.
    """
    repo, base = _repo(tmp_path)
    (repo / "notes.md").write_text("operator notes\n", encoding="utf-8")
    (repo / "bench_local").mkdir()
    (repo / "bench_local" / "run.sh").write_text("echo bench\n", encoding="utf-8")
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="first",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )

    async def _validate(_publication):
        return {"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.reverted_count == 1
    # The patch itself is gone.
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    # Everything the patch never named is still here.
    assert (repo / "notes.md").read_text(encoding="utf-8") == "operator notes\n"
    assert (repo / "bench_local" / "run.sh").read_text(encoding="utf-8") == "echo bench\n"


@pytest.mark.asyncio
async def test_a_revert_unstages_a_patch_whose_commit_never_landed(tmp_path: Path) -> None:
    """Reverting the working tree is not enough once a commit attempt staged it.

    A staged leftover reads as a dirty tree, which would make every later patch
    in the same run skip on an admission check it has nothing to do with.
    """
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="first",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        repo,
        base,
        kernel_name="second",
        kernel_path="second.py",
        patch=_patch(repo, "second.py", "VALUE = 3\n"),
    )

    def _commit_nothing(_repo: Path, _message: str, _paths: list[str]) -> tuple[bool, str]:
        return True, "nothing to commit"

    calls: list[str] = []

    async def _validate(publication):
        calls.append(publication.identity["kernel_name"])
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    original = integration._git_commit_kept
    integration._git_commit_kept = _commit_nothing  # type: ignore[assignment]
    try:
        summary = await integrate_controller_patches(
            patches_root=patches,
            session_dir=session_dir,
            shared_state=_state(session_dir, repo),
            validator=_validate,
        )
    finally:
        integration._git_commit_kept = original  # type: ignore[assignment]

    assert [result.status for result in summary.results] == [
        "reverted_commit_failed",
        "reverted_commit_failed",
    ]
    # The second patch was reached, so the first one's revert left no dirty index.
    assert calls == ["first", "second"]
    assert _git(repo, "status", "--porcelain") == ""


@pytest.mark.asyncio
async def test_controller_base_mismatch_is_rejected_before_apply(tmp_path: Path) -> None:
    repo, _base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        "a" * 40,
        kernel_name="mismatch",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )

    async def _must_not_validate(_publication):
        raise AssertionError("baseline mismatch must not reach E2E")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_must_not_validate,
    )

    assert summary.results[0].status == "skipped_baseline_mismatch"
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert int(_git(repo, "rev-list", "--count", "HEAD")) == 1


@pytest.mark.asyncio
async def test_a_dirty_patch_path_is_skipped_without_cleaning_the_repository(tmp_path: Path) -> None:
    """The refusal must leave the operator's own edits exactly where they are."""
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="dirty",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    (repo / "first.py").write_text("USER_CHANGE = True\n", encoding="utf-8")
    (repo / "second.py").write_text("UNRELATED = True\n", encoding="utf-8")

    async def _must_not_validate(_publication):
        raise AssertionError("dirty worktree must not reach E2E")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_must_not_validate,
    )

    assert summary.results[0].status == "skipped_dirty_worktree"
    assert (repo / "first.py").read_text(encoding="utf-8") == "USER_CHANGE = True\n"
    assert (repo / "second.py").read_text(encoding="utf-8") == "UNRELATED = True\n"


@pytest.mark.asyncio
async def test_a_note_alongside_a_real_commit_does_not_revert_the_keep(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # _git_commit_kept documents its note as carrying "any detail", so a caller
    # that reads any note as failure would revert a KEEP that did commit and had
    # already passed the serving gate.
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="noted",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )

    def _commit_with_advisory_note(repo_root, message, paths):
        committed, _note = _git_commit_kept(repo_root, message, paths)
        return committed, "staged 1 path"

    monkeypatch.setattr(integration, "_git_commit_kept", _commit_with_advisory_note)

    async def _keep(_publication):
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_keep,
    )

    assert [result.status for result in summary.results] == ["kept"]
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert int(_git(repo, "rev-list", "--count", "HEAD")) == 2


@pytest.mark.asyncio
async def test_a_commit_that_never_lands_reverts_without_poisoning_the_next_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # The benign no-op shape: success, a note, and no commit. Admitting it would
    # record a keep_commit that does not carry the change and leave the worktree
    # dirty, which makes every later publication fail the dirty-worktree check.
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="first",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        repo,
        base,
        kernel_name="second",
        kernel_path="second.py",
        patch=_patch(repo, "second.py", "VALUE = 3\n"),
    )

    def _no_op_for_first(repo_root, message, paths):
        if "first" in message:
            return True, "nothing to commit"
        return _git_commit_kept(repo_root, message, paths)

    monkeypatch.setattr(integration, "_git_commit_kept", _no_op_for_first)

    async def _keep(_publication):
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    state = _state(session_dir, repo)
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=state,
        validator=_keep,
    )

    assert [result.status for result in summary.results] == ["reverted_commit_failed", "kept"]
    assert summary.results[0].reason == "nothing to commit"
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (repo / "second.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    assert int(_git(repo, "rev-list", "--count", "HEAD")) == 2
    assert [entry["operator_id"] for entry in state.optimization_stack] == [
        "kernel:forge-loop:second:standalone:unknown:triton:mi355x"
    ]


@pytest.mark.asyncio
async def test_patches_from_separate_repositories_each_keep_their_own_baseline(tmp_path: Path) -> None:
    # A framework session hands the controller more than one editable repository
    # (sglang and aiter here), and their HEADs are unrelated. Each publication
    # must be graded against its own repository's baseline; requiring one shared
    # commit would discard every patch from the second repository.
    aiter, aiter_base = _named_repo(tmp_path, "aiter", "moe.py")
    sglang, sglang_base = _named_repo(tmp_path, "sglang", "norm.py")
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        aiter,
        aiter_base,
        kernel_name="moe_stage1",
        kernel_path="moe.py",
        patch=_patch(aiter, "moe.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        sglang,
        sglang_base,
        kernel_name="rmsnorm",
        kernel_path="norm.py",
        patch=_patch(sglang, "norm.py", "VALUE = 3\n"),
    )
    seen: list[str] = []

    async def _validate(publication):
        seen.append(publication.identity["kernel_name"])
        return {"decision": "KEEP", "new_tput": 110.0 + len(seen), "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        # The configured root is their common parent so both repositories are
        # admissible; the point under test is the baseline, not the allowlist.
        shared_state=_state(session_dir, tmp_path),
        validator=_validate,
    )

    assert seen == ["moe_stage1", "rmsnorm"]
    assert summary.kept_count == 2
    assert (aiter / "moe.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (sglang / "norm.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    assert int(_git(aiter, "rev-list", "--count", "HEAD")) == 2
    assert int(_git(sglang, "rev-list", "--count", "HEAD")) == 2


@pytest.mark.asyncio
async def test_second_base_within_one_repository_is_still_rejected(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="first",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        repo,
        "c" * 40,
        kernel_name="second",
        kernel_path="second.py",
        patch=_patch(repo, "second.py", "VALUE = 3\n"),
    )
    seen: list[str] = []

    async def _validate(publication):
        seen.append(publication.identity["kernel_name"])
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert seen == ["first"]
    assert summary.kept_count == 1
    assert [result.status for result in summary.results] == ["kept", "skipped_baseline_mismatch"]
    assert "pinned to controller base" in summary.results[1].reason
    assert (repo / "second.py").read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.asyncio
async def test_publication_outside_configured_roots_is_rejected(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    publication_root = tmp_path / "publication"
    allowed_root.mkdir()
    publication_root.mkdir()
    allowed_repo, _allowed_base = _repo(allowed_root)
    publication_repo, publication_base = _repo(publication_root)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        publication_repo,
        publication_base,
        kernel_name="outside",
        kernel_path="first.py",
        patch=_patch(publication_repo, "first.py", "VALUE = 2\n"),
    )

    async def _must_not_validate(_publication):
        raise AssertionError("outside repo must not reach E2E")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, allowed_repo),
        validator=_must_not_validate,
    )

    assert summary.results[0].status == "skipped_invalid"
    assert "outside the configured patch target roots" in summary.results[0].reason


@pytest.mark.asyncio
async def test_an_invalid_publication_is_skipped_and_the_next_one_still_lands(tmp_path: Path) -> None:
    """One bad publication must not cost the patches queued behind it."""
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    broken = _publish(
        patches,
        repo,
        base,
        kernel_name="aaa_broken",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    (broken / "publication.json").write_text("{not json", encoding="utf-8")
    _publish(
        patches,
        repo,
        base,
        kernel_name="zzz_good",
        kernel_path="second.py",
        patch=_patch(repo, "second.py", "VALUE = 3\n"),
    )

    async def _validate(_publication):
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    statuses = {result.status for result in summary.results}
    assert "skipped_invalid" in statuses
    assert summary.kept_count == 1
    assert (repo / "second.py").read_text(encoding="utf-8") == "VALUE = 3\n"


@pytest.mark.asyncio
async def test_a_patch_that_does_not_apply_is_reverted_not_left_half_staged(tmp_path: Path) -> None:
    """A malformed diff fails at ``git apply``; the tree must come back clean."""
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="malformed",
        kernel_path="first.py",
        patch="diff --git a/first.py b/first.py\n@@ this is not a hunk @@\n",
    )
    validated = False

    async def _validate(_publication):
        nonlocal validated
        validated = True
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.kept_count == 0
    assert [r.status for r in summary.results] == ["reverted_apply_conflict"]
    # A patch that never applied must not reach the benchmark.
    assert validated is False
    assert _git(repo, "status", "--porcelain") == ""


@pytest.mark.asyncio
async def test_a_validator_that_raises_reverts_its_patch_and_continues(tmp_path: Path) -> None:
    """An E2E that dies is a failed patch, not a failed integration run."""
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="aaa_raises",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    _publish(
        patches,
        repo,
        base,
        kernel_name="zzz_survives",
        kernel_path="second.py",
        patch=_patch(repo, "second.py", "VALUE = 3\n"),
    )

    async def _validate(publication):
        if publication.identity["kernel_name"] == "aaa_raises":
            raise RuntimeError("serving benchmark died")
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    statuses = [r.status for r in summary.results]
    assert statuses[0] == "reverted_e2e_failed"
    assert "serving benchmark died" in (summary.results[0].reason or "")
    assert summary.kept_count == 1
    # The raising patch left nothing behind; the next one still landed.
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (repo / "second.py").read_text(encoding="utf-8") == "VALUE = 3\n"


@pytest.mark.asyncio
async def test_a_repo_root_outside_the_allowed_targets_is_refused(tmp_path: Path) -> None:
    """Integration may only stage into repositories the session declared."""
    repo, base = _repo(tmp_path)
    other, other_base = _named_repo(tmp_path, "other", "third.py")
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        other,
        other_base,
        kernel_name="foreign",
        kernel_path="third.py",
        patch=_patch(other, "third.py", "VALUE = 9\n"),
    )

    async def _validate(_publication):
        raise AssertionError("a foreign repository must never reach validation")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.kept_count == 0
    assert [r.status for r in summary.results] == ["skipped_invalid"]


@pytest.mark.asyncio
async def test_a_keep_carries_the_server_settings_its_validation_measured(tmp_path: Path) -> None:
    """The KEEP is only reproducible with the args and envs it was measured under."""
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="tuned",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )

    async def _validate(_publication):
        return {
            "decision": "KEEP",
            "new_tput": 120.0,
            "gain_pct": 20.0,
            "extra_server_args": "--enable-foo",
            "extra_envs": {"FOO": "1"},
        }

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    state = _state(session_dir, repo)
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=state,
        validator=_validate,
    )

    assert summary.kept_count == 1
    assert state.current_best["extra_server_args"] == "--enable-foo"
    assert state.current_best["extra_envs"] == {"FOO": "1"}


@pytest.mark.asyncio
async def test_a_commit_that_lands_is_reported_even_if_the_ledger_write_fails(tmp_path: Path) -> None:
    """The Git commit and the SharedState write are not one transaction.

    Losing the ledger write must not be reported as a lost patch: the commit is
    in the repository either way, and calling it anything but ``kept`` would send
    the next patch at a HEAD the result says does not exist.
    """
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="landed",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )

    async def _validate(_publication):
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    state = _state(session_dir, repo)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("state file is read-only")

    state.save = _boom  # type: ignore[method-assign]
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=state,
        validator=_validate,
    )

    assert summary.kept_count == 1
    result = summary.results[0]
    assert result.status == "kept"
    assert "SharedState recording failed" in (result.reason or "")
    # The commit is real regardless of what the ledger managed to record.
    assert int(_git(repo, "rev-list", "--count", "HEAD")) == 2
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def _patch_adding_a_file(repo: Path, modified: str, created: str) -> str:
    """A diff that both edits a tracked file and introduces a new one."""
    original = (repo / modified).read_text(encoding="utf-8")
    (repo / modified).write_text("VALUE = 2\n", encoding="utf-8")
    (repo / created).write_text("HELPER = True\n", encoding="utf-8")
    _git(repo, "add", "-N", created)
    patch = _git(repo, "diff", "--binary", "--", modified, created)
    (repo / modified).write_text(original, encoding="utf-8")
    _git(repo, "rm", "--quiet", "--cached", "--force", created)
    (repo / created).unlink()
    return patch + "\n"


@pytest.mark.asyncio
async def test_a_revert_the_diff_cannot_undo_restores_what_head_knows(tmp_path: Path) -> None:
    """A validation that edits the source leaves a patch reverse-apply refuses.

    That is the state a partially applied patch is in too. The fallback restores
    every path HEAD still has a version of and leaves the ones the patch created
    alone -- by then nothing can prove such a file was not already the operator's.
    """
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="creates_a_file",
        kernel_path="first.py",
        patch=_patch_adding_a_file(repo, "first.py", "helper.py"),
    )

    async def _validate(_publication):
        # Something in the E2E path rewrites the source under the patch.
        (repo / "first.py").write_text("VALUE = 999  # instrumented\n", encoding="utf-8")
        return {"decision": "REVERT", "reason": "no gain"}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.kept_count == 0
    result = summary.results[0]
    assert result.status == "reverted_e2e_failed"
    assert "left files the patch created in place: helper.py" in (result.reason or "")
    # The tracked file is back at its committed content...
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    # ...and the file the patch created is still on disk, unstaged.
    assert (repo / "helper.py").exists()
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""


@pytest.mark.asyncio
async def test_dirt_on_a_file_the_patch_never_touches_does_not_block_it(tmp_path: Path) -> None:
    """Hyperloom dirties the framework tree itself; a repo-wide check never passes.

    ``ensure_sglang_patched_for_tracelens`` and its ck-blockscale sibling patch
    the serving source in place and leave it uncommitted for the whole session,
    and every other lane leaves its own KEEP uncommitted too. A repository-wide
    admission check therefore refuses every patch that reaches it -- which is how
    a 4.5-hour campaign's only micro-validated patch was thrown away on a file it
    never opened.
    """
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="target",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    # Stand-in for the in-place instrumentation: a tracked file the patch does
    # not name, modified and left uncommitted.
    (repo / "second.py").write_text("INSTRUMENTED = True\n", encoding="utf-8")

    async def _validate(_publication):
        return {"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.kept_count == 1
    assert summary.results[0].status == "kept"
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    # The unrelated edit is still there, uncommitted and unharmed.
    assert (repo / "second.py").read_text(encoding="utf-8") == "INSTRUMENTED = True\n"
    assert _git(repo, "status", "--porcelain", "--untracked-files=no", "--", "second.py").strip()


@pytest.mark.asyncio
async def test_dirt_on_a_file_the_patch_does_touch_still_blocks_it(tmp_path: Path) -> None:
    """Scoping the check narrows it; it does not remove it.

    An uncommitted edit on a path the patch modifies cannot be told apart from
    the patch's own change afterwards, so the patch is still refused.
    """
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="target",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    (repo / "first.py").write_text("VALUE = 41  # someone else's edit\n", encoding="utf-8")

    async def _validate(_publication):
        raise AssertionError("a patch on a dirty path must never reach validation")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.kept_count == 0
    result = summary.results[0]
    assert result.status == "skipped_dirty_worktree"
    assert "first.py" in (result.reason or "")
    # The other edit is left exactly as it was.
    assert (repo / "first.py").read_text(encoding="utf-8") == "VALUE = 41  # someone else's edit\n"


@pytest.mark.asyncio
async def test_a_run_that_admitted_nothing_does_not_report_as_completed(tmp_path: Path) -> None:
    """A phase that dropped every patch at the door must not read as a clean run."""
    repo, base = _repo(tmp_path)
    patches = tmp_path / "cycle" / "result" / "patches"
    _publish(
        patches,
        repo,
        base,
        kernel_name="target",
        kernel_path="first.py",
        patch=_patch(repo, "first.py", "VALUE = 2\n"),
    )
    (repo / "first.py").write_text("VALUE = 41\n", encoding="utf-8")

    async def _validate(_publication):
        raise AssertionError("unreachable")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    summary = await integrate_controller_patches(
        patches_root=patches,
        session_dir=session_dir,
        shared_state=_state(session_dir, repo),
        validator=_validate,
    )

    assert summary.status == "no_patch_admitted"
    assert summary.kept_count == 0
    assert summary.skipped_count == 1
