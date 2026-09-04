# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name
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
async def test_dirty_integration_worktree_is_skipped_without_cleanup(tmp_path: Path) -> None:
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
    (repo / "second.py").write_text("USER_CHANGE = True\n", encoding="utf-8")

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
    assert (repo / "second.py").read_text(encoding="utf-8") == "USER_CHANGE = True\n"


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
