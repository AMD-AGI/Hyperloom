# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Commits into a campaign workspace must carry a git identity.

A container host has no domain for git to guess an address from, so a commit
into a workspace holding no ``user.name``/``user.email`` names no author and
fails. The identity is written repo-locally once, and only as a fallback: the
workspace is the operator's own repository, so an identity it already carries
must survive.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

from kernelforge.knowledge import experience_integration
from kernelforge.loop import task_preparer

FORGE_IDENTITY = "KernelForge <kernel-forge@localhost>"


def _no_identity_to_borrow(monkeypatch) -> None:
    """Deny git every identity source except an explicit one.

    ``user.useConfigOnly`` stands in for the container this failed on, where
    the hostname has no domain and git will not invent an address.
    """
    for var in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "EMAIL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.useConfigOnly")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")


def _workspace_without_an_identity(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A real repo carrying history but no persistent user.name/user.email."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("ORIGINAL_KERNEL\n", encoding="utf-8")
    driver = workspace / "driver.py"
    driver.write_text("# driver\n", encoding="utf-8")
    task_preparer._git(workspace, "init", "-q")
    task_preparer._git(workspace, "add", "-A")
    task_preparer._git(
        workspace,
        "-c",
        "user.name=seed",
        "-c",
        "user.email=seed@localhost",
        "commit",
        "-q",
        "-m",
        "seed",
    )
    return workspace, kernel, driver


def _conforming_preparation(monkeypatch, driver: Path) -> None:
    """Everything ahead of the commit succeeds, so only the commit is on trial."""

    def fake_materialize_reference(_workspace):
        ref_dir = driver.parent / task_preparer.REFERENCE_SUBDIR
        ref_dir.mkdir(exist_ok=True)
        (ref_dir / "CONTRACT.md").write_text("driver contract\n", encoding="utf-8")
        return ref_dir

    async def fake_agent(**_kwargs):
        driver.write_text("# prepared driver\n", encoding="utf-8")
        return "prepared"

    async def fake_preflight(*_args, **_kwargs):
        return task_preparer.PreflightResult(
            ok=True,
            correctness_ok=True,
            bench_ok=True,
            graph_ok=True,
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", fake_materialize_reference)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", fake_preflight)


def _prepare(tmp_path: Path, workspace: Path, kernel: Path, driver: Path):
    return asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(
                model="test-model",
                experiments_dir=str(tmp_path / "forge_experiments"),
            ),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=[],
            source_files=[str(kernel)],
            preflight=task_preparer.PreflightResult(
                ok=False,
                correctness_ok=False,
                bench_ok=False,
                reasons=["driver missing"],
            ),
        )
    )


def _last_author(workspace: Path) -> str:
    code, author = task_preparer._git(workspace, "log", "-1", "--format=%an <%ae>")
    assert code == 0, author
    return author.strip()


def test_prepass_commit_lands_when_git_cannot_auto_detect_an_identity(tmp_path, monkeypatch):
    """Preparation must not depend on an identity the host happens to offer."""
    _no_identity_to_borrow(monkeypatch)
    workspace, kernel, driver = _workspace_without_an_identity(tmp_path)
    _conforming_preparation(monkeypatch, driver)
    base_sha = task_preparer._git_head(workspace)

    result = _prepare(tmp_path, workspace, kernel, driver)

    assert result.ok is True, result.message
    assert task_preparer._git_head(workspace) != base_sha


def test_prepass_commit_is_attributed_to_forge(tmp_path, monkeypatch):
    """The commit names preparation, not whatever identity was lying around."""
    _no_identity_to_borrow(monkeypatch)
    workspace, kernel, driver = _workspace_without_an_identity(tmp_path)
    _conforming_preparation(monkeypatch, driver)

    result = _prepare(tmp_path, workspace, kernel, driver)

    assert result.ok is True, result.message
    assert _last_author(workspace) == FORGE_IDENTITY


def test_a_configured_identity_is_not_rewritten(tmp_path, monkeypatch):
    """The campaign workspace is the operator's own repository, so its identity stands."""
    _no_identity_to_borrow(monkeypatch)
    workspace, kernel, driver = _workspace_without_an_identity(tmp_path)
    task_preparer._git(workspace, "config", "user.name", "operator")
    task_preparer._git(workspace, "config", "user.email", "operator@example.com")
    _conforming_preparation(monkeypatch, driver)

    result = _prepare(tmp_path, workspace, kernel, driver)

    assert result.ok is True, result.message
    assert _last_author(workspace) == "operator <operator@example.com>"


def test_a_later_commit_into_the_same_repo_also_lands(tmp_path, monkeypatch):
    """The identity must outlast preparation: the warm-start commit is a separate site."""
    _no_identity_to_borrow(monkeypatch)
    workspace, kernel, driver = _workspace_without_an_identity(tmp_path)
    _conforming_preparation(monkeypatch, driver)
    assert _prepare(tmp_path, workspace, kernel, driver).ok is True

    (workspace / "warm.md").write_text("warm start\n", encoding="utf-8")
    sha = experience_integration._git_commit_all(
        str(workspace),
        "warm-start",
        allowed_paths={"warm.md"},
    )

    assert sha == task_preparer._git_head(workspace)
    assert _last_author(workspace) == FORGE_IDENTITY
