# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Git revisions and disposable workspaces for Ave optimization."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

from hyperloom_common.git_safety import app_git, repo_git

GIT_REV_PARSE = "rev-parse"
COMMIT_IDENTITY = ("-c", "user.name=Ave", "-c", "user.email=ave@localhost")


class WorkspaceError(ValueError):
    """An agent changed something outside the step's editable source files."""


class StepWorkspace:
    """Check direct agent edits and restore a disposable worktree after a session."""

    def __init__(self, workspace: Path, revision: str, driver: Path):
        self.workspace = workspace
        self.revision = revision
        self.driver = driver
        self.driver_source = driver.read_bytes()
        self.untracked = self.untracked_files()

    def untracked_files(self) -> set[str]:
        listed = app_git(self.workspace, "ls-files", "--others", "--exclude-standard", "-z", check=True)
        return set(listed.stdout.split("\0")[:-1])

    def validate(self, editable_files: tuple[str, ...]) -> list[str]:
        """Require unchanged measurement inputs before the host accepts any timing."""
        if self.driver.is_symlink() or not self.driver.is_file() or self.driver.read_bytes() != self.driver_source:
            raise WorkspaceError("agent changed the protected driver")
        if app_git(self.workspace, GIT_REV_PARSE, "HEAD", check=True).stdout.strip() != self.revision:
            raise WorkspaceError("agent changed Git HEAD; the host owns commits")
        changed = app_git(
            self.workspace,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--name-only",
            "-z",
            self.revision,
            check=True,
        ).stdout.split("\0")[:-1]
        if not set(changed) <= set(editable_files) or self.untracked_files() - self.untracked:
            raise WorkspaceError("agent changed files outside the step's source scope")
        if any((self.workspace / name).is_symlink() or not (self.workspace / name).is_file() for name in changed):
            raise WorkspaceError("editable sources must remain regular files")
        return changed

    def restore(self, revision: str) -> None:
        """Keep only the host-verified revision and remove new untracked session files."""
        added = self.untracked_files() - self.untracked
        app_git(self.workspace, "checkout", "--detach", "--force", revision, check=True)
        if added:
            app_git(self.workspace, "clean", "-fd", "--", *added, check=True)
        if self.driver.is_symlink():
            self.driver.unlink()
        self.driver.write_bytes(self.driver_source)


def snapshot(repo: Path) -> str:
    """Capture tracked serving sources without changing their branch, index or worktree."""
    revision = repo_git(repo, *COMMIT_IDENTITY, "stash", "create").stdout.strip()
    return revision or repo_git(repo, GIT_REV_PARSE, "HEAD").stdout.strip()


@contextmanager
def checkout(repo: Path, revision: str, destination: Path) -> Iterator[Path]:
    """Optimize a detached revision without changing the serving checkout."""
    repo_git(repo, "worktree", "add", "--detach", str(destination), revision)
    try:
        yield destination
    finally:
        repo_git(repo, "worktree", "remove", "--force", str(destination))
