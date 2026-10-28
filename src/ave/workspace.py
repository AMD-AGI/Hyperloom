# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Git revisions and disposable workspaces for Ave optimization."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import subprocess
from collections.abc import Iterator

GIT_REV_PARSE = "rev-parse"


def git(*args: str, cwd: Path) -> str:
    """Run Git with a fixed commit identity and surface failures."""
    command = ["git", "-c", "user.name=Ave", "-c", "user.email=ave@localhost", *args]
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True, timeout=120).stdout


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
        return set(git("ls-files", "--others", "--exclude-standard", "-z", cwd=self.workspace).split("\0")[:-1])

    def validate(self, editable_files: tuple[str, ...]) -> list[str]:
        """Require unchanged measurement inputs before the host accepts any timing."""
        if self.driver.is_symlink() or not self.driver.is_file() or self.driver.read_bytes() != self.driver_source:
            raise WorkspaceError("agent changed the protected driver")
        if git(GIT_REV_PARSE, "HEAD", cwd=self.workspace).strip() != self.revision:
            raise WorkspaceError("agent changed Git HEAD; the host owns commits")
        changed = git("diff", "--no-renames", "--name-only", "-z", self.revision, cwd=self.workspace).split("\0")[:-1]
        if not set(changed) <= set(editable_files) or self.untracked_files() - self.untracked:
            raise WorkspaceError("agent changed files outside the step's source scope")
        if any((self.workspace / name).is_symlink() or not (self.workspace / name).is_file() for name in changed):
            raise WorkspaceError("editable sources must remain regular files")
        return changed

    def restore(self, revision: str) -> None:
        """Keep only the host-verified revision and remove new untracked session files."""
        added = self.untracked_files() - self.untracked
        git("checkout", "--detach", "--force", revision, cwd=self.workspace)
        if added:
            git("clean", "-fd", "--", *added, cwd=self.workspace)
        if self.driver.is_symlink():
            self.driver.unlink()
        self.driver.write_bytes(self.driver_source)


def snapshot(repo: Path) -> str:
    """Capture tracked serving sources without changing their branch, index or worktree."""
    revision = git("stash", "create", cwd=repo).strip()
    return revision or git(GIT_REV_PARSE, "HEAD", cwd=repo).strip()


@contextmanager
def checkout(repo: Path, revision: str, destination: Path) -> Iterator[Path]:
    """Optimize a detached revision without changing the serving checkout."""
    git("worktree", "add", "--detach", str(destination), revision, cwd=repo)
    try:
        yield destination
    finally:
        git("worktree", "remove", "--force", str(destination), cwd=repo)
