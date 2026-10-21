# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Git revisions and disposable workspaces for Ave optimization."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator

GIT_REV_PARSE = "rev-parse"


def git(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run an Ave workspace operation, surfacing Git failures."""
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=120)


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
        return set(
            filter(None, git("ls-files", "--others", "--exclude-standard", "-z", cwd=self.workspace).stdout.split("\0"))
        )

    def validate(self, editable_files: tuple[str, ...]) -> list[str]:
        """Require unchanged measurement inputs before the host accepts any timing."""
        if self.driver.is_symlink() or not self.driver.is_file() or self.driver.read_bytes() != self.driver_source:
            raise WorkspaceError("agent changed the protected driver")
        if git(GIT_REV_PARSE, "HEAD", cwd=self.workspace).stdout.strip() != self.revision:
            raise WorkspaceError("agent changed Git HEAD; the host owns commits")
        changed = list(
            filter(
                None,
                git("diff", "--no-renames", "--name-only", "-z", self.revision, cwd=self.workspace).stdout.split("\0"),
            )
        )
        if not set(changed) <= set(editable_files) or self.untracked_files() - self.untracked:
            raise WorkspaceError("agent changed files outside the step's source scope")
        if any((self.workspace / name).is_symlink() or not (self.workspace / name).is_file() for name in changed):
            raise WorkspaceError("editable sources must remain regular files")
        return changed

    def restore(self, revision: str) -> None:
        """Keep only the host-verified revision and remove new untracked session files."""
        added = self.untracked_files() - self.untracked
        git("checkout", "--detach", "--force", revision, cwd=self.workspace)
        for name in added:
            path = self.workspace / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        if self.driver.is_symlink():
            self.driver.unlink()
        self.driver.write_bytes(self.driver_source)


def snapshot(repo: Path) -> str:
    """Capture tracked serving sources without changing their branch, index or worktree."""
    head = git(GIT_REV_PARSE, "HEAD", cwd=repo).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="ave-index-") as directory:
        index = Path(directory) / "index"
        serving_index = git(GIT_REV_PARSE, "--path-format=absolute", "--git-path", "index", cwd=repo).stdout.strip()
        shutil.copyfile(serving_index, index)
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        git("add", "--update", "--", ".", cwd=repo, env=env)
        tree = git("write-tree", cwd=repo, env=env).stdout.strip()
    if tree == git(GIT_REV_PARSE, f"{head}^{{tree}}", cwd=repo).stdout.strip():
        return head
    return git(
        "-c",
        "user.name=Hyperloom",
        "-c",
        "user.email=hyperloom@localhost",
        "commit-tree",
        tree,
        "-p",
        head,
        "-m",
        "Snapshot serving sources for Ave",
        cwd=repo,
    ).stdout.strip()


@contextmanager
def checkout(repo: Path, revision: str, destination: Path) -> Iterator[Path]:
    """Optimize a detached revision without changing the serving checkout."""
    git("worktree", "add", "--detach", str(destination), revision, cwd=repo)
    try:
        yield destination
    finally:
        git("worktree", "remove", "--force", str(destination), cwd=repo)
