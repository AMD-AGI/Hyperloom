# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Give the forge-loop a git workspace over a framework tree, owning none of it.

The loop keeps and reverts with ``git add -u`` and ``git restore``, which only
see TRACKED files, and its commits are its deliverable: it expects a workspace
it may write history into. Fusion cannot hand it a copy, because the benchmark
and the serving gate import the framework from its real install path, so it
edits the live tree and isolates the git side instead.

``git init --separate-git-dir`` leaves a one-line ``.git`` pointer file in the
tree and keeps every object under the run's output directory, so git resolves
the shadow from the tree itself and the location never reaches a child process.
A tree that already owns ``.git`` (an editable checkout) cannot take a pointer
without losing its own repository, so that case routes through
``GIT_DIR``/``GIT_WORK_TREE`` instead, which the agent does inherit.

Only the framework package is indexed. An installed framework sits beside
gigabytes of unrelated wheels, and the exclude that keeps them out of the index
also keeps git from walking them when it looks for untracked files.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.loop.path_ownership import runtime_gitignore_globs

log = logging.getLogger("forge_fusion")

_GIT_TIMEOUT_SEC = 120

#: Branch the baseline is committed onto. The loop refuses a workspace on an
#: unnamed, ``main`` or ``master`` branch, and a fresh repository is on one.
SHADOW_BRANCH = "forge-fusion"

# Whitelist: exclude every top-level entry, then re-admit the indexed ones. Git
# will not descend into an excluded directory, so re-admitting the directory
# itself is what makes this work. Artifact patterns come last to apply inside it.
_EXCLUDE_HEADER = "/*\n"
_EXCLUDE_ARTIFACTS = "".join(f"{glob}\n" for glob in runtime_gitignore_globs())


def _git(repo: str, *args: str, env: dict[str, str], timeout: int = _GIT_TIMEOUT_SEC) -> subprocess.CompletedProcess:
    """Run one git command in ``repo``; ``env`` overlays the process environment."""
    return git(*args, cwd=repo, check=False, timeout=timeout, env=env)


def _admit(root: Path, relative: str) -> str:
    """A negated exclude line re-admitting ``relative``, slashed if it is a dir."""
    return f"!/{relative}{'/' if (root / relative).is_dir() else ''}"


def _relative(root: Path, path: str) -> str:
    """``path`` as a root-relative posix path."""
    return Path(path).resolve().relative_to(root).as_posix()


def _index_scope(repo_root: str, source_file: str) -> str:
    """The one entry under ``repo_root`` worth indexing: the framework package.

    Taken as the first path component of ``source_file`` relative to
    ``repo_root``, so a PEP 420 namespace package resolves like a conventional
    one. That prefix is also what a canonical KB source path is anchored to,
    which keeps a diff taken here applicable where the KB later replays it.
    Returns "" when the source does not live under the root.
    """
    if not repo_root or not source_file:
        return ""
    try:
        rel = Path(source_file).resolve().relative_to(Path(repo_root).resolve())
    except (OSError, ValueError):
        return ""
    return rel.parts[0] if rel.parts else ""


@dataclass
class ShadowRepo:
    """A git repository over the framework tree whose history nobody else owns.

    ``root`` is the work tree and what the loop receives as ``--workspace``;
    ``git_dir`` holds every object and ref, under the run's output directory.
    ``pointer_path`` is the ``.git`` file this wrote, removed on disposal; it is
    empty on the editable-checkout path, where ``env`` carries GIT_DIR instead.
    """

    root: str
    git_dir: str
    base_commit: str
    env: dict[str, str] = field(default_factory=dict)
    created_paths: tuple[str, ...] = ()
    pointer_path: str = ""

    def reset_to_base(self) -> bool:
        """Put the framework tree back as the campaign found it.

        ``clean`` takes no pathspec because the exclude file lists only cache
        directories safe to leave behind; artefacts that affect measurement
        (compiled extensions, build output) are tracked and restored by reset.
        """
        for args in (("reset", "--hard", "-q", self.base_commit), ("clean", "-fdq")):
            result = _git(self.root, *args, env=self.env)
            if result.returncode != 0:
                log.error(
                    "could not restore %s with git %s: %s",
                    self.root,
                    args[0],
                    (result.stderr or result.stdout).strip(),
                )
                return False
        return True

    def dispose(self) -> None:
        """Drop the repository, and the placeholders the author never wrote into.

        Only the EMPTY placeholders: one with content holds a fused kernel the
        export still has to read, and the run's own restore removes those after.
        A leftover git dir is inert scratch under the output directory, and this
        runs in a ``finally`` where raising would mask the campaign's own error.
        """
        for path in self.created_paths:
            target = Path(path)
            if target.is_file() and target.stat().st_size == 0:
                target.unlink()
        if self.pointer_path:
            Path(self.pointer_path).unlink(missing_ok=True)
        shutil.rmtree(self.git_dir, ignore_errors=True)


def ensure_git_workspace(
    repo_root: str, source_file: str, *, git_dir: str, extra_paths: tuple[str, ...] = ()
) -> ShadowRepo | None:
    """Build a repository over ``repo_root`` whose git data lives in ``git_dir``.

    ``extra_paths`` are files the campaign must find already TRACKED -- the
    placeholder the author writes its fused kernel into. Each is created empty,
    overwriting whatever a crashed earlier run left there, because the loop
    stages a keep with ``git add -u`` and a file untracked at the base commit
    can never enter a commit, so the kept state would not match what was
    benchmarked.

    Returns None when no workspace could be established, which the caller must
    treat as "the loop cannot keep or revert here".
    """
    if not repo_root or not Path(repo_root).is_dir():
        return None
    scope = _index_scope(repo_root, source_file)
    if not scope:
        log.error("%s does not live under %s; no shadow workspace", source_file, repo_root)
        return None

    root = Path(repo_root).resolve()
    git_path = Path(git_dir)
    pointer = root / ".git"
    # --separate-git-dir MOVES an existing repository into the target, and
    # dispose() would then delete the developer's history, so a tree that owns
    # .git is routed through the environment the agent inherits instead.
    detached = pointer.exists()
    if detached:
        log.warning(
            "%s is a git checkout; the shadow routes through GIT_DIR=%s, which the forge-loop agent inherits",
            root,
            git_dir,
        )
    env = {"GIT_DIR": str(git_path), "GIT_WORK_TREE": str(root)} if detached else {}
    init = ("init", "-q") if detached else ("init", "-q", f"--separate-git-dir={git_path}")

    try:
        shutil.rmtree(git_path, ignore_errors=True)
        git_path.parent.mkdir(parents=True, exist_ok=True)
        for path in extra_paths:
            placeholder = Path(path)
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            placeholder.write_text("", encoding="utf-8")
        # A placeholder normally sits inside the package the scope admits, but a
        # framework whose source is directly in the export root has no such
        # package, so each is named too.
        indexed = list(dict.fromkeys([scope, *(_relative(root, p) for p in extra_paths)]))

        # The exclude goes into the git dir, which only exists once init has run.
        result = _git(str(root), *init, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"git init failed: {(result.stderr or result.stdout).strip()}")
        (git_path / "info").mkdir(parents=True, exist_ok=True)
        (git_path / "info" / "exclude").write_text(
            _EXCLUDE_HEADER + "".join(f"{_admit(root, entry)}\n" for entry in indexed) + _EXCLUDE_ARTIFACTS,
            encoding="utf-8",
        )
        for args in (
            ("config", "user.email", "forge-fuse@localhost"),
            ("config", "user.name", "forge-fuse"),
            ("add", "--", *indexed),
            ("commit", "-q", "-m", "fusion baseline", "--no-gpg-sign"),
            # After the commit, so the branch points at it and not at an unborn
            # HEAD. ``git init -b`` needs a newer git than this runs on.
            ("checkout", "-q", "-b", SHADOW_BRANCH),
            ("rev-parse", "HEAD"),
        ):
            result = _git(str(root), *args, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
        base_commit = result.stdout.strip()  # the rev-parse above
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        log.error("could not initialize a shadow repo over %s: %s", repo_root, exc)
        if not detached:
            pointer.unlink(missing_ok=True)
        shutil.rmtree(git_path, ignore_errors=True)
        for path in extra_paths:
            Path(path).unlink(missing_ok=True)
        return None

    log.info("shadow repo over %s indexed %s", repo_root, ", ".join(indexed))
    return ShadowRepo(
        root=str(root),
        git_dir=str(git_path),
        base_commit=base_commit,
        env=env,
        created_paths=tuple(extra_paths),
        pointer_path="" if detached else str(pointer),
    )
