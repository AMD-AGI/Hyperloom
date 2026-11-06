# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom_common.git_safety.app_git``: no program a worktree or its repository's config names runs."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from hyperloom_common.git_safety import app_git

_COMMIT = ("-c", "user.email=t@t.local", "-c", "user.name=t", "commit", "-qm")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A repository holding ``mod.py`` and a linked worktree of it, the shape the harvest diffs."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "mod.py").write_text("old = 1\n", encoding="utf-8")
    _git(tmp_path, "init", "-q", str(base))
    _git(base, "add", "-A")
    _git(base, *_COMMIT, "base")
    worktree = tmp_path / "worktree"
    _git(base, "worktree", "add", "-q", "-b", "task", str(worktree))
    return base, worktree


def _leaves(canary: Path, then: str) -> str:
    """A shell command that creates ``canary``, then runs ``then``."""
    return "sh -c " + shlex.quote(f"touch {shlex.quote(str(canary))}; {then}")


# Each pair is a driver program and a command that runs it on a changed path, so every case reached the program before.
_FILTER_RUNS = [
    ("clean", ("add", "-A", "-N")),
    ("clean", ("diff", "--no-ext-diff", "HEAD")),
    ("clean", ("status", "--porcelain")),
    ("smudge", ("checkout", "--", "mod.py")),
    ("process", ("add", "-A", "-N")),
    ("process", ("diff", "--no-ext-diff", "HEAD")),
    ("process", ("status", "--porcelain")),
    ("process", ("checkout", "--", "mod.py")),
]


@pytest.mark.parametrize(("program", "command"), _FILTER_RUNS, ids=[f"{p}-{c[0]}" for p, c in _FILTER_RUNS])
def test_a_filter_driver_the_repository_configures_never_runs(tmp_path, program, command):
    base, worktree = _repo_with_worktree(tmp_path)
    canary = tmp_path / "ran"
    _git(base, "config", f"filter.evil.{program}", _leaves(canary, "cat"))
    (worktree / ".gitattributes").write_text("* filter=evil\n", encoding="utf-8")
    (worktree / "mod.py").write_text("new = 2\n", encoding="utf-8")
    (worktree / "added.py").write_text("fresh = 3\n", encoding="utf-8")

    done = app_git(worktree, *command)

    assert not canary.exists()
    assert done.returncode == 0, done.stderr


def test_a_filter_driver_with_a_dotted_name_is_blanked_too(tmp_path):
    base, worktree = _repo_with_worktree(tmp_path)
    canary = tmp_path / "ran"
    _git(base, "config", "filter.Evil.Name.clean", _leaves(canary, "cat"))
    (worktree / ".gitattributes").write_text("* filter=Evil.Name\n", encoding="utf-8")
    (worktree / "mod.py").write_text("new = 2\n", encoding="utf-8")

    done = app_git(worktree, "diff", "--no-ext-diff", "HEAD")

    assert not canary.exists()
    assert "+new = 2" in done.stdout


def test_a_filter_driver_the_command_line_cannot_blank_runs_nothing(tmp_path):
    """``-c filter.a=b.clean=`` sets ``filter.a``; the attribute ``filter=a=b`` still names the driver ``a=b``."""
    base, worktree = _repo_with_worktree(tmp_path)
    canary = tmp_path / "ran"
    _git(base, "config", "filter.a=b.clean", _leaves(canary, "cat"))
    (worktree / ".gitattributes").write_text("* filter=a=b\n", encoding="utf-8")
    (worktree / "mod.py").write_text("new = 2\n", encoding="utf-8")

    with pytest.raises(OSError, match="holds '='"):
        app_git(worktree, "diff", "--no-ext-diff", "HEAD")

    assert not canary.exists()


# A promisor remote is fetched from as soon as git needs an object the repository lacks.
_REMOTES = {
    "ext": ({"protocol.ext.allow": "always"}, "transport 'ext' not allowed"),
    "ssh": ({"remote.origin.url": "ssh://example.invalid/r.git"}, "Could not read from remote repository"),
}


@pytest.mark.parametrize(
    "command", [("diff", "--no-ext-diff", "HEAD"), ("checkout", "--", "big.txt")], ids=["diff", "checkout"]
)
@pytest.mark.parametrize("transport", sorted(_REMOTES))
def test_a_remote_the_repository_configures_is_never_run(tmp_path, transport, command):
    base, worktree = _repo_with_worktree(tmp_path)
    canary = tmp_path / "ran"
    (worktree / "big.txt").write_text("x\n", encoding="utf-8")
    _git(worktree, "add", "big.txt")
    _git(worktree, *_COMMIT, "big")
    blob = _git(worktree, "rev-parse", "HEAD:big.txt").strip()
    (base / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    (worktree / "big.txt").write_text("y\n", encoding="utf-8")
    extra, refused = _REMOTES[transport]
    config = {
        "core.repositoryformatversion": "1",
        "extensions.partialClone": "origin",
        "remote.origin.promisor": "true",
        "remote.origin.url": "ext::sh -c " + f"touch {canary}; exit 1".replace(" ", "% "),
        "core.sshCommand": _leaves(canary, "exit 1"),
        **extra,
    }
    for key, value in config.items():
        _git(base, "config", key, value)

    done = app_git(worktree, *command)

    assert not canary.exists()
    assert refused in done.stderr


def test_a_repository_nested_in_the_tree_is_not_run_in(tmp_path):
    _base, worktree = _repo_with_worktree(tmp_path)
    canary = tmp_path / "ran"
    nested = worktree / "vendored"
    nested.mkdir()
    (nested / "f.txt").write_text("a\n", encoding="utf-8")
    _git(tmp_path, "init", "-q", str(nested))
    _git(nested, "add", "-A")
    _git(nested, *_COMMIT, "nested")
    _git(nested, "config", "filter.inner.clean", _leaves(canary, "cat"))
    (nested / ".gitattributes").write_text("* filter=inner\n", encoding="utf-8")
    (nested / "f.txt").write_text("b\n", encoding="utf-8")

    app_git(worktree, "add", "-A", "-N")
    done = app_git(worktree, "diff", "--no-ext-diff", "HEAD")

    assert not canary.exists()
    assert "Subproject commit" in done.stdout
