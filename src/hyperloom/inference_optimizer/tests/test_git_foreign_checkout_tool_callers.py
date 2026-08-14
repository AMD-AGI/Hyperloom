# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kernel agent's Forge backend must survive a foreign-owned checkout.

``forge_submit.py`` builds its own argv instead of going through
``executors/_git.py``, and the repo it drives is the framework source tree the
documented container recipe bind-mounts. Git refuses every call on it, and the
refusals read as facts about the repository: "no default branch", "HEAD is
unresolvable", "the best commit does not contain the source it was validated
on".

``GIT_TEST_ASSUME_DIFFERENT_OWNER`` is git's own hook for this path, so the
tests need no root and no foreign-owned directory. The repo is built first and
only then declared foreign, otherwise the fixture could not commit.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from hyperloom.agents.kernel.tools.backends import forge_submit

_IDENT = ("-c", "user.email=t@t.local", "-c", "user.name=t")


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True, capture_output=True)
    (path / "kern.py").write_text("committed base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), *_IDENT, "commit", "-q", "-m", "base"],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def foreign_repo(tmp_path, monkeypatch):
    """A real repo, turned foreign only after setup so init/commit can run."""
    repo = _init_repo(tmp_path / "repo")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    return repo


def test_the_default_branch_is_not_reported_as_missing(foreign_repo):
    """A refusal here reads as "this repo has no default branch".

    ``_prepare_inplace`` uses the answer to recover from a leftover forge
    branch, and an empty one makes it skip the run instead.
    """
    assert forge_submit._default_branch(str(foreign_repo)) == "main"


def test_a_worktree_is_prepared_on_a_foreign_owned_repo(foreign_repo, tmp_path):
    """The isolated worktree is how forge avoids mutating the live repo."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    prepared = forge_submit._prepare_worktree(
        str(foreign_repo / "kern.py"),
        str(foreign_repo),
        output_dir,
        "forge/test/kern-abc123",
    )

    assert prepared is not None
    worktree, kernel_file, base_commit = prepared
    assert Path(worktree).is_dir()
    assert Path(kernel_file).read_text(encoding="utf-8") == "committed base\n"
    assert base_commit


def test_the_validated_best_commit_is_exported_from_a_foreign_owned_repo(foreign_repo, tmp_path):
    """The export reads blobs with a bare ``subprocess.run``, not ``_run``.

    Refused, it reports a validated commit as not containing the source it was
    validated on, which aborts a run that had already produced its result.
    """
    guard = ["-c", f"safe.directory={foreign_repo}"]
    base_commit = subprocess.run(
        ["git", *guard, "-C", str(foreign_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (foreign_repo / "kern.py").write_text("optimized\n", encoding="utf-8")
    subprocess.run(["git", *guard, "-C", str(foreign_repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", *guard, "-C", str(foreign_repo), *_IDENT, "commit", "-q", "-m", "best"],
        check=True,
        capture_output=True,
    )
    best_commit = subprocess.run(
        ["git", *guard, "-C", str(foreign_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    primary, changed = forge_submit._export_best_artifacts(
        str(foreign_repo),
        base_commit,
        str(foreign_repo / "kern.py"),
        str(foreign_repo / "kern.py"),
        output_dir,
        best_commit=best_commit,
    )

    assert Path(primary).read_text(encoding="utf-8") == "optimized\n"
    assert changed == ["kern.py"]
    assert (output_dir / "optimized_versions" / "forge.patch").read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Static guards: forge_submit builds ~40 argv across 30 call sites
# ---------------------------------------------------------------------------
def _forge_submit_ast() -> tuple[ast.Module, dict[int, str]]:
    tree = ast.parse(Path(forge_submit.__file__).read_text(encoding="utf-8"))
    owners: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                owners.setdefault(id(child), node.name)
    return tree, owners


def test_forge_submit_builds_every_git_argv_through_the_guard():
    """Too many call sites to remember per site, so none of them may spell out ``git``."""
    tree, owners = _forge_submit_ast()

    offenders = [
        (owners.get(id(node), "<module>"), node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and node.elts
        and isinstance(node.elts[0], ast.Constant)
        and node.elts[0].value == "git"
        and owners.get(id(node)) != "_git_argv"
    ]

    assert not offenders, f"git argv built outside _git_argv: {offenders}"


def test_forge_submit_stays_import_light():
    """``tools/`` scripts run standalone on remote nodes, without ``hyperloom``.

    A module-level import breaks that silently, so the git guard is imported
    inside the helper and falls back to the plain argv when it is unavailable.
    """
    tree, owners = _forge_submit_ast()

    module_level: list[str] = []
    for node in ast.walk(tree):
        if owners.get(id(node)) is not None:
            continue
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hyperloom"):
            module_level.append(node.module)
        elif isinstance(node, ast.Import):
            module_level += [alias.name for alias in node.names if alias.name.startswith("hyperloom")]

    assert not module_level, f"forge_submit must not import hyperloom at module level: {module_level}"
