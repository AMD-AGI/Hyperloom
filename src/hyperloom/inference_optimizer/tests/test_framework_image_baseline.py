# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework source trees are pinned at their served state by install.sh."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from hyperloom.orchestrator.actions.executors._git import _git_head_sha
from hyperloom.orchestrator.actions.executors._patch_git import _git_reset_clean, _git_stash_if_dirty
from hyperloom.orchestrator.specialists.subprocess_ import _setup_worktree

_INSTALL = Path(__file__).resolve().parents[1] / "assets" / "install.sh"
_FUNCTIONS = (
    "commit_image_baseline",
    "_image_baseline_exclude_runtime_caches",
    "pin_image_baseline",
    "pin_framework_image_baselines",
)
_CONSTANTS = ("IMAGE_BASELINE_RUNTIME_CACHES", "IMAGE_BASELINE_UNTRACKED_LIMIT_BYTES")


def _script(body: str) -> str:
    text = _INSTALL.read_text(encoding="utf-8")
    chunks = [re.search(rf"^{name}=.*$", text, re.MULTILINE).group() for name in _CONSTANTS]
    chunks += [re.search(rf"^{name}\(\) \{{.*?^\}}$", text, re.MULTILINE | re.DOTALL).group() for name in _FUNCTIONS]
    prelude = 'log() { :; }; die() { echo "$*" >&2; return 1; }; DRY_RUN=0; CHECK_ONLY=0'
    return "\n".join(["set -euo pipefail", prelude, *chunks, body])


def _pin(roots: str, *, extra: str = "", cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    script = _script(f'{extra}\nFRAMEWORK_SOURCE_ROOTS="{roots}"; pin_framework_image_baselines')
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=False, cwd=cwd)


_GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e"}
_GIT_ENV.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=True, env=_GIT_ENV
    ).stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _image_repo(tmp_path: Path) -> Path:
    """A checkout whose working tree differs from HEAD the way a prebuilt image's does."""
    repo = tmp_path / "sglang"
    _write(repo / ".gitignore", "*.so\n")
    _write(repo / "pkg" / "edited.py", "edited = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "upstream")
    _write(repo / "pkg" / "edited.py", "edited = 2\n")
    _write(repo / "pkg" / "shipped.py", "shipped = 1\n")
    _write(repo / "pkg" / "native.so", "binary")
    _write(repo / "aiter" / "jit" / "flydsl_cache" / "entry.pkl", "cache")
    return repo


def _pinned(tmp_path: Path) -> Path:
    repo = _image_repo(tmp_path)
    assert _pin(str(repo / "pkg")).returncode == 0
    return repo


def test_the_served_state_becomes_head(tmp_path: Path) -> None:
    repo = _image_repo(tmp_path)
    upstream = _git(repo, "rev-parse", "HEAD")

    result = _pin(f"{repo / 'pkg'}:{tmp_path / 'not-a-checkout'}")

    assert result.returncode == 0, result.stderr
    assert _git(repo, "rev-parse", "HEAD^") == upstream
    assert _git(repo, "rev-parse", "refs/hyperloom/image-baseline") == _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines() == [
        ".gitignore",
        "pkg/edited.py",
        "pkg/shipped.py",
    ]
    assert _git(repo, "show", "HEAD:pkg/edited.py") == "edited = 2"
    assert (repo / "aiter" / "jit" / "flydsl_cache" / "entry.pkl").exists()


def test_identical_images_pin_to_the_same_commit(tmp_path: Path) -> None:
    first = _image_repo(tmp_path)
    second = tmp_path / "copy"
    shutil.copytree(first, second)

    assert _pin(f"{first}:{second}").returncode == 0

    assert _git(first, "rev-parse", "HEAD") == _git(second, "rev-parse", "HEAD")


def test_a_pinned_tree_is_left_alone(tmp_path: Path) -> None:
    repo = _pinned(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    _write(repo / "pkg" / "edited.py", "edited = 3\n")

    assert _pin(str(repo)).returncode == 0

    assert _git(repo, "rev-parse", "HEAD") == head


def test_check_only_leaves_the_tree_alone(tmp_path: Path) -> None:
    repo = _image_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    assert _pin(str(repo), extra="CHECK_ONLY=1").returncode == 0

    assert _git(repo, "rev-parse", "HEAD") == head


def test_an_empty_root_does_not_pin_the_working_directory(tmp_path: Path) -> None:
    repo = _image_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    assert _pin("", cwd=repo).returncode == 0

    assert _git(repo, "rev-parse", "HEAD") == head


def test_a_tree_with_unmerged_paths_is_refused(tmp_path: Path) -> None:
    repo = _image_repo(tmp_path)
    _git(repo, "checkout", "-qb", "other")
    _git(repo, "commit", "-qam", "other")
    _git(repo, "checkout", "-q", "-")
    _write(repo / "pkg" / "edited.py", "edited = main\n")
    _git(repo, "commit", "-qam", "main")
    subprocess.run(["git", "-C", str(repo), "merge", "-q", "other"], capture_output=True, check=False, env=_GIT_ENV)
    assert _git(repo, "ls-files", "--unmerged")
    head = _git(repo, "rev-parse", "HEAD")

    result = _pin(str(repo))

    assert result.returncode != 0
    assert "unmerged" in result.stderr
    assert _git(repo, "rev-parse", "HEAD") == head


def test_untracked_content_over_the_limit_is_refused(tmp_path: Path) -> None:
    repo = _image_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")

    result = _pin(str(repo), extra="IMAGE_BASELINE_UNTRACKED_LIMIT_BYTES=5")

    assert result.returncode != 0
    assert _git(repo, "rev-parse", "HEAD") == head


def test_a_candidate_reverts_to_the_served_tree(tmp_path: Path) -> None:
    repo = _pinned(tmp_path)

    assert _git_stash_if_dirty(repo) == ("clean", "")
    _write(repo / "pkg" / "edited.py", "edited = candidate\n")
    ok, error = _git_reset_clean(repo, _git_head_sha(repo))

    assert ok, error
    assert (repo / "pkg" / "edited.py").read_text() == "edited = 2\n"
    assert (repo / "pkg" / "shipped.py").read_text() == "shipped = 1\n"


def test_a_specialist_worktree_is_cut_from_the_served_tree(tmp_path: Path) -> None:
    repo = _pinned(tmp_path)

    worktree, error = _setup_worktree(repo, tmp_path / "specialist", "specialist/branch")

    assert error == ""
    assert (worktree / "pkg" / "edited.py").read_text() == "edited = 2\n"
    assert (worktree / "pkg" / "shipped.py").read_text() == "shipped = 1\n"
