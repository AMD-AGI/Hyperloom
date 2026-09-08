# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tool-side git callers must survive a foreign-owned checkout.

The framework explorer's isolation helpers and the pod-side TraceLens patcher
build their own argv instead of going through ``executors/_git.py``, and the
trees they drive are the ones the documented container recipe bind-mounts. Git
refuses every call on them, and the refusals read as facts about the
repository: "no default branch", "HEAD is unresolvable".

``GIT_TEST_ASSUME_DIFFERENT_OWNER`` is git's own hook for this path, so the
tests need no root and no foreign-owned directory. The repo is built first and
only then declared foreign, otherwise the fixture could not commit.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from hyperloom.agents.framework import isolation

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


def _multinode_patcher():
    """Load the pod-side script by path; ``multi_node/scripts`` is not a package."""
    import hyperloom.inference_optimizer as io_pkg

    script = Path(io_pkg.__file__).parent / "multi_node" / "scripts" / "apply_tracelens_patch_multinode.py"
    spec = importlib.util.spec_from_file_location("_tracelens_patcher_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# isolation: its own _run_git, located by cwd rather than -C
# ---------------------------------------------------------------------------
def test_isolation_run_git_locates_the_repo_from_cwd(foreign_repo):
    """``_run_git`` raises on a non-zero exit, so a refusal aborts provisioning.

    It passes ``cwd=`` and never ``-C``, so the executors/_git.py fix could not
    reach it and the helper has to resolve the repo from the working directory.
    """
    isolation._run_git(["git", "status", "--porcelain"], cwd=foreign_repo, timeout_sec=60)
