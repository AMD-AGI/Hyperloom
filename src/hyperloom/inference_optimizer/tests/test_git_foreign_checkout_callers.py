# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Callers that reach for git directly must survive a foreign-owned checkout.

These sites do not go through ``executors/_git.py``, and their refusals do not
surface as errors: one silently rebases a patch snapshot on the dirty worktree,
the other reports "no version tags" for a repo whose tags are intact.

``GIT_TEST_ASSUME_DIFFERENT_OWNER`` is git's own hook for this path, so the
tests need no root and no foreign-owned directory.
"""

from __future__ import annotations

import subprocess

import pytest

_COMMITTED = "committed base\n"
_DIRTY = "dirty worktree\n"

_PATCH = """--- a/kern.py
+++ b/kern.py
@@ -1 +1 @@
-committed base
+patched
"""


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=t@t.local",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "base",
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def foreign_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kern.py").write_text(_COMMITTED, encoding="utf-8")
    _init_repo(repo)
    subprocess.run(["git", "-C", str(repo), "tag", "v0.1.0"], check=True, capture_output=True)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    return repo


def test_patch_snapshot_base_comes_from_head_not_the_dirty_worktree(foreign_repo, tmp_path):
    """The on-disk fallback exists for non-git frameworks, not for a refusal.

    With the worktree dirty, silently falling back to it puts the patch on a
    base that was never committed, and nothing reports that it happened.
    """
    from hyperloom.orchestrator.kernel import request_handlers

    (foreign_repo / "kern.py").write_text(_DIRTY, encoding="utf-8")
    patch = tmp_path / "fusion.patch"
    patch.write_text(_PATCH, encoding="utf-8")

    snap = request_handlers.materialize_unified_patch_snapshot(
        patch_path=patch,
        repo_root=foreign_repo,
        snapshot_dir=tmp_path / "snap",
    )

    from pathlib import Path

    # Snapshot mode carries final bytes, so success itself is the evidence: the
    # patch only applies against the committed base, never the dirty one.
    assert (Path(snap) / "kern.py").read_text(encoding="utf-8") == "patched\n"


# Verbs git refuses on a foreign-owned checkout, measured rather than assumed.
# The apply family is absent on purpose: it works the tree as plain files and
# never validates repository ownership, even under --index.
_BLOCKED_VERBS = ("rev-parse", "status", "tag", "show", "clean", "ls-files", "checkout", "reset")

_GUARDED_MODULES = (
    "orchestrator/kernel/request_handlers.py",
    "orchestrator/framework/targeted_build.py",
    "orchestrator/actions/executors/baseline.py",
)


@pytest.mark.parametrize("relpath", _GUARDED_MODULES)
def test_no_unguarded_git_call_for_a_blocked_verb(relpath):
    """These modules build argv themselves instead of using executors/_git.py.

    A refusal at any of them is silent or misattributed, so the exception has to
    travel with every call rather than being remembered per site.
    """
    from pathlib import Path

    import hyperloom

    source = (Path(hyperloom.__file__).parent / relpath).read_text(encoding="utf-8")

    offenders = [
        line.strip()
        for line in source.splitlines()
        if '"git"' in line and "safe_directory_args" not in line and any(f'"{verb}"' in line for verb in _BLOCKED_VERBS)
    ]

    assert not offenders, f"{relpath} calls git unguarded: {offenders}"
