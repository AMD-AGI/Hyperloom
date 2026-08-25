# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Already-applied patches must be a satisfied no-op, not an apply failure.

A specialist commonly writes both a superset patch and the subset it contains
(e.g. an FLA state-layout revert, plus that same revert bundled with a config
fix). Whichever lands first makes the other's hunks a no-op, and both POSIX
``patch`` and ``git apply --check`` reject that with a non-zero exit that looks
exactly like "does not apply". Treating it as a hard failure aborted the whole
enablement combo and reverted a correctly-applied fix.

Both apply channels resolve the ambiguity with a *reverse* dry-run, which
succeeds only when every hunk's post-state is already present. These tests pin
the three outcomes that matter: no-op on full overlap, real failure on no
overlap, and real failure on *partial* overlap (where a no-op would be wrong).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors._nogit_patch import (
    _apply_patch_no_git,
    _reverse_applies_cleanly,
)


_SUBSET = """\
--- a/mod/alpha.py
+++ b/mod/alpha.py
@@ -1,3 +1,3 @@
 header
-old_alpha
+new_alpha
 footer
"""

# Superset: the subset's hunk plus a second file the subset never touches.
_SUPERSET = (
    _SUBSET
    + """\
--- a/mod/beta.py
+++ b/mod/beta.py
@@ -1,3 +1,3 @@
 header
-old_beta
+new_beta
 footer
"""
)

_UNRELATED = """\
--- a/mod/alpha.py
+++ b/mod/alpha.py
@@ -1,3 +1,3 @@
 header
-nothing_matches_this
+replacement
 footer
"""


def _tree(root: Path) -> None:
    """Materialize the two-file source tree both patches target."""
    (root / "mod").mkdir(parents=True)
    (root / "mod" / "alpha.py").write_text("header\nold_alpha\nfooter\n", encoding="utf-8")
    (root / "mod" / "beta.py").write_text("header\nold_beta\nfooter\n", encoding="utf-8")


def _patch(root: Path, name: str, body: str) -> Path:
    """Write ``body`` as a patch file under ``root`` and return its path."""
    p = root / name
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def workspace(tmp_path):
    """A source tree, a patch dir and a backup dir, all under tmp_path."""
    src = tmp_path / "src"
    _tree(src)
    patches = tmp_path / "patches"
    patches.mkdir()
    return src, patches, tmp_path / "backups"


def test_nogit_superset_then_subset_is_a_noop(workspace):
    """Subset after superset succeeds with no backups; both edits survive."""
    src, patches, backups = workspace
    sup = _patch(patches, "superset.patch", _SUPERSET)
    sub = _patch(patches, "subset.patch", _SUBSET)

    ok, err, recs, fb = _apply_patch_no_git(src, sup, backups, seq_offset=0)
    assert ok, err
    assert recs, "superset should have backed up the files it changed"

    ok2, err2, recs2, fb2 = _apply_patch_no_git(src, sub, backups, seq_offset=len(recs))
    assert ok2, err2
    assert fb2 is None
    # The no-op owns no backups — the patch that made the edits owns them, so a
    # revert restores the tree exactly once.
    assert recs2 == []

    assert "new_alpha" in (src / "mod" / "alpha.py").read_text(encoding="utf-8")
    assert "new_beta" in (src / "mod" / "beta.py").read_text(encoding="utf-8")


def test_nogit_subset_then_superset_fails_closed(workspace):
    """The reverse order is only *partly* satisfied, so it must fail, not no-op.

    Applying the subset first leaves the superset's extra ``beta`` hunk still
    outstanding. Reporting a no-op there would silently drop a real edit, so the
    reverse probe rejects it and the apply fails closed — the executor reverts
    and hands the specialist ``retry_feedback`` to reauthor from. The narrow
    no-op is deliberately limited to a *fully* satisfied patch.
    """
    src, patches, backups = workspace
    sup = _patch(patches, "superset.patch", _SUPERSET)
    sub = _patch(patches, "subset.patch", _SUBSET)

    ok, _, recs, _ = _apply_patch_no_git(src, sub, backups, seq_offset=0)
    assert ok
    ok2, err2, recs2, fb2 = _apply_patch_no_git(src, sup, backups, seq_offset=len(recs))
    assert not ok2
    assert "failed at all strip levels" in err2
    assert fb2 is not None, "a real failure must carry reauthor feedback"
    # Nothing was half-applied: beta is untouched.
    assert "old_beta" in (src / "mod" / "beta.py").read_text(encoding="utf-8")


def test_nogit_partial_overlap_still_fails(workspace):
    """A patch only *partly* present is a real failure, not a no-op.

    The superset's beta hunk has not been applied, so silently reporting success
    would drop a real edit.
    """
    src, patches, backups = workspace
    sup = _patch(patches, "superset.patch", _SUPERSET)
    ok, _, recs, _ = _apply_patch_no_git(src, sup, backups, seq_offset=0)
    assert ok
    # Undo only beta, leaving alpha applied -> superset is half-present.
    (src / "mod" / "beta.py").write_text("header\nold_beta\nfooter\n", encoding="utf-8")

    assert not _reverse_applies_cleanly(src, sup)


def test_nogit_unrelated_patch_still_fails(workspace):
    """A patch that genuinely does not apply keeps failing with feedback."""
    src, patches, backups = workspace
    bad = _patch(patches, "bad.patch", _UNRELATED)

    ok, err, recs, fb = _apply_patch_no_git(src, bad, backups, seq_offset=0)
    assert not ok
    assert "failed at all strip levels" in err
    assert fb is not None
    assert recs == []


def _git_tree(root: Path) -> None:
    """Initialise ``root`` as a committed git work tree."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(root), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True)


def test_git_channel_superset_then_subset_is_a_noop(tmp_path):
    """The git channel resolves the same ambiguity via ``git apply -R --check``."""
    src = tmp_path / "src"
    _tree(src)
    _git_tree(src)
    patches = tmp_path / "patches"
    patches.mkdir()
    sup = _patch(patches, "superset.patch", _SUPERSET)
    sub = _patch(patches, "subset.patch", _SUBSET)

    ok, err, _ = ip._git_apply_collect_feedback(src, sup)
    assert ok, err
    ok2, err2, fb2 = ip._git_apply_collect_feedback(src, sub)
    assert ok2, err2
    assert fb2 is None
    assert "new_alpha" in (src / "mod" / "alpha.py").read_text(encoding="utf-8")


def test_git_channel_unrelated_patch_still_fails(tmp_path):
    """A non-applying patch still fails on the git channel, with feedback."""
    src = tmp_path / "src"
    _tree(src)
    _git_tree(src)
    patches = tmp_path / "patches"
    patches.mkdir()
    bad = _patch(patches, "bad.patch", _UNRELATED)

    ok, err, fb = ip._git_apply_collect_feedback(src, bad)
    assert not ok
    assert fb is not None
