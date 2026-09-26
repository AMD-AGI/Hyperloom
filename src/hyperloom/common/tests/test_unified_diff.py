# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``hyperloom.common.unified_diff``."""

from __future__ import annotations

from hyperloom.common.unified_diff import (
    parse_unified_diff,
    strip_diff_path,
    strip_path_components,
    touched_paths,
)


def test_parse_unified_diff_with_and_without_git_header() -> None:
    patch = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/foo.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def added():\n"
        "+    return 1\n"
        " context line\n"
        "-removed line\n"
    )
    changes = parse_unified_diff(patch)
    assert len(changes) == 1
    assert changes[0].path == "src/foo.py"
    assert changes[0].is_new is True
    assert "def added():" in changes[0].added

    bare = "--- a/x.py\n+++ b/x.py\n@@\n+line\n"
    bare_changes = parse_unified_diff(bare)
    assert bare_changes and bare_changes[0].path == "x.py"

    assert parse_unified_diff("random preamble\nnothing here\n") == []


def test_parse_unified_diff_deletion_keeps_old_path() -> None:
    patch = (
        "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )
    [change] = parse_unified_diff(patch)
    assert change.path == "gone.py"
    assert change.is_deleted is True
    assert change.removed == ["x"]


def test_parse_unified_diff_names_sections_without_header_pair() -> None:
    patch = (
        "diff --git a/lib.so b/lib.so\n"
        "index 111..222 100644\n"
        "Binary files a/lib.so and b/lib.so differ\n"
        "diff --git a/old.py b/new.py\n"
        "similarity index 100%\n"
        "rename from old.py\n"
        "rename to new.py\n"
        "diff --git a/run.sh b/run.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )
    assert [c.path for c in parse_unified_diff(patch)] == ["lib.so", "new.py", "run.sh"]


def test_parse_unified_diff_rename_with_hunk_uses_new_path() -> None:
    patch = "diff --git a/old.py b/new.py\n--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-a\n+b\n"
    [change] = parse_unified_diff(patch)
    assert change.path == "new.py"
    assert (change.added, change.removed) == (["b"], ["a"])


def test_touched_paths_dedups_in_diff_order() -> None:
    diff = "diff --git a/x.py b/x.py\ndiff --git a/y.c b/y.c\ndiff --git a/x.py b/x.py\n"
    assert touched_paths(diff) == ["x.py", "y.c"]
    assert touched_paths("") == []


def test_strip_diff_path_variants() -> None:
    assert strip_diff_path("a/src/foo.py\t2024") == "src/foo.py"
    assert strip_diff_path(" b/src/foo.py \t2024-01-01 00:00") == "src/foo.py"
    assert strip_diff_path("/dev/null") == "/dev/null"
    assert strip_diff_path("plain/path.py") == "plain/path.py"


def test_strip_path_components_mimics_git_apply_p() -> None:
    assert strip_path_components("a/b/c.py", 0) == "a/b/c.py"
    assert strip_path_components("a/b/c.py", 1) == "b/c.py"
    assert strip_path_components("a/b/c.py", 5) == "c.py"
