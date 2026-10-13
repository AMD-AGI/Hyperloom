# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract for ``parse_patch_manifest``: the git header vocabulary it must recognize."""

import pytest

from hyperloom.common.unified_diff import parse_patch_manifest


def test_modification_is_one_write_against_the_prefixed_destination():
    patch = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-a\n+b\n"

    assert parse_patch_manifest(patch) == [
        {"op": "write", "path": "b/pkg/mod.py", "mode": "", "binary": False, "is_new": False}
    ]


@pytest.mark.parametrize(
    "header",
    [
        "new file mode 100644\n--- /dev/null\n+++ b/new.py\n",
        "--- /dev/null\n+++ b/new.py\n",
    ],
    ids=["new-file-mode", "dev-null-source"],
)
def test_creation_is_flagged_is_new(header):
    """``is_new`` is what keeps a caller from pre-seeding a base under a path the patch creates."""
    descriptors = parse_patch_manifest(f"diff --git a/new.py b/new.py\n{header}@@ -0,0 +1 @@\n+x\n")

    assert [(d["op"], d["path"], d["is_new"]) for d in descriptors] == [("write", "b/new.py", True)]


@pytest.mark.parametrize(
    "header",
    [
        "deleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n",
        "--- a/gone.py\n+++ /dev/null\n",
    ],
    ids=["deleted-file-mode", "dev-null-destination"],
)
def test_deletion_is_recorded_against_the_source_path(header):
    descriptors = parse_patch_manifest(f"diff --git a/gone.py b/gone.py\n{header}@@ -1 +0,0 @@\n-x\n")

    assert descriptors == [{"op": "delete", "path": "a/gone.py", "mode": "", "binary": False, "is_new": False}]


def test_rename_yields_a_delete_of_the_source_and_a_create_of_the_destination():
    patch = "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"

    assert parse_patch_manifest(patch) == [
        {"op": "delete", "path": "a/old.py", "mode": "", "binary": False, "is_new": False},
        {"op": "write", "path": "b/new.py", "mode": "", "binary": False, "is_new": True},
    ]


def test_rename_with_edits_still_names_the_destination_from_the_git_header():
    """The ``+++`` line of a rename-with-edits names the destination; both paths stay prefixed."""
    patch = (
        "diff --git a/old.py b/new.py\nsimilarity index 80%\nrename from old.py\nrename to new.py\n"
        "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-a\n+b\n"
    )

    assert [(d["op"], d["path"]) for d in parse_patch_manifest(patch)] == [
        ("delete", "a/old.py"),
        ("write", "b/new.py"),
    ]


def test_copy_creates_the_destination_only():
    patch = "diff --git a/src.py b/dup.py\nsimilarity index 100%\ncopy from src.py\ncopy to dup.py\n"

    assert parse_patch_manifest(patch) == [
        {"op": "write", "path": "b/dup.py", "mode": "", "binary": False, "is_new": True}
    ]


def test_chmod_only_section_takes_its_path_from_the_diff_line():
    """A mode change carries no ``---``/``+++`` pair, so the ``diff --git`` line is the only path source."""
    patch = "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"

    assert parse_patch_manifest(patch) == [
        {"op": "write", "path": "b/run.sh", "mode": "0755", "binary": False, "is_new": False}
    ]


def test_binary_section_is_flagged_so_content_is_sourced_from_a_snapshot():
    patch = "diff --git a/img.png b/img.png\nnew file mode 100644\nGIT binary patch\nliteral 4\n"

    assert parse_patch_manifest(patch) == [
        {"op": "write", "path": "b/img.png", "mode": "0644", "binary": True, "is_new": True}
    ]


def test_quoted_path_is_unquoted_before_any_containment_check():
    """Git C-quotes paths with special characters; leaving them quoted makes a ``..`` check read the wrong string."""
    patch = 'diff --git "a/my dir/f.py" "b/my dir/f.py"\n--- /dev/null\n+++ "b/my dir/f.py"\n@@ -0,0 +1 @@\n+x\n'

    assert parse_patch_manifest(patch) == [
        {"op": "write", "path": "b/my dir/f.py", "mode": "", "binary": False, "is_new": True}
    ]


def test_prefixless_diff_is_parsed_without_a_git_header():
    patch = "--- target.py\n+++ target.py\n@@ -1 +1 @@\n-a\n+b\n"

    assert [(d["op"], d["path"]) for d in parse_patch_manifest(patch)] == [("write", "target.py")]


def test_multiple_sections_are_returned_in_diff_order():
    patch = (
        "diff --git a/one.py b/one.py\n--- a/one.py\n+++ b/one.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/two.py b/two.py\n--- a/two.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n"
    )

    assert [(d["op"], d["path"]) for d in parse_patch_manifest(patch)] == [
        ("write", "b/one.py"),
        ("delete", "a/two.py"),
    ]


@pytest.mark.parametrize("patch", ["", "   \n"], ids=["empty", "blank"])
def test_empty_patch_is_rejected(patch):
    with pytest.raises(ValueError, match="empty patch"):
        parse_patch_manifest(patch)


def test_section_without_a_determinable_target_is_rejected():
    with pytest.raises(ValueError, match="cannot determine target path"):
        parse_patch_manifest("--- \n@@ -1 +1 @@\n-a\n+b\n")
