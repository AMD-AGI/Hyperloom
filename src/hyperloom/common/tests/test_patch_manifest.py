# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract for ``parse_patch_manifest``: the git header vocabulary it must recognize."""

import pytest

from hyperloom.common.unified_diff import PatchOp, PatchTarget, parse_patch_manifest

WRITE = PatchOp.WRITE
DELETE = PatchOp.DELETE


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        pytest.param(
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -1 +1 @@\n-a\n+b\n",
            [PatchTarget(WRITE, "b/mod.py")],
            id="modify",
        ),
        pytest.param(
            "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n",
            [PatchTarget(WRITE, "b/new.py", mode="0644", is_new=True)],
            id="create-new-file-mode",
        ),
        pytest.param(
            "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n",
            [PatchTarget(WRITE, "b/new.py", is_new=True)],
            id="create-dev-null-source",
        ),
        pytest.param(
            "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n",
            [PatchTarget(DELETE, "a/gone.py")],
            id="delete-deleted-file-mode",
        ),
        pytest.param(
            "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n",
            [PatchTarget(DELETE, "a/gone.py")],
            id="delete-dev-null-destination",
        ),
        pytest.param(
            "diff --git a/gone.bin b/gone.bin\ndeleted file mode 100644\nindex eaf36c1..0000000\n"
            "Binary files a/gone.bin and /dev/null differ\n",
            [PatchTarget(DELETE, "b/gone.bin")],
            id="delete-binary-header-only",
        ),
        pytest.param(
            "diff --git a/empty.txt b/empty.txt\ndeleted file mode 100644\nindex e69de29..0000000\n",
            [PatchTarget(DELETE, "b/empty.txt")],
            id="delete-empty-file-header-only",
        ),
        pytest.param(
            "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n",
            [PatchTarget(DELETE, "a/old.py"), PatchTarget(WRITE, "b/new.py", is_new=True)],
            id="rename",
        ),
        pytest.param(
            "diff --git a/old.py b/new.py\nsimilarity index 80%\nrename from old.py\nrename to new.py\n"
            "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-a\n+b\n",
            [PatchTarget(DELETE, "a/old.py"), PatchTarget(WRITE, "b/new.py", is_new=True)],
            id="rename-with-edits",
        ),
        pytest.param(
            "diff --git old.py new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n",
            [PatchTarget(DELETE, "old.py"), PatchTarget(WRITE, "new.py", is_new=True)],
            id="rename-without-a-b-prefix",
        ),
        pytest.param(
            "diff --git a/src.py b/dup.py\nsimilarity index 100%\ncopy from src.py\ncopy to dup.py\n",
            [PatchTarget(WRITE, "b/dup.py", is_new=True)],
            id="copy",
        ),
        pytest.param(
            "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n",
            [PatchTarget(WRITE, "b/run.sh", mode="0755")],
            id="chmod-only",
        ),
        pytest.param(
            "diff --git a/my run.sh b/my run.sh\nold mode 100644\nnew mode 100755\n",
            [PatchTarget(WRITE, "b/my run.sh", mode="0755")],
            id="chmod-only-path-with-space",
        ),
        pytest.param(
            "diff --git a/img.png b/img.png\nnew file mode 100644\nGIT binary patch\nliteral 4\n",
            [PatchTarget(WRITE, "b/img.png", mode="0644", is_new=True)],
            id="binary",
        ),
        pytest.param(
            'diff --git "a/d\\303\\251.bin" "b/d\\303\\251.bin"\nindex eaf36c1..f75c32a 100644\n'
            'Binary files "a/d\\303\\251.bin" and "b/d\\303\\251.bin" differ\n',
            [PatchTarget(WRITE, "b/dé.bin")],
            id="binary-quoted-path",
        ),
        pytest.param(
            "diff --git b in.dat b in.dat\nindex eaf36c1..f75c32a 100644\nBinary files b in.dat and b in.dat differ\n",
            [PatchTarget(WRITE, "b in.dat")],
            id="binary-no-prefix-path-with-space",
        ),
        pytest.param(
            'diff --git "a/my dir/f.py" "b/my dir/f.py"\n--- /dev/null\n+++ "b/my dir/f.py"\n@@ -0,0 +1 @@\n+x\n',
            [PatchTarget(WRITE, "b/my dir/f.py", is_new=True)],
            id="quoted-path",
        ),
        pytest.param(
            'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\nindex 7d4290a..407de30 100644\n'
            '--- "a/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -1 +1 @@\n-x = 1\n+x = 2\n',
            [PatchTarget(WRITE, "b/café.py")],
            id="quoted-path-octal-utf8-bytes",
        ),
        pytest.param(
            'diff --git "a/né\\tx.py" "b/né\\tx.py"\nnew file mode 100644\nindex 0000000..edf0eff\n'
            '--- /dev/null\n+++ "b/né\\tx.py"\n@@ -0,0 +1 @@\n+hi\n',
            [PatchTarget(WRITE, "b/né\tx.py", mode="0644", is_new=True)],
            id="quoted-path-raw-utf8-and-escaped-tab",
        ),
        pytest.param(
            "--- target.py\n+++ target.py\n@@ -1 +1 @@\n-a\n+b\n",
            [PatchTarget(WRITE, "target.py")],
            id="no-git-header",
        ),
        pytest.param(
            "--- a/q.sql\n+++ b/q.sql\n@@ -1,2 +1,2 @@\n--- old comment\n+++ new\n select 1;\n"
            "--- a/r.sql\n+++ b/r.sql\n@@ -1 +1 @@\n-a\n+b\n",
            [PatchTarget(WRITE, "b/q.sql"), PatchTarget(WRITE, "b/r.sql")],
            id="no-git-header-hunk-lines-that-read-as-headers",
        ),
    ],
)
def test_header_vocabulary_maps_to_targets(patch, expected):
    """A rename is two targets; ``is_new`` keeps a caller from pre-seeding a base under a path the patch creates."""
    assert parse_patch_manifest(patch) == expected


def test_sections_are_returned_in_diff_order():
    patch = (
        "diff --git a/one.py b/one.py\n--- a/one.py\n+++ b/one.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/two.py b/two.py\n--- a/two.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n"
    )

    assert parse_patch_manifest(patch) == [
        PatchTarget(WRITE, "b/one.py"),
        PatchTarget(DELETE, "a/two.py"),
    ]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        pytest.param("   \n", "empty patch", id="blank"),
        pytest.param("--- \n@@ -1 +1 @@\n-a\n+b\n", "cannot determine target path", id="no-target"),
        pytest.param("--- /dev/null\n+++ /dev/null\n@@ -0,0 +0,0 @@\n", "missing source path", id="dev-null-both"),
        pytest.param(
            'diff --git "a/\\377.py" "b/\\377.py"\n--- "a/\\377.py"\n+++ "b/\\377.py"\n@@ -1 +1 @@\n-a\n+b\n',
            "is not valid UTF-8",
            id="quoted-path-not-utf8",
        ),
        pytest.param(
            "diff --git a/x y b/z\nold mode 100644\nnew mode 100755\n",
            "cannot determine target path",
            id="header-names-not-one-path",
        ),
    ],
)
def test_unusable_patch_is_rejected(patch, message):
    """A section naming no real path cannot be applied or reverted, so it raises rather than resolving to nothing."""
    with pytest.raises(ValueError, match=message):
        parse_patch_manifest(patch)
