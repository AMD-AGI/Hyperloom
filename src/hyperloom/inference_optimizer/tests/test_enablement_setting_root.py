# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The replay script must name the root its patches belong to."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.enablement.artifacts import _root_the_patches_name


def _patches(tmp_path: Path, *headers: str) -> tuple[Path, list[dict]]:
    dest = tmp_path / "reports" / "enablement" / "patches"
    dest.mkdir(parents=True)
    rels = []
    for i, header in enumerate(headers, 1):
        name = f"{i:03d}_p.patch"
        (dest / name).write_text(f"--- a/{header}\n+++ b/{header}\n@@ -1 +1 @@\n-x\n+y\n", encoding="utf-8")
        rels.append(f"patches/{name}")
    return dest, [{"patches": rels}]


def _raw_patch(tmp_path: Path, body: str) -> tuple[Path, list[dict]]:
    dest = tmp_path / "reports" / "enablement" / "patches"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "001_p.patch").write_text(body, encoding="utf-8")
    return dest, [{"patches": ["patches/001_p.patch"]}]


def test_a_recorded_root_the_patches_do_not_belong_to_is_corrected(tmp_path: Path):
    """The defect this closes: every patch failed to apply, and nothing said why.

    ``framework_root`` is the root of whichever round last set it, so a run that
    touched a second tree at any point leaves the script naming that tree. The
    patches are then applied to a package they were not written against.
    """
    dest, rounds = _patches(tmp_path, "vllm/platforms/rocm.py", "vllm/model_executor/layers/mhc.py")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "vllm")


def test_a_root_that_already_agrees_is_kept(tmp_path: Path):
    dest, rounds = _patches(tmp_path, "vllm/platforms/rocm.py")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "vllm")) == str(site / "vllm")


def test_patches_that_disagree_leave_the_recorded_root_alone(tmp_path: Path):
    """Two trees in one recipe is not something a single root can express.

    Guessing one of them would silently drop the other half of the replay.
    """
    dest, rounds = _patches(tmp_path, "vllm/platforms/rocm.py", "aiter/ops/triton/x.py")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "aiter")


def test_no_sibling_by_that_name_leaves_the_recorded_root_alone(tmp_path: Path):
    """A path that does not exist is worse than one that at least came from the run."""
    dest, rounds = _patches(tmp_path, "vllm/platforms/rocm.py")
    site = tmp_path / "site-packages"
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "aiter")


def test_multi_tree_diff_prefixes_are_not_mistaken_for_directories(tmp_path: Path):
    """git writes ``a2/``/``b3/`` when one diff spans several trees."""
    dest = tmp_path / "reports" / "enablement" / "patches"
    dest.mkdir(parents=True)
    (dest / "001_p.patch").write_text(
        "diff --git a2/vllm/x.py b2/vllm/x.py\n--- a2/vllm/x.py\n+++ b2/vllm/x.py\n", encoding="utf-8"
    )
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, [{"patches": ["patches/001_p.patch"]}], str(site / "aiter")) == str(
        site / "vllm"
    )


def test_an_empty_recorded_root_is_returned_unchanged(tmp_path: Path):
    dest, rounds = _patches(tmp_path, "vllm/x.py")

    assert _root_the_patches_name(dest, rounds, "") == ""


def test_two_trees_inside_one_patch_leave_the_recorded_root_alone(tmp_path: Path):
    """A patch can touch several files. Stopping at the first is not reading it.

    Redirecting to whichever tree happened to come first would silently produce
    a replay that applies only part of the recipe.
    """
    dest, rounds = _raw_patch(
        tmp_path,
        "--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-x\n+y\n"
        "--- a/aiter/y.py\n+++ b/aiter/y.py\n@@ -1 +1 @@\n-x\n+y\n",
    )
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "aiter")


def test_a_creation_patch_is_read_from_its_new_side(tmp_path: Path):
    """A new file's old side is /dev/null; reading it records nothing useful."""
    dest, rounds = _raw_patch(tmp_path, "--- /dev/null\n+++ b/vllm/new_kernel.py\n@@ -0,0 +1 @@\n+x\n")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "vllm")


def test_a_creation_alongside_an_ordinary_patch_still_agrees(tmp_path: Path):
    """Both name the same tree, so the pair must not read as a disagreement."""
    dest = tmp_path / "reports" / "enablement" / "patches"
    dest.mkdir(parents=True)
    (dest / "001_p.patch").write_text("--- /dev/null\n+++ b/vllm/new_kernel.py\n@@ -0,0 +1 @@\n+x\n", encoding="utf-8")
    (dest / "002_p.patch").write_text("--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-x\n+y\n", encoding="utf-8")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)
    rounds = [{"patches": ["patches/001_p.patch", "patches/002_p.patch"]}]

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "vllm")


def test_a_deletion_patch_is_read_from_its_old_side(tmp_path: Path):
    dest, rounds = _raw_patch(tmp_path, "--- a/vllm/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "vllm")


def test_hunk_payload_shaped_like_headers_is_not_read_as_one(tmp_path: Path):
    """A removed line ``-- x`` and an added ``++ x`` render with header prefixes.

    Matching prefixes globally lets patch content invent a tree name, which makes
    the patches read as disagreeing and silently keeps the known-wrong root.
    """
    dest, rounds = _raw_patch(
        tmp_path,
        "--- a/vllm/x.py\n"
        "+++ b/vllm/x.py\n"
        "@@ -1,3 +1,3 @@\n"
        "---- aiter/not_a_header.py\n"
        "++++ aiter/not_a_header.py\n"
        " context\n",
    )
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "vllm")


def test_a_context_bearing_hunk_does_not_swallow_the_next_file_section(tmp_path: Path):
    """A context line belongs to both sides, so the counts are not summed.

    Summing them over-counts the body by the number of context lines, and the
    hunk then consumes the following section's headers -- so a patch spanning
    two trees reads as naming only the first, and the root is redirected to it.
    """
    dest, rounds = _raw_patch(
        tmp_path,
        "--- a/vllm/x.py\n"
        "+++ b/vllm/x.py\n"
        "@@ -1,3 +1,3 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three\n"
        "--- a/aiter/y.py\n"
        "+++ b/aiter/y.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n",
    )
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "aiter")


def test_a_no_newline_marker_does_not_shift_the_counts(tmp_path: Path):
    dest, rounds = _raw_patch(
        tmp_path,
        "--- a/vllm/x.py\n"
        "+++ b/vllm/x.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "\\ No newline at end of file\n"
        "+y\n"
        "\\ No newline at end of file\n",
    )
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "vllm")


def test_a_rename_only_patch_still_names_its_tree(tmp_path: Path):
    """Rename-only, mode-only and binary patches carry no ---/+++ pair.

    Excluding them from the agreement lets a recipe spanning two trees look
    unanimous, and the tree named only by the excluded patch becomes
    unreplayable.
    """
    dest = tmp_path / "reports" / "enablement" / "patches"
    dest.mkdir(parents=True)
    (dest / "001_p.patch").write_text(
        "diff --git a/aiter/old.py b/aiter/new.py\nsimilarity index 100%\n"
        "rename from aiter/old.py\nrename to aiter/new.py\n",
        encoding="utf-8",
    )
    (dest / "002_p.patch").write_text("--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-x\n+y\n", encoding="utf-8")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)
    rounds = [{"patches": ["patches/001_p.patch", "patches/002_p.patch"]}]

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "aiter")


def test_a_patch_that_names_no_tree_at_all_keeps_the_recorded_root(tmp_path: Path):
    """Unclassifiable is not absent: it may be the one that targets elsewhere."""
    dest = tmp_path / "reports" / "enablement" / "patches"
    dest.mkdir(parents=True)
    (dest / "001_p.patch").write_text("GIT binary patch\nliteral 0\n", encoding="utf-8")
    (dest / "002_p.patch").write_text("--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-x\n+y\n", encoding="utf-8")
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)
    rounds = [{"patches": ["patches/001_p.patch", "patches/002_p.patch"]}]

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "aiter")


def test_a_git_header_agreeing_with_its_body_still_resolves(tmp_path: Path):
    dest, rounds = _raw_patch(
        tmp_path,
        "diff --git a/vllm/x.py b/vllm/x.py\n--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-x\n+y\n",
    )
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "aiter").mkdir(parents=True)

    assert _root_the_patches_name(dest, rounds, str(site / "aiter")) == str(site / "vllm")
