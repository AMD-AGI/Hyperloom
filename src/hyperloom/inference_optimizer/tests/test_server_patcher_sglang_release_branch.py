# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A build off SGLang's release branch must get the release-branch patch set.

TraceLens ships two sets per version: the plain one is cut against the point
release tag, the ``_sgldev`` one against the release branch the ``lmsysorg``
images are built from. Selecting the plain set for a branch build is what makes
``io_struct.patch`` fail to apply and silently drops kernel-shape profiling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._server_patcher import (
    _is_release_branch_build,
    _resolve_versioned_patches_dir,
    _versioned_patches_subdir_names,
)

# The two version strings the pre-release legs actually report: a bare point
# release from SGLANG_PRETEND_VERSION, and an image built off the release branch.
POINT_RELEASE = "0.5.18"
BRANCH_BUILD = "0.5.18.dev20260825+g0c7ff19e3b"


def _make(root: Path, *names: str) -> dict[str, Path]:
    made: dict[str, Path] = {}
    for name in names:
        d = root / name
        d.mkdir(parents=True)
        (d / "io_struct.patch").write_text("--- a/x\n+++ b/x\n")
        made[name] = d
    return made


@pytest.mark.parametrize(
    "version,expected",
    [
        (BRANCH_BUILD, True),
        ("0.5.18.dev20260825", True),
        ("0.5.19.dev1+g14b647cf27", True),
        (POINT_RELEASE, False),
        ("0.5.18", False),
        ("", False),
    ],
)
def test_release_branch_detection(version: str, expected: bool) -> None:
    assert _is_release_branch_build(version) is expected


def test_branch_build_prefers_the_variant_then_falls_back() -> None:
    assert _versioned_patches_subdir_names(BRANCH_BUILD) == [
        "sglang_0_5_18_sgldev",
        "sglang_0_5_18",
    ]


def test_point_release_asks_only_for_the_plain_set() -> None:
    assert _versioned_patches_subdir_names(POINT_RELEASE) == ["sglang_0_5_18"]


def test_branch_build_resolves_to_the_variant(tmp_path: Path) -> None:
    made = _make(tmp_path, "sglang_0_5_18", "sglang_0_5_18_sgldev")

    assert _resolve_versioned_patches_dir(tmp_path, BRANCH_BUILD) == made["sglang_0_5_18_sgldev"]


def test_point_release_resolves_to_the_plain_set(tmp_path: Path) -> None:
    made = _make(tmp_path, "sglang_0_5_18", "sglang_0_5_18_sgldev")

    assert _resolve_versioned_patches_dir(tmp_path, POINT_RELEASE) == made["sglang_0_5_18"]


def test_branch_build_takes_the_plain_set_when_no_variant_exists(tmp_path: Path) -> None:
    made = _make(tmp_path, "sglang_0_5_18")

    assert _resolve_versioned_patches_dir(tmp_path, BRANCH_BUILD) == made["sglang_0_5_18"]


def test_fallback_to_an_older_version_keeps_the_matching_variant(tmp_path: Path) -> None:
    # No 0.5.18 set at all: the nearest not-newer version is 0.5.17, which also
    # ships both variants, and a branch build still wants the branch one.
    made = _make(tmp_path, "sglang_0_5_17", "sglang_0_5_17_sgldev")

    assert _resolve_versioned_patches_dir(tmp_path, BRANCH_BUILD) == made["sglang_0_5_17_sgldev"]
    assert _resolve_versioned_patches_dir(tmp_path, POINT_RELEASE) == made["sglang_0_5_17"]
