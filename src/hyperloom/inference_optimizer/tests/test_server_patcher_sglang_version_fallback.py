# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Version fallback for the versioned patch subdir resolver and version gate."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors._server_patcher import (
    _patch_set_writes,
    _patch_target_paths,
    _resolve_versioned_patches_dir,
    _version_accepted,
    _versioned_patches_subdir_name,
)


def _mk(root: Path, name: str, patch: str = "roofline.patch") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / patch).write_text("# stub patch\n", encoding="utf-8")
    return d


def test_exact_version_subdir_preferred(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    _mk(root, "sglang_0_5_11")
    exact = _mk(root, "sglang_0_5_14")
    assert _resolve_versioned_patches_dir(root, "0.5.14") == exact


def test_nearest_not_newer_when_exact_missing(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    older = _mk(root, "sglang_0_5_11")
    _mk(root, "sglang_0_4_9")
    # No exact dir -> nearest not-newer same-minor is 0.5.11.
    assert _resolve_versioned_patches_dir(root, "0.5.14") == older


def test_only_newer_available_returns_none(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    _mk(root, "sglang_0_6_0")
    assert _resolve_versioned_patches_dir(root, "0.5.14") is None


def test_empty_subdir_ignored(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    (root / "sglang_0_5_11").mkdir(parents=True)  # no *.patch inside
    assert _resolve_versioned_patches_dir(root, "0.5.14") is None


def test_cross_minor_fallback_when_no_same_minor(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    older = _mk(root, "sglang_0_4_9")
    # No same-minor subdir; fall back to the nearest not-newer subdir -> 0.4.9.
    assert _resolve_versioned_patches_dir(root, "0.5.14") == older


def test_subdir_name_maps_a_dotted_version(tmp_path: Path):
    assert _versioned_patches_subdir_name("0.5.14") == "sglang_0_5_14"
    # A dev/local suffix still resolves to its numeric head.
    assert _versioned_patches_subdir_name("0.5.10.dev4") == "sglang_0_5_10"
    assert _versioned_patches_subdir_name("main") is None


def test_a_vendor_manifest_replaces_the_builtin_minors(tmp_path: Path, monkeypatch):
    for var in (
        "HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS",
        "HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS",
    ):
        monkeypatch.delenv(var, raising=False)

    # 0.6 is outside the built-in 0.5 allowlist, so only a manifest admits it.
    assert _version_accepted("0.6.0", patches_dir=tmp_path) is False
    (tmp_path / "SUPPORTED_VERSIONS.txt").write_text("# c\n0.6.0\n", encoding="utf-8")
    assert _version_accepted("0.6.0", patches_dir=tmp_path) is True


def test_an_operator_pin_wins_over_the_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.setenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", "0.5")
    (tmp_path / "SUPPORTED_VERSIONS.txt").write_text("0.6.0\n", encoding="utf-8")

    assert _version_accepted("0.5.14", patches_dir=tmp_path) is True
    assert _version_accepted("0.6.0", patches_dir=tmp_path) is False


def test_only_the_files_a_patch_set_writes_become_sentinels(tmp_path: Path):
    """A release that stopped shipping a file must not leave a dead sentinel."""
    patch = tmp_path / "annotations.patch"
    patch.write_text(
        "--- a/python/sglang/srt/managers/scheduler.py\n+++ b/python/sglang/srt/managers/scheduler.py\n",
        encoding="utf-8",
    )
    written = _patch_target_paths([patch])

    assert _patch_set_writes(written, ("srt", "managers", "scheduler.py")) is True
    assert _patch_set_writes(written, ("srt", "managers", "io_struct.py")) is False


def test_a_created_file_is_read_from_the_patch_not_the_source_side(tmp_path: Path):
    """New-file patches carry ``/dev/null`` on the source side."""
    patch = tmp_path / "new_file.patch"
    patch.write_text(
        "--- /dev/null\n+++ b/python/sglang/srt/model_executor/step_span_utils.py\n",
        encoding="utf-8",
    )
    written = _patch_target_paths([patch])

    assert "/dev/null" not in written
    assert _patch_set_writes(written, ("srt", "model_executor", "step_span_utils.py")) is True
