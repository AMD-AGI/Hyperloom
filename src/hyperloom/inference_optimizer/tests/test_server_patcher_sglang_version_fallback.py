# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Version fallback for the SGLang TraceLens patch subdir resolver."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors._server_patcher import (
    _resolve_sglang_patches_dir,
)


def _mk(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "roofline.patch").write_text("# stub patch\n", encoding="utf-8")
    return d


def test_exact_version_subdir_preferred(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    _mk(root, "sglang_0_5_11")
    exact = _mk(root, "sglang_0_5_14")
    assert _resolve_sglang_patches_dir(root, "0.5.14") == exact


def test_nearest_not_newer_when_exact_missing(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    older = _mk(root, "sglang_0_5_11")
    _mk(root, "sglang_0_4_9")
    # 0.5.14 has no exact dir → nearest not-newer same-minor is 0.5.11.
    assert _resolve_sglang_patches_dir(root, "0.5.14") == older


def test_only_newer_available_returns_none(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    _mk(root, "sglang_0_6_0")
    assert _resolve_sglang_patches_dir(root, "0.5.14") is None


def test_empty_subdir_ignored(tmp_path: Path):
    root = tmp_path / "sglang_roofline_patches"
    (root / "sglang_0_5_11").mkdir(parents=True)  # no *.patch inside
    assert _resolve_sglang_patches_dir(root, "0.5.14") is None