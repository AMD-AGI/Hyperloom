# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Version fallback for the versioned patch subdir resolver and version gate."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors._server_patcher import (
    _resolve_versioned_patches_dir,
    _version_accepted,
    _version_gate_for,
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


def test_prefix_selects_the_framework_tree(tmp_path: Path):
    root = tmp_path / "vllm"
    exact = _mk(root, "vllm_0_25_0", "moe.patch")
    _mk(root, "sglang_0_5_14")
    assert _resolve_versioned_patches_dir(root, "0.25.0", prefix="vllm") == exact
    assert _versioned_patches_subdir_name("0.25.0", prefix="vllm") == "vllm_0_25_0"


def test_required_patch_skips_a_dir_holding_other_patches(tmp_path: Path):
    root = tmp_path / "vllm"
    _mk(root, "vllm_0_25_1", "other.patch")
    wanted = _mk(root, "vllm_0_25_0", "moe.patch")
    # 0.25.1 is the nearest not-newer dir, but it has no moe.patch, so the
    # resolver must keep falling back rather than hand back a dir without it.
    assert _resolve_versioned_patches_dir(root, "0.25.1", prefix="vllm", required_patch="moe.patch") == wanted


def test_vllm_gate_fails_closed_without_a_manifest(tmp_path: Path, monkeypatch):
    for var in (
        "HYPERLOOM_VLLM_SERVING_PATCH_EXACT_VERSIONS",
        "HYPERLOOM_VLLM_SERVING_PATCH_ALLOWED_MINORS",
        "HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS",
        "HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS",
    ):
        monkeypatch.delenv(var, raising=False)
    gate = _version_gate_for("vllm")

    assert _version_accepted("0.25.0", patches_dir=tmp_path, gate=gate) is False
    (tmp_path / "SUPPORTED_VERSIONS.txt").write_text("# c\n0.25.0\n", encoding="utf-8")
    assert _version_accepted("0.25.0", patches_dir=tmp_path, gate=gate) is True


def test_sglang_env_pin_does_not_leak_onto_vllm(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", "0.5")
    monkeypatch.delenv("HYPERLOOM_VLLM_SERVING_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_VLLM_SERVING_PATCH_ALLOWED_MINORS", raising=False)
    (tmp_path / "SUPPORTED_VERSIONS.txt").write_text("0.25.0\n", encoding="utf-8")

    assert _version_accepted("0.25.0", patches_dir=tmp_path, gate=_version_gate_for("vllm")) is True
    assert _version_accepted("0.25.0", patches_dir=tmp_path, gate=_version_gate_for("sglang")) is False
