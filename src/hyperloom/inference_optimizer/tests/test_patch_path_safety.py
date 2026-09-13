# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for pod-side patch path constraints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_patch_path_safety(unique_name: str):
    path = _repo_root() / "multi_node" / "scripts" / "patch_path_safety.py"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def backup_root(tmp_path, monkeypatch) -> Path:
    """Point the pod-side backup root at a writable temporary directory."""
    bak = tmp_path / "bak"
    bak.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(bak))
    return bak


def test_non_aiter_jit_build_shape_is_rejected(tmp_path):
    """The shape check is the only guard before a recursive move of the tree."""
    pps = _load_patch_path_safety("pps_jit_shape")
    bare = tmp_path / "random" / "jit" / "build"
    bare.mkdir(parents=True)

    with pytest.raises(ValueError, match="invalid AITER jit/build path"):
        pps.assert_aiter_jit_build_allowed(bare)


def test_assert_backup_path_allowed_under_root(backup_root):
    bak = backup_root
    pps = _load_patch_path_safety("pps_happy")
    backup = bak / "mod.bak"
    backup.write_text("y", encoding="utf-8")
    pps.assert_backup_path_allowed(backup)


def test_assert_backup_path_rejects_backup_outside_root(backup_root):
    bak = backup_root
    pps = _load_patch_path_safety("pps_bad_backup")
    outside = bak.parent / "escape.bak"
    outside.write_text("evil", encoding="utf-8")
    with pytest.raises(ValueError, match="backup_path"):
        pps.assert_backup_path_allowed(outside)


def test_assert_backup_dir_allowed_under_root(backup_root):
    pps = _load_patch_path_safety("pps_bdir")
    bak = backup_root
    pps.assert_backup_dir_allowed(bak / "nested")
