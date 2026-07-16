# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
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
def patch_env(tmp_path, monkeypatch):
    fw = tmp_path / "fw"
    fw.mkdir()
    bak = tmp_path / "bak"
    bak.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", f"{fw}/")
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(bak))
    return fw, bak


def test_assert_revert_paths_allowed_happy_path(patch_env):
    fw, bak = patch_env
    pps = _load_patch_path_safety("pps_happy")
    target = fw / "mod.py"
    target.write_text("x", encoding="utf-8")
    backup = bak / "mod.bak"
    backup.write_text("y", encoding="utf-8")
    pps.assert_revert_paths_allowed(target, backup)


def test_assert_revert_rejects_backup_outside_root(patch_env):
    fw, bak = patch_env
    pps = _load_patch_path_safety("pps_bad_backup")
    target = fw / "mod.py"
    target.write_text("x", encoding="utf-8")
    outside = bak.parent / "escape.bak"
    outside.write_text("evil", encoding="utf-8")
    with pytest.raises(ValueError, match="backup_path"):
        pps.assert_revert_paths_allowed(target, outside)


def test_assert_revert_rejects_target_outside_framework(patch_env):
    fw, bak = patch_env
    pps = _load_patch_path_safety("pps_bad_target")
    outside = fw.parent / "escape.py"
    outside.write_text("x", encoding="utf-8")
    backup = bak / "mod.bak"
    backup.write_text("y", encoding="utf-8")
    with pytest.raises(ValueError, match="target_path"):
        pps.assert_revert_paths_allowed(outside, backup)


def test_assert_backup_dir_allowed_under_root(patch_env):
    pps = _load_patch_path_safety("pps_bdir")
    _, bak = patch_env
    pps.assert_backup_dir_allowed(bak / "nested")
