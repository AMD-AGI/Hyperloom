# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""xDiT TraceLens patch plan discovery and version fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors import _server_patcher as sp


def _mk_patches(tmp_path: Path, versions: list[str]) -> Path:
    d = tmp_path / "xdit_patches"
    d.mkdir()
    for v in versions:
        (d / f"config_xdit_v{v}.patch").write_text("dummy\n", encoding="utf-8")
    return d


# --- _resolve_xdit_patch_file fallback ---


def test_exact_match_wins(tmp_path: Path):
    d = _mk_patches(tmp_path, ["26.7", "26.8"])
    got = sp._resolve_xdit_patch_file(d, "26.7")
    assert got is not None and got.name == "config_xdit_v26.7.patch"


def test_nearest_lower_fallback(tmp_path: Path):
    d = _mk_patches(tmp_path, ["26.7"])
    got = sp._resolve_xdit_patch_file(d, "26.8")
    assert got is not None and got.name == "config_xdit_v26.7.patch"


def test_only_higher_returns_none(tmp_path: Path):
    d = _mk_patches(tmp_path, ["26.8"])
    assert sp._resolve_xdit_patch_file(d, "26.7") is None


def test_empty_dir_returns_none(tmp_path: Path):
    d = tmp_path / "xdit_patches"
    d.mkdir()
    assert sp._resolve_xdit_patch_file(d, "26.7") is None


def test_missing_dir_returns_none(tmp_path: Path):
    assert sp._resolve_xdit_patch_file(tmp_path / "absent", "26.7") is None


# --- _discover_xdit_plan skip paths ---


@pytest.fixture()
def xdit_tree(tmp_path: Path):
    """A minimal xDiT source tree and TraceLens patch tree."""
    source = tmp_path / "xDiT"
    source.mkdir()
    (source / ".git").mkdir()
    sentinel = source / "xfuser" / "config"
    sentinel.mkdir(parents=True)
    (sentinel / "args.py").write_text("profile_capture_phase\n", encoding="utf-8")
    tracelens = tmp_path / "TraceLens"
    patches = tracelens / "examples" / "custom_workflows" / "inference_analysis" / "xdit_patches"
    patches.mkdir(parents=True)
    (patches / "config_xdit_v26.7.patch").write_text("dummy\n", encoding="utf-8")
    return source, tracelens


_KNOWN_SHA = "c30266378c815c2f31f95b9d21d2818276050547"


def test_plan_built_for_known_sha(xdit_tree, monkeypatch):
    source, tracelens = xdit_tree
    monkeypatch.setenv("XDIT_PATH", str(source))
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens))
    with patch.object(sp, "_git_head_sha", return_value=_KNOWN_SHA):
        plan = sp._discover_xdit_plan()
    assert plan is not None
    assert plan.framework == "xdit"
    assert plan.version == "26.7"


def test_skip_when_source_root_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDIT_PATH", str(tmp_path / "nonexistent"))
    assert sp._discover_xdit_plan() is None


def test_skip_when_sha_unreadable(xdit_tree, monkeypatch):
    source, _tracelens = xdit_tree
    monkeypatch.setenv("XDIT_PATH", str(source))
    with patch.object(sp, "_git_head_sha", return_value=""):
        assert sp._discover_xdit_plan() is None


def test_skip_when_sha_unknown(xdit_tree, monkeypatch):
    source, _tracelens = xdit_tree
    monkeypatch.setenv("XDIT_PATH", str(source))
    with patch.object(sp, "_git_head_sha", return_value="deadbeef" * 5):
        assert sp._discover_xdit_plan() is None


def test_skip_when_tracelens_root_missing(xdit_tree, monkeypatch, tmp_path):
    source, _tracelens = xdit_tree
    monkeypatch.setenv("XDIT_PATH", str(source))
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "empty-cache"))
    with patch.object(sp, "_git_head_sha", return_value=_KNOWN_SHA):
        assert sp._discover_xdit_plan() is None


def test_skip_when_sentinel_missing(xdit_tree, monkeypatch):
    source, tracelens = xdit_tree
    monkeypatch.setenv("XDIT_PATH", str(source))
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens))
    (source / "xfuser" / "config" / "args.py").unlink()
    with patch.object(sp, "_git_head_sha", return_value=_KNOWN_SHA):
        assert sp._discover_xdit_plan() is None
