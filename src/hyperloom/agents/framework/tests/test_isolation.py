# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.agents.framework.isolation (disk preflight)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hyperloom.agents.framework.isolation import (
    DiskPreflightError,
    disk_preflight,
)


def test_disk_preflight_passes_when_enough_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 1GB floor is easily satisfied by a normal scratch mount."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "0.001")
    disk_preflight(tmp_path / "wd", n_candidates=2)


def test_disk_preflight_raises_when_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Floor higher than available free GB triggers DiskPreflightError."""
    usage = shutil.disk_usage(str(tmp_path))
    impossible_gb = (usage.free / (1024**3)) + 10_000
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", str(impossible_gb))
    with pytest.raises(DiskPreflightError) as exc:
        disk_preflight(tmp_path / "wd", n_candidates=1)
    assert "insufficient disk" in str(exc.value)
    assert "required" in str(exc.value)


def test_disk_preflight_n_candidates_scales_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """required = max(floor, n * per_candidate) — large n breaches a low floor."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "0.001")
    usage = shutil.disk_usage(str(tmp_path))
    free_gb = usage.free / (1024**3)
    n = int(free_gb / 1.5) + 50
    with pytest.raises(DiskPreflightError):
        disk_preflight(tmp_path / "wd", n_candidates=n, per_candidate_gb=1.5)


def test_disk_preflight_env_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FRAMEWORK_EXPLORER_DISK_MIN_GB sets the floor."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "999999")
    with pytest.raises(DiskPreflightError):
        disk_preflight(tmp_path / "wd", n_candidates=1)
