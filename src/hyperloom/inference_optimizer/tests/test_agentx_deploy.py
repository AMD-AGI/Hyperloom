# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for runtime deployment of AgentX assets into the benchmarks dir."""

from __future__ import annotations

import os

import pytest

from hyperloom.inference_optimizer.agentx.deploy import (
    agentx_asset_dir,
    deploy_agentx_assets,
)


def test_asset_dir_has_all_files():
    d = agentx_asset_dir()
    assert (d / "aiperf_client.sh").exists()
    assert (d / "map_aiperf.py").exists()
    assert (d / "aiperf_phase_gate.py").exists()


def test_deploy_copies_and_is_executable(tmp_path):
    dst = tmp_path / "benchmarks"
    written = deploy_agentx_assets(dst)
    assert (dst / "aiperf_client.sh").exists()
    assert (dst / "map_aiperf.py").exists()
    assert (dst / "aiperf_phase_gate.py").exists()
    assert os.access(dst / "aiperf_client.sh", os.X_OK)
    assert len(written) == 3


def test_deploy_idempotent(tmp_path):
    dst = tmp_path / "benchmarks"
    deploy_agentx_assets(dst)
    deploy_agentx_assets(dst)  # second run must not raise
    assert (dst / "map_aiperf.py").exists()


def test_deploy_modes(tmp_path):
    dst = tmp_path / "benchmarks"
    deploy_agentx_assets(dst)
    assert (os.stat(dst / "aiperf_client.sh").st_mode & 0o777) == 0o700
    assert (os.stat(dst / "map_aiperf.py").st_mode & 0o777) == 0o600
    assert (os.stat(dst / "aiperf_phase_gate.py").st_mode & 0o777) == 0o600


def test_deploy_leaves_no_temp_files(tmp_path):
    dst = tmp_path / "benchmarks"
    deploy_agentx_assets(dst)
    leftovers = [p.name for p in dst.iterdir() if p.name.startswith(".")]
    assert leftovers == [], f"temp files leaked: {leftovers}"


def test_deploy_missing_source_asset_raises(tmp_path, monkeypatch):
    # Point the asset dir at a location missing one required file.
    fake_assets = tmp_path / "fake_assets"
    fake_assets.mkdir()
    (fake_assets / "aiperf_client.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.deploy.agentx_asset_dir",
        lambda: fake_assets,
    )
    with pytest.raises(FileNotFoundError):
        deploy_agentx_assets(tmp_path / "benchmarks")
