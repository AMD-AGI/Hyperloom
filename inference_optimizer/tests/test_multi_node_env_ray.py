"""Tests for multi-node Ray address helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne


def test_ray_gcs_address_from_state_prefers_ray_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps({"ray_address": "10.1.2.3:6379", "head_pod_ip": "10.9.9.9"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.3:6379"


def test_ray_gcs_address_from_state_fallback_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.1.2.4"}), encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.4:6379"


def test_export_ray_address_to_os(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.0.0.5"}), encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    mne.export_ray_address_to_os()
    assert os.environ.get("RAY_ADDRESS") == "10.0.0.5:6379"
