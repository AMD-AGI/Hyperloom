# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""External (SaFE-less) multi-node mode: env-synthesized state and provision skip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.multi_node import _provision_multi_node_rayjob_stack
from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node.state_paths import resolve_state_file


@pytest.fixture()
def _external_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Minimal infera PD external env without SaFE credentials."""
    monkeypatch.delenv("SAFE_API_URL", raising=False)
    monkeypatch.delenv("SAFE_API_KEY", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://frontend:8000")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.1")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_IPS", "10.0.1.2")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_KEY", str(tmp_path / "id_ed25519"))
    (tmp_path / "id_ed25519").write_text("fake-key", encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("PD_MODE", "disaggregated")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "infera")
    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    return session


def test_build_external_state_from_env_pd_topology(_external_env: Path) -> None:
    state = ext.build_external_state_from_env()
    assert state["external"] is True
    assert state["service_url"] == "http://frontend:8000"
    assert state["backend"] == "infera"
    assert state["prefill_pod_ips"] == ["10.0.1.1"]
    assert state["decode_pod_ips"] == ["10.0.1.2"]
    assert state["last_restart_pd_prefill_nodes"] == 1
    assert state["last_restart_pd_decode_nodes"] == 1


def test_load_multi_node_state_falls_back_to_env(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ext.load_multi_node_state()["service_url"] == "http://frontend:8000"


def test_load_multi_node_state_prefers_env_over_stale_disk(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External env must override a leftover SaFE state file in the same session."""
    state_path = resolve_state_file()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "backend": "infera",
                "service_url": "http://old-safe:8000",
                "prefill_pod_ips": ["10.9.9.9"],
                "ssh_key_path": "/old/key",
            }
        ),
        encoding="utf-8",
    )
    loaded = ext.load_multi_node_state()
    assert loaded["service_url"] == "http://frontend:8000"
    assert loaded["prefill_pod_ips"] == ["10.0.1.1"]
    assert loaded.get("external") is True


def test_load_multi_node_state_uses_disk_when_safe_present(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SaFE creds present => external env ignored; on-disk state wins."""
    monkeypatch.setenv("SAFE_API_URL", "http://safe")
    monkeypatch.setenv("SAFE_API_KEY", "key")
    state_path = resolve_state_file()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"backend": "infera", "service_url": "http://safe-cluster:8000"}),
        encoding="utf-8",
    )
    assert ext.load_multi_node_state()["service_url"] == "http://safe-cluster:8000"


def test_external_service_url_ignored_when_safe_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAFE_API_URL", "http://safe")
    monkeypatch.setenv("SAFE_API_KEY", "key")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://frontend:8000")
    assert ext.external_service_url() == ""


def test_provision_external_writes_state_and_skips_safe(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        nodes=2,
        mn_backend="infera",
        mn_image=None,
        model="/models/test",
        no_kernel=True,
    )
    _provision_multi_node_rayjob_stack(args)
    state_path = resolve_state_file()
    assert state_path.is_file()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["external"] is True
    assert saved["prefill_pod_ips"] == ["10.0.1.1"]
    assert os.environ["BENCHMARK_BASE_URL"] == "http://frontend:8000"
    assert os.environ["MAGPIE_RUN_PHASE"] == "client"


def test_provision_external_infera_requires_ssh(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_KEY", raising=False)
    args = argparse.Namespace(
        nodes=2,
        mn_backend="infera",
        mn_image=None,
        model="/models/test",
        no_kernel=True,
    )
    with pytest.raises(SystemExit) as ei:
        _provision_multi_node_rayjob_stack(args)
    assert ei.value.code == 2
