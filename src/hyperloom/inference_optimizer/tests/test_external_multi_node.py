# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cluster hand-off: env-synthesized multi-node state and its adoption guards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.multi_node import _prepare_multi_node_state
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


def test_external_state_default_ssh_port_matches_image(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the platform omits ``HYPERLOOM_MN_EXT_SSH_PORT``, use the image default."""
    from hyperloom.inference_optimizer.multi_node._internal.ssh_client import DEFAULT_SSH_PORT

    monkeypatch.delenv("HYPERLOOM_MN_EXT_SSH_PORT", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.1")
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)

    state = ext.build_external_state_from_env()
    assert state["ssh_port"] == DEFAULT_SSH_PORT
    assert state["prefill_pods"][0]["sshPort"] == DEFAULT_SSH_PORT


def test_external_pod_ssh_ports_follow_the_lws_ordinal(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each pod of a multi-node role listens on ``base + role_offset + ordinal``.

    Regression guard. ``mn-idle.sh`` is launched with
    ``MN_SSH_PORT=$(( role_base + ${LWS_WORKER_INDEX:-0} ))``, so the pods of one
    LeaderWorkerSet group listen on consecutive ports. External mode used to hand
    the same ``role_base`` to every IP in a role, which is invisible while a role
    has a single pod but makes every non-leader unreachable as soon as the role
    spans nodes. The platform supplies the IP lists in LWS ordinal order (leader
    first), so the list index IS the ordinal.
    """
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "2222")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.1,10.0.1.2")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_IPS", "10.0.2.1,10.0.2.2")

    state = ext.build_external_state_from_env()

    # prefill keeps the base; decode is strided so co-located roles never collide.
    assert [p["sshPort"] for p in state["prefill_pods"]] == [2222, 2223]
    assert [p["sshPort"] for p in state["decode_pods"]] == [2232, 2233]
    # Ports must pair with the right IP, in the order the platform listed them.
    assert [(p["podIP"], p["sshPort"]) for p in state["prefill_pods"]] == [
        ("10.0.1.1", 2222),
        ("10.0.1.2", 2223),
    ]


def test_external_aggregated_worker_ports_follow_the_lws_ordinal(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aggregated worker role is the common multi-node shape: one LWS group."""
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "2222")
    monkeypatch.setenv("PD_MODE", "aggregated")
    monkeypatch.delenv("HYPERLOOM_MN_EXT_PREFILL_IPS", raising=False)
    monkeypatch.delenv("HYPERLOOM_MN_EXT_DECODE_IPS", raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_WORKER_IPS", "10.0.3.1,10.0.3.2,10.0.3.3")

    state = ext.build_external_state_from_env()

    assert [p["sshPort"] for p in state["worker_pods"]] == [2222, 2223, 2224]


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


def test_load_multi_node_state_honours_handoff_alongside_safe_creds(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The platform sandbox case: LLM creds and a cluster hand-off coexist.

    SAFE_API_* authenticate the LLM gateway and are set in essentially every
    platform sandbox, so gating the hand-off on them made multi-node unusable
    exactly where the integration runs. The hand-off must still win here.
    """
    monkeypatch.setenv("SAFE_API_URL", "http://safe")
    monkeypatch.setenv("SAFE_API_KEY", "key")
    state_path = resolve_state_file()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"backend": "infera", "service_url": "http://stale-cluster:8000"}),
        encoding="utf-8",
    )
    loaded = ext.load_multi_node_state()
    assert loaded["service_url"] == "http://frontend:8000"
    assert loaded["prefill_pod_ips"] == ["10.0.1.1"]
    assert loaded.get("external") is True


def test_external_mode_signals_ignore_safe_creds(
    _external_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every external-mode signal keys off the hand-off, never off SaFE creds."""
    monkeypatch.setenv("SAFE_API_URL", "http://safe")
    monkeypatch.setenv("SAFE_API_KEY", "key")
    assert ext.external_service_url() == "http://frontend:8000"
    assert ext.external_has_ssh_control() is True
    assert ext.external_has_server_control() is True


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
    _prepare_multi_node_state(args)
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
        _prepare_multi_node_state(args)
    assert ei.value.code == 2


def test_bootstrap_accepts_handed_over_rayjob_without_rayjob_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A handed-over RayJob has head_pod_ip but no rayjob_id; bootstrap must run.

    Nothing writes rayjob_id now that the platform owns cluster creation, so
    demanding it in the state guard rejected every external RayJob before the
    first bootstrap could submit anything.
    """
    from hyperloom.inference_optimizer.multi_node import cli as mncli

    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://head:8888")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.2.1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")

    state = ext.load_multi_node_state()
    assert state["head_pod_ip"] == "10.0.2.1"
    assert not state.get("rayjob_id")

    # Stop right after the guard: reaching the Ray Dashboard proves it passed.
    class _Reached(RuntimeError):
        pass

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise _Reached("reached the dashboard client")

    monkeypatch.setattr(mncli, "_ray_dashboard_client", _boom)
    args = argparse.Namespace(script=None, force=False, print_logs=False)
    with pytest.raises(_Reached):
        mncli.cmd_bootstrap(args)
