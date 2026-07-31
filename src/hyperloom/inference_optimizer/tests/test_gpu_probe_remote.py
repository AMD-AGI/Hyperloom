# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the multi-node remote GPU-type probe."""

from __future__ import annotations

import subprocess
import types

import pytest

from hyperloom.inference_optimizer.multi_node._internal import gpu_probe


@pytest.fixture(autouse=True)
def _has_handoff(monkeypatch):
    # external_service_url() must return non-empty or the probe short-circuits.
    monkeypatch.setattr(gpu_probe, "external_service_url", lambda: "http://claw-ray:8888")


class _FakeRayClient:
    """Minimal RayDashboardClient stand-in for the rayjob probe path."""

    def __init__(self, logs: str, *, status: str = "SUCCEEDED") -> None:
        self._logs = logs
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def submit_job(self, _entrypoint):
        return "sub-1"

    def get_job(self, _sub_id):
        return {"status": self._status}

    def get_job_logs(self, _sub_id):
        return self._logs


def test_rayjob_probe_parses_product_name(monkeypatch):
    # rocm-smi on the Ray head reports MI325X -> resolved to mi325x.
    monkeypatch.setattr(
        gpu_probe,
        "build_external_state_from_env",
        lambda: {"backend": "rayjob", "head_pod_ip": "10.0.0.1"},
    )
    monkeypatch.setattr(
        gpu_probe.ray_dashboard,
        "RayDashboardClient",
        lambda *_a, **_k: _FakeRayClient("Card series: AMD Instinct MI325X OAM"),
    )

    assert gpu_probe.remote_autodetect_gpu_type(timeout_s=1) == "mi325x"


def test_infera_probe_parses_product_name(monkeypatch):
    # SSH into the first GPU pod; rocm-smi output resolves to mi325x.
    monkeypatch.setattr(
        gpu_probe,
        "build_external_state_from_env",
        lambda: {
            "backend": "infera",
            "ssh_key_path": "/tmp/key",
            "ssh_known_hosts": "",
            "prefill_pods": [{"podIP": "10.0.0.2", "sshPort": 27720}],
        },
    )
    monkeypatch.setattr(
        gpu_probe.ssh_known_hosts,
        "refresh_known_hosts",
        lambda _hosts, dest: dest,
    )

    def _fake_ssh_run(host, command, **_kwargs):
        assert host == "10.0.0.2"
        return subprocess.CompletedProcess(
            args=[command], returncode=0, stdout="GPU[0] : MI325X\n", stderr=""
        )

    monkeypatch.setattr(gpu_probe.ssh_client, "ssh_run", _fake_ssh_run)

    assert gpu_probe.remote_autodetect_gpu_type(timeout_s=1) == "mi325x"


def test_probe_failure_falls_back_to_none(monkeypatch):
    # Any transport error is swallowed so the caller can fall back to --gpu-type.
    monkeypatch.setattr(
        gpu_probe,
        "build_external_state_from_env",
        lambda: {"backend": "rayjob", "head_pod_ip": "10.0.0.1"},
    )

    def _boom(*_a, **_k):
        raise RuntimeError("dashboard unreachable")

    monkeypatch.setattr(gpu_probe.ray_dashboard, "RayDashboardClient", _boom)

    assert gpu_probe.remote_autodetect_gpu_type(timeout_s=1) is None


def test_no_handoff_returns_none(monkeypatch):
    # Without an external service URL there is no cluster to probe.
    monkeypatch.setattr(gpu_probe, "external_service_url", lambda: "")
    assert gpu_probe.remote_autodetect_gpu_type(timeout_s=1) is None


def test_parse_gpu_type_prefers_specific_tag():
    # gcnArchName fallback resolves gfx942 -> mi300x runner.
    assert gpu_probe._parse_gpu_type("gfx942:sramecc+:xnack-") == "mi300x"
    # MI325X is not shadowed by the MI300X substring rule.
    assert gpu_probe._parse_gpu_type("AMD Instinct MI325X") == "mi325x"
    assert gpu_probe._parse_gpu_type("nothing here") is None
