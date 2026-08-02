# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the multi-node remote GPU-type probe."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

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
        return subprocess.CompletedProcess(args=[command], returncode=0, stdout="GPU[0] : MI325X\n", stderr="")

    monkeypatch.setattr(gpu_probe.ssh_client, "ssh_run", _fake_ssh_run)

    assert gpu_probe.remote_autodetect_gpu_type(timeout_s=1) == "mi325x"


def _infera_state():
    return {
        "backend": "infera",
        "ssh_key_path": "/tmp/key",
        "ssh_known_hosts": "",
        "prefill_pods": [{"podIP": "10.0.0.2", "sshPort": 27720}],
    }


def _track_scratch_dirs(monkeypatch, tmp_path):
    """Redirect the probe's keyscan mkdtemp into tmp_path and record what it made."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(gpu_probe.tempfile, "mkdtemp", _tracking_mkdtemp)
    monkeypatch.setattr(gpu_probe, "build_external_state_from_env", _infera_state)
    monkeypatch.setattr(gpu_probe.ssh_known_hosts, "refresh_known_hosts", lambda _hosts, dest: dest)
    return created


@pytest.mark.parametrize("ssh_explodes", [False, True])
def test_infera_probe_removes_its_scratch_known_hosts(monkeypatch, tmp_path, ssh_explodes):
    """The one-shot keyscan dir must not accumulate under /tmp, even on failure."""
    created = _track_scratch_dirs(monkeypatch, tmp_path)

    def _fake_ssh_run(_host, command, **_kwargs):
        if ssh_explodes:
            raise RuntimeError("ssh died")
        return subprocess.CompletedProcess(args=[command], returncode=0, stdout="MI325X", stderr="")

    monkeypatch.setattr(gpu_probe.ssh_client, "ssh_run", _fake_ssh_run)

    gpu_probe.remote_autodetect_gpu_type(timeout_s=1)

    assert created, "the probe should have keyscanned into a scratch dir"
    assert not any(p.exists() for p in created)


def test_infera_probe_parses_output_despite_a_nonzero_exit(monkeypatch, tmp_path):
    """_PROBE_CMD's `||` fallback can print a usable name and still exit non-zero.

    Gating the parse on the exit status would discard that answer, so a non-zero
    return is only logged.
    """
    _track_scratch_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gpu_probe.ssh_client,
        "ssh_run",
        lambda _host, command, **_kw: subprocess.CompletedProcess(
            args=[command], returncode=1, stdout="gfx942", stderr="rocm-smi: not found"
        ),
    )

    assert gpu_probe.remote_autodetect_gpu_type(timeout_s=1) == "mi300x"


def test_rayjob_probe_parses_logs_despite_a_failed_job(monkeypatch):
    """Same rule on the Ray path: a FAILED job may still have logged the name."""
    monkeypatch.setattr(
        gpu_probe,
        "build_external_state_from_env",
        lambda: {"backend": "rayjob", "head_pod_ip": "10.0.0.1"},
    )
    monkeypatch.setattr(
        gpu_probe.ray_dashboard,
        "RayDashboardClient",
        lambda *_a, **_k: _FakeRayClient("AMD Instinct MI325X", status="FAILED"),
    )

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


def test_remote_read_env_rayjob(monkeypatch):
    # printenv on the Ray head returns the pod's SGLANG_USE_AITER value.
    monkeypatch.setattr(
        gpu_probe,
        "build_external_state_from_env",
        lambda: {"backend": "rayjob", "head_pod_ip": "10.0.0.1"},
    )
    monkeypatch.setattr(
        gpu_probe.ray_dashboard,
        "RayDashboardClient",
        # Sentinel-wrapped value interleaved with a Ray INFO line.
        lambda *_a, **_k: _FakeRayClient("INFO Runtime env is setting up.\n___MNENV[0]MNENV___\n"),
    )
    assert gpu_probe.remote_read_env("SGLANG_USE_AITER", timeout_s=1) == "0"


def test_remote_read_env_infera(monkeypatch):
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
    monkeypatch.setattr(gpu_probe.ssh_known_hosts, "refresh_known_hosts", lambda _h, dest: dest)
    monkeypatch.setattr(
        gpu_probe.ssh_client,
        "ssh_run",
        lambda host, command, **_k: subprocess.CompletedProcess(
            args=[command], returncode=0, stdout="___MNENV[0]MNENV___\n", stderr=""
        ),
    )
    assert gpu_probe.remote_read_env("SGLANG_USE_AITER", timeout_s=1) == "0"


def test_remote_read_env_unset_returns_none(monkeypatch):
    # An unset var prints empty sentinel brackets -> None.
    monkeypatch.setattr(
        gpu_probe,
        "build_external_state_from_env",
        lambda: {"backend": "rayjob", "head_pod_ip": "10.0.0.1"},
    )
    monkeypatch.setattr(
        gpu_probe.ray_dashboard,
        "RayDashboardClient",
        lambda *_a, **_k: _FakeRayClient("___MNENV[]MNENV___\n"),
    )
    assert gpu_probe.remote_read_env("SGLANG_USE_AITER", timeout_s=1) is None


def test_parse_gpu_type_prefers_specific_tag():
    # gcnArchName fallback resolves gfx942 -> mi300x runner.
    assert gpu_probe._parse_gpu_type("gfx942:sramecc+:xnack-") == "mi300x"
    # MI325X is not shadowed by the MI300X substring rule.
    assert gpu_probe._parse_gpu_type("AMD Instinct MI325X") == "mi325x"
    assert gpu_probe._parse_gpu_type("nothing here") is None
