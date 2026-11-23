# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for reading a multi-node cluster's cards remotely."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest

import hyperloom_common
from hyperloom_common import gpu_card
from hyperloom_common.amd_smi import AmdSmiError
from hyperloom.inference_optimizer.multi_node._internal import gpu_probe
from hyperloom.orchestrator.tests._fixtures import MI325X_CARD

#: What ``amd-smi static --json`` printed on a four-card MI325X node.
_MI325X_STATIC = (
    Path(hyperloom_common.__file__).parent / "tests" / "fixtures" / "amd_smi_static_mi325x.json"
).read_text(encoding="utf-8")

_RAY_STATE = {"backend": "rayjob", "head_pod_ip": "10.0.0.1"}


@pytest.fixture(autouse=True)
def _has_handoff(monkeypatch):
    # external_service_url() must return non-empty or there is no cluster to read.
    monkeypatch.setattr(gpu_probe, "external_service_url", lambda: "http://claw-ray:8888")


class _FakeRayClient:
    """Minimal RayDashboardClient stand-in for the rayjob probe path."""

    def __init__(self, logs: str, *, status: str = "SUCCEEDED") -> None:
        self._logs = logs
        self._status = status
        self.entrypoints: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def submit_job(self, entrypoint):
        self.entrypoints.append(entrypoint)
        return "sub-1"

    def get_job(self, _sub_id):
        return {"status": self._status}

    def get_job_logs(self, _sub_id):
        return self._logs


def _ray(monkeypatch, logs: str, *, status: str = "SUCCEEDED") -> _FakeRayClient:
    client = _FakeRayClient(logs, status=status)
    monkeypatch.setattr(gpu_probe, "build_external_state_from_env", lambda: _RAY_STATE)
    monkeypatch.setattr(gpu_probe.ray_dashboard, "RayDashboardClient", lambda *_a, **_k: client)
    return client


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


def test_the_ray_head_is_asked_with_each_probe_in_turn_and_reports_its_cards(monkeypatch):
    client = _ray(monkeypatch, _MI325X_STATIC)

    assert gpu_probe.remote_cards(timeout_s=1) == [MI325X_CARD] * 4
    assert client.entrypoints == [" || ".join(gpu_card.PROBES)]


def test_an_infera_pod_reports_its_cards_over_ssh(monkeypatch, tmp_path):
    _track_scratch_dirs(monkeypatch, tmp_path)

    def _fake_ssh_run(target, command, **_kwargs):
        assert (target.host, command) == ("10.0.0.2", " || ".join(gpu_card.PROBES))
        return subprocess.CompletedProcess(args=[command], returncode=0, stdout=_MI325X_STATIC, stderr="")

    monkeypatch.setattr(gpu_probe.ssh_client, "ssh_run", _fake_ssh_run)

    assert gpu_probe.remote_cards(timeout_s=1) == [MI325X_CARD] * 4


@pytest.mark.parametrize("ssh_explodes", [False, True])
def test_infera_probe_removes_its_scratch_known_hosts(monkeypatch, tmp_path, ssh_explodes):
    """The one-shot keyscan dir must not accumulate under /tmp, even on failure."""
    created = _track_scratch_dirs(monkeypatch, tmp_path)

    def _fake_ssh_run(_host, command, **_kwargs):
        if ssh_explodes:
            raise RuntimeError("ssh died")
        return subprocess.CompletedProcess(args=[command], returncode=0, stdout=_MI325X_STATIC, stderr="")

    monkeypatch.setattr(gpu_probe.ssh_client, "ssh_run", _fake_ssh_run)

    if ssh_explodes:
        with pytest.raises(AmdSmiError, match="ssh died"):
            gpu_probe.remote_cards(timeout_s=1)
    else:
        gpu_probe.remote_cards(timeout_s=1)

    assert created, "the probe should have keyscanned into a scratch dir"
    assert not any(p.exists() for p in created)


def test_a_failed_remote_amd_smi_is_unreadable_and_says_why(monkeypatch, tmp_path):
    """No fallback probe stands behind amd-smi, so its failure is the answer."""
    _track_scratch_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gpu_probe.ssh_client,
        "ssh_run",
        lambda _host, command, **_kw: subprocess.CompletedProcess(
            args=[command], returncode=127, stdout="", stderr="amd-smi: command not found"
        ),
    )

    with pytest.raises(AmdSmiError, match="exited 127: amd-smi: command not found"):
        gpu_probe.remote_cards(timeout_s=1)


def test_a_failed_ray_job_is_unreadable(monkeypatch):
    _ray(monkeypatch, _MI325X_STATIC, status="FAILED")

    with pytest.raises(AmdSmiError, match="ended FAILED"):
        gpu_probe.remote_cards(timeout_s=1)


@pytest.mark.parametrize("logs", ["", "amd-smi: no GPU found", '{"gpu_data": []}'])
def test_output_that_describes_no_card_is_unreadable(monkeypatch, logs):
    _ray(monkeypatch, logs)

    with pytest.raises(AmdSmiError, match="no GPU is described"):
        gpu_probe.remote_cards(timeout_s=1)


_BACKENDS = [_infera_state(), _RAY_STATE]


def _probe_failing_with(monkeypatch, tmp_path, failure, state):
    _track_scratch_dirs(monkeypatch, tmp_path)

    def _fail(*_a, **_k):
        raise failure

    monkeypatch.setattr(gpu_probe.ssh_client, "ssh_run", _fail)
    monkeypatch.setattr(gpu_probe.ray_dashboard, "RayDashboardClient", _fail)
    monkeypatch.setattr(gpu_probe, "build_external_state_from_env", lambda: state)
    return gpu_probe.remote_cards(timeout_s=1)


@pytest.mark.parametrize("state", _BACKENDS)
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("dashboard unreachable"),
        OSError("no route to host"),
        RuntimeError("dashboard error"),
        subprocess.TimeoutExpired(["ssh"], 1),
    ],
    ids=["http-error", "os-error", "runtime-error", "timeout"],
)
def test_expected_transport_failures_are_unreadable_cards(monkeypatch, tmp_path, failure, state):
    with pytest.raises(AmdSmiError, match=f"via {state['backend']} failed"):
        _probe_failing_with(monkeypatch, tmp_path, failure, state)


@pytest.mark.parametrize("state", _BACKENDS)
def test_an_unexpected_probe_error_propagates(monkeypatch, tmp_path, state):
    with pytest.raises(KeyError, match="status"):
        _probe_failing_with(monkeypatch, tmp_path, KeyError("status"), state)


@pytest.mark.parametrize(
    "state",
    [{"backend": "rayjob"}, {"backend": "infera", "prefill_pods": [{"podIP": "10.0.0.2"}]}, {"backend": "slurm"}],
    ids=["ray-without-head", "infera-without-key", "unsupported-backend"],
)
def test_a_hand_off_the_probe_cannot_reach_is_unreadable(monkeypatch, state):
    monkeypatch.setattr(gpu_probe, "build_external_state_from_env", lambda: state)

    with pytest.raises(AmdSmiError):
        gpu_probe.remote_cards(timeout_s=1)


def test_no_handoff_is_unreadable(monkeypatch):
    # Without an external service URL there is no cluster to read.
    monkeypatch.setattr(gpu_probe, "external_service_url", lambda: "")
    with pytest.raises(AmdSmiError, match="no multi-node cluster"):
        gpu_probe.remote_cards(timeout_s=1)


def test_a_failing_state_build_propagates(monkeypatch):
    def _boom():
        raise RuntimeError("bad hand-off env")

    monkeypatch.setattr(gpu_probe, "build_external_state_from_env", _boom)
    with pytest.raises(RuntimeError, match="bad hand-off env"):
        gpu_probe.remote_cards(timeout_s=1)
