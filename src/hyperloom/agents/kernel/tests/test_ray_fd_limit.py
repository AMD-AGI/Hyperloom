# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for Ray fd-limit preflight before ``ray start``.

Ensure the child raylet does not inherit the low container default."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
BACKENDS_DIR = TOOLS_DIR / "backends"
for d in (str(TOOLS_DIR), str(BACKENDS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import ray_runtime  # noqa: E402

# Minimum soft RLIMIT_NOFILE the raylet needs to stay up.
TARGET_NOFILE = 65536
KERNEL_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = KERNEL_ROOT / "scripts" / "install.sh"


def _extract_shell_function(name: str) -> str:
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


class _FakeResource:
    """Stand-in for the stdlib ``resource`` module with in-memory rlimits.

    Records every ``getrlimit`` / ``setrlimit`` into ``events`` so a test
    can assert ordering relative to the ``ray start`` subprocess. A process
    may raise its soft limit only up to the hard limit (raising the hard
    limit raises ``ValueError`` here)."""

    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft: int, hard: int, events: list):
        self._soft = soft
        self._hard = hard
        self._events = events

    def _exceeds_hard(self, value: int) -> bool:
        """True when ``value`` is above the current hard cap, treating
        ``RLIM_INFINITY`` (-1) as +infinity on either side."""
        if self._hard == self.RLIM_INFINITY:
            return False
        if value == self.RLIM_INFINITY:
            return True
        return value > self._hard

    def getrlimit(self, which):
        assert which == self.RLIMIT_NOFILE
        self._events.append(("getrlimit", (self._soft, self._hard)))
        return (self._soft, self._hard)

    def setrlimit(self, which, limits):
        assert which == self.RLIMIT_NOFILE
        soft, hard = limits
        # Soft may rise up to the hard cap; the hard cap may be lowered but not raised.
        if self._exceeds_hard(soft) or (hard != self._hard and self._exceeds_hard(hard)):
            raise ValueError("current limit exceeds maximum limit")
        self._soft = soft
        self._hard = hard
        self._events.append(("setrlimit", (soft, hard)))


class _Proc:
    returncode = 0


def test_install_sh_fd_limit_function_returns_success_when_hard_cap_sufficient():
    """Shell install preflight must not return 1 when the hard cap is high enough."""
    body = f"""
set -euo pipefail
RAY_MIN_NOFILE=65536
FAKE_SOFT=1024
FAKE_HARD=524288
log() {{ :; }}
warn() {{ :; }}
ulimit() {{
  case "$1" in
    -Sn)
      if [ "$#" -eq 1 ]; then
        printf '%s\\n' "$FAKE_SOFT"
      else
        FAKE_SOFT="$2"
      fi
      ;;
    -Hn) printf '%s\\n' "$FAKE_HARD" ;;
    *) return 2 ;;
  esac
}}
{_extract_shell_function("ensure_fd_limit_for_ray")}
ensure_fd_limit_for_ray
"""
    result = subprocess.run(["bash", "-c", body], text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr


def _install_fake_ray_start(monkeypatch, events):
    """Make ``ray status`` report 'down' initially, 'up' after ray start."""
    started = False

    monkeypatch.setattr(ray_runtime, "ray_status_ok", lambda: started)

    def _fake_run(cmd, **kwargs):
        nonlocal started
        if cmd[:2] == ["ray", "start"]:
            events.append(("ray_start", tuple(cmd)))
            started = True
        elif cmd[:2] == ["ray", "stop"]:
            events.append(("ray_stop", tuple(cmd)))
            started = False
        return _Proc()

    monkeypatch.setattr(ray_runtime.subprocess, "run", _fake_run)


def test_ensure_fd_limit_raises_low_soft_limit(monkeypatch):
    """Container default (1024) must be raised to TARGET under a high hard cap."""
    events: list = []
    fake = _FakeResource(soft=1024, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)

    soft, hard = ray_runtime.ensure_fd_limit(TARGET_NOFILE)

    assert ("setrlimit", (TARGET_NOFILE, 1048576)) in events
    assert soft == TARGET_NOFILE
    assert hard == 1048576


def test_ensure_fd_limit_noop_when_already_high(monkeypatch):
    """Already-high soft limit must not trigger a setrlimit."""
    events: list = []
    fake = _FakeResource(soft=1048576, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)

    soft, _hard = ray_runtime.ensure_fd_limit(TARGET_NOFILE)

    assert all(kind != "setrlimit" for kind, _ in events)
    assert soft == 1048576


def test_ensure_fd_limit_clamps_to_low_hard_limit_and_warns(monkeypatch):
    """When the hard cap < target, raise soft to the hard cap and warn.

    Unprivileged-container case: only ``docker --ulimit nofile=...`` can lift
    the hard cap, so the runtime raises soft as high as allowed and warns."""
    events: list = []
    fake = _FakeResource(soft=1024, hard=4096, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)

    warnings: list = []
    monkeypatch.setattr(
        ray_runtime,
        "_fd_limit_warn",
        lambda msg: warnings.append(msg),
        raising=False,
    )

    soft, hard = ray_runtime.ensure_fd_limit(TARGET_NOFILE)

    assert ("setrlimit", (4096, 4096)) in events
    assert soft == 4096 and hard == 4096
    assert warnings, "must warn when hard cap is below the raylet fd target"


def test_ensure_fd_limit_unlimited_hard_targets_min_soft_without_warning(monkeypatch):
    """An unlimited hard cap (RLIM_INFINITY = -1) must be treated as "no
    ceiling": raise soft to exactly ``min_soft`` (NOT min(min_soft, -1) = -1)
    and emit NO warning."""
    events: list = []
    fake = _FakeResource(soft=1024, hard=_FakeResource.RLIM_INFINITY, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)

    warnings: list = []
    monkeypatch.setattr(
        ray_runtime,
        "_fd_limit_warn",
        lambda msg: warnings.append(msg),
        raising=False,
    )

    soft, hard = ray_runtime.ensure_fd_limit(TARGET_NOFILE)

    assert ("setrlimit", (TARGET_NOFILE, _FakeResource.RLIM_INFINITY)) in events
    assert soft == TARGET_NOFILE
    assert hard == _FakeResource.RLIM_INFINITY
    assert not warnings, "unlimited hard cap must NOT trigger a false 'too low' warning"


def test_ensure_fd_limit_unlimited_soft_is_noop(monkeypatch):
    """An already-unlimited soft limit (RLIM_INFINITY = -1) must be treated
    as already-sufficient: no setrlimit, no warning."""
    events: list = []
    fake = _FakeResource(
        soft=_FakeResource.RLIM_INFINITY,
        hard=_FakeResource.RLIM_INFINITY,
        events=events,
    )
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)

    warnings: list = []
    monkeypatch.setattr(
        ray_runtime,
        "_fd_limit_warn",
        lambda msg: warnings.append(msg),
        raising=False,
    )

    soft, _hard = ray_runtime.ensure_fd_limit(TARGET_NOFILE)

    assert all(kind != "setrlimit" for kind, _ in events)
    assert soft == _FakeResource.RLIM_INFINITY
    assert not warnings


def test_ensure_ray_cluster_runs_fd_preflight_before_ray_start(monkeypatch):
    """``ensure_ray_cluster`` must raise the fd limit BEFORE ``ray start``."""
    events: list = []
    fake = _FakeResource(soft=1024, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
    _install_fake_ray_start(monkeypatch, events)

    ray_runtime.ensure_ray_cluster(num_gpus=1)

    kinds = [kind for kind, _ in events]
    assert "setrlimit" in kinds, (
        "ensure_ray_cluster must run an RLIMIT_NOFILE preflight before starting Ray (issue #433)"
    )
    assert "ray_start" in kinds
    assert kinds.index("setrlimit") < kinds.index("ray_start"), (
        "fd-limit preflight must run BEFORE 'ray start' so the raylet inherits the raised limit"
    )


def test_force_restart_local_cluster_runs_fd_preflight_before_ray_start(monkeypatch):
    """``force_restart_local_cluster`` must also raise the fd limit first."""
    events: list = []
    fake = _FakeResource(soft=1024, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
    _install_fake_ray_start(monkeypatch, events)

    ray_runtime.force_restart_local_cluster(num_gpus=1)

    kinds = [kind for kind, _ in events]
    assert "setrlimit" in kinds, (
        "force_restart_local_cluster must run an RLIMIT_NOFILE preflight before starting Ray (issue #433)"
    )
    assert "ray_start" in kinds
    assert kinds.index("setrlimit") < kinds.index("ray_start")


def test_ray_status_timeout_is_treated_as_down(monkeypatch):
    """A stale ``ray_current_cluster`` can make ``ray status`` hang on dead GCS.

    Treat timeout as "no usable cluster" so startup can rebuild a local head
    instead of carrying a stale GCS address into a long optimizer session.
    """
    monkeypatch.delenv("HYPERLOOM_RAY_STATUS_TIMEOUT_SEC", raising=False)
    calls: list = []

    def _timeout_run(cmd, **kwargs):
        calls.append((tuple(cmd), kwargs))
        raise subprocess.TimeoutExpired(cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(ray_runtime.subprocess, "run", _timeout_run)

    assert ray_runtime.ray_status_ok() is False
    assert calls
    assert calls[0][1].get("timeout") == ray_runtime.DEFAULT_RAY_STATUS_TIMEOUT_SEC


def test_ensure_ray_cluster_clears_stale_ray_before_start(monkeypatch):
    """When discovery is down, clear stale Ray state before starting a fresh head."""
    events: list = []
    fake = _FakeResource(soft=1048576, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
    _install_fake_ray_start(monkeypatch, events)

    ray_runtime.ensure_ray_cluster(num_gpus=1)

    kinds = [kind for kind, _ in events]
    assert "ray_stop" in kinds
    assert "ray_start" in kinds
    assert kinds.index("ray_stop") < kinds.index("ray_start")


def test_ensure_ray_cluster_binds_dashboard_to_loopback(monkeypatch):
    """The local head must bind the dashboard/jobs API to loopback, not 0.0.0.0,
    so the unauthenticated Ray Jobs endpoint is not exposed on the pod network."""
    events: list = []
    fake = _FakeResource(soft=1048576, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
    _install_fake_ray_start(monkeypatch, events)

    ray_runtime.ensure_ray_cluster()

    starts = [cmd for kind, cmd in events if kind == "ray_start"]
    assert starts, "ray start was not invoked"
    assert "--dashboard-host=127.0.0.1" in starts[0]
    assert "--dashboard-host=0.0.0.0" not in starts[0]


def test_force_restart_local_cluster_binds_dashboard_to_loopback(monkeypatch):
    """``force_restart_local_cluster`` must also bind the dashboard to loopback."""
    events: list = []
    fake = _FakeResource(soft=1048576, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
    _install_fake_ray_start(monkeypatch, events)

    ray_runtime.force_restart_local_cluster()

    starts = [cmd for kind, cmd in events if kind == "ray_start"]
    assert starts, "ray start was not invoked"
    assert "--dashboard-host=127.0.0.1" in starts[0]
    assert "--dashboard-host=0.0.0.0" not in starts[0]


def _assert_declares_serving_slot(start_cmd: tuple) -> None:
    """The head must declare the ``serving_slot`` custom resource."""
    assert "--resources" in start_cmd, start_cmd
    idx = start_cmd.index("--resources")
    payload = json.loads(start_cmd[idx + 1])
    assert payload.get("serving_slot") == 1, payload


def test_ensure_ray_cluster_declares_serving_slot(monkeypatch):
    """The single-node head declares serving_slot=1 (authoritative serving mutex)."""
    events: list = []
    fake = _FakeResource(soft=1048576, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
    _install_fake_ray_start(monkeypatch, events)

    ray_runtime.ensure_ray_cluster(num_gpus=4)

    starts = [cmd for kind, cmd in events if kind == "ray_start"]
    assert starts, "ray start was not invoked"
    _assert_declares_serving_slot(starts[0])
    # num_gpus is still passed alongside the custom resource.
    assert "--num-gpus=4" in starts[0]


def test_force_restart_local_cluster_declares_serving_slot(monkeypatch):
    """A version-mismatch restart re-declares serving_slot on the fresh head."""
    events: list = []
    fake = _FakeResource(soft=1048576, hard=1048576, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
    _install_fake_ray_start(monkeypatch, events)

    ray_runtime.force_restart_local_cluster()

    starts = [cmd for kind, cmd in events if kind == "ray_start"]
    assert starts, "ray start was not invoked"
    _assert_declares_serving_slot(starts[0])


import pytest  # noqa: E402


_ISO_ENV_VARS = ("HL_RAY_HEAD_PORT", "RAY_ADDRESS")


def _arg_value(start_cmd: tuple, flag: str):
    """Return the ``=value`` of a ``--flag=value`` token in a ray start argv."""
    for tok in start_cmd:
        if tok.startswith(f"{flag}="):
            return tok.split("=", 1)[1]
    return None


class TestLocalHeadPortIsolation:
    """Free-port isolation for spur host-network co-location.

    Co-scheduled sessions share only the host network, so the sole collision is
    Ray's fixed default ports (GCS 6379 / dashboard 8265 / client 10001): the
    later head connects to the earlier head's GCS and aborts with a session-name
    mismatch. Each head is bound to FREE probed ports; rendezvous is via the
    container-private ``/tmp/ray/ray_current_cluster`` (no ``--temp-dir``, no
    ``RAY_ADDRESS`` pin), so ``ray.init(address="auto")`` still discovers it.
    """

    @pytest.fixture(autouse=True)
    def _clean_ray_env(self, monkeypatch):
        """Scrub override/address env so each case is deterministic."""
        for var in _ISO_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_free_tcp_port_is_bindable(self):
        """The probed port is a real, currently-free loopback TCP port."""
        import socket

        port = ray_runtime._free_tcp_port()
        assert isinstance(port, int) and 0 < port < 65536
        # It was released, so we can immediately bind it ourselves.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))

    def test_isolated_ports_are_free_and_distinct(self):
        """GCS / dashboard / client all get distinct probed ports by default."""
        gcs, extra = ray_runtime._isolated_head_port_args()
        dash = int(_arg_value(tuple(extra), "--dashboard-port"))
        client = int(_arg_value(tuple(extra), "--ray-client-server-port"))
        assert len({gcs, dash, client}) == 3
        for p in (gcs, dash, client):
            assert 0 < p < 65536

    def test_no_ray_address_pinned(self, monkeypatch):
        """We must NOT set RAY_ADDRESS: ray.init(address='auto') does discovery."""
        import os as _os

        ray_runtime._isolated_head_port_args()
        assert "RAY_ADDRESS" not in _os.environ

    def test_head_port_override_pins_gcs_only(self, monkeypatch):
        """HL_RAY_HEAD_PORT pins the GCS port; dashboard/client stay probed."""
        monkeypatch.setenv("HL_RAY_HEAD_PORT", "6500")
        gcs, extra = ray_runtime._isolated_head_port_args()
        assert gcs == 6500
        assert _arg_value(tuple(extra), "--dashboard-port") is not None
        assert _arg_value(tuple(extra), "--ray-client-server-port") is not None

    def test_ensure_ray_cluster_binds_isolated_ports(self, monkeypatch):
        """End-to-end: ensure_ray_cluster emits probed GCS/dashboard/client ports."""
        events: list = []
        fake = _FakeResource(soft=1048576, hard=1048576, events=events)
        monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)
        _install_fake_ray_start(monkeypatch, events)

        ray_runtime.ensure_ray_cluster(num_gpus=8)

        start = [cmd for kind, cmd in events if kind == "ray_start"][0]
        assert _arg_value(start, "--port") is not None
        assert _arg_value(start, "--dashboard-port") is not None
        assert _arg_value(start, "--ray-client-server-port") is not None
        # No --temp-dir (issue #55244); dashboard bound to loopback (security).
        assert not any(t.startswith("--temp-dir=") for t in start)
        assert "--dashboard-host=127.0.0.1" in start
        assert "--num-gpus=8" in start
        _assert_declares_serving_slot(start)
