# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for Ray fd-limit preflight before ``ray start``.

Ensure the child raylet does not inherit the low container default."""

from __future__ import annotations

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
    """Make ``ray status`` report 'down' and capture ``ray start``."""
    monkeypatch.setattr(ray_runtime, "ray_status_ok", lambda: False)

    def _fake_run(cmd, **kwargs):
        if cmd[:2] == ["ray", "start"]:
            events.append(("ray_start", tuple(cmd)))
        elif cmd[:2] == ["ray", "stop"]:
            events.append(("ray_stop", tuple(cmd)))
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
