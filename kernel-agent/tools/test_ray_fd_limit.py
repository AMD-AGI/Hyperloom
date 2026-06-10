# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for issue #433: at the container default
``ulimit -n`` (1024) the Ray raylet is unstable (SIGABRT on startup /
left as a zombie that only ``ray stop --force`` / SIGKILL can clear).
Ray's raylet opens a large number of fds (sockets, plasma store,
per-worker pipes); on a 384-CPU node 1024 is far too low.

The kernel-agent starts Ray itself via ``ray start --head`` from
``ensure_ray_cluster`` and ``force_restart_local_cluster``. The child
raylet inherits this process's ``RLIMIT_NOFILE``, so the runtime must
run an fd-limit *preflight* that raises the soft limit (to
``min(target, hard)``) BEFORE spawning ``ray start``.

These tests encode that contract:
  1. ``ensure_fd_limit`` raises a too-low soft limit up to the target,
  2. ``ensure_fd_limit`` is a no-op when the soft limit is already high,
  3. ``ensure_fd_limit`` clamps to the hard limit and warns when the
     hard limit itself is below the target (the docker ``--ulimit``
     requirement can only be satisfied at container-launch time), and
  4. both ``ensure_ray_cluster`` and ``force_restart_local_cluster``
     run the preflight BEFORE ``ray start``.

Until the preflight lands, every test below fails (no ``ensure_fd_limit``
symbol / no ``resource.setrlimit`` call before ``ray start``), which is
exactly the on-disk gap described in #433.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
BACKENDS_DIR = TOOLS_DIR / "backends"
for d in (str(TOOLS_DIR), str(BACKENDS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import ray_runtime  # noqa: E402

# Minimum soft RLIMIT_NOFILE the raylet needs to stay up (issue #433).
TARGET_NOFILE = 65536


class _FakeResource:
    """Stand-in for the stdlib ``resource`` module with in-memory rlimits.

    Records every ``getrlimit`` / ``setrlimit`` into ``events`` so a test
    can assert ordering relative to the ``ray start`` subprocess. Mimics
    the kernel's rule that a process may raise its soft limit only up to
    the hard limit (raising the hard limit raises ``ValueError`` here,
    matching the unprivileged container case)."""

    RLIMIT_NOFILE = 7  # value is irrelevant; kept distinct + truthy
    RLIM_INFINITY = -1  # matches the stdlib resource module sentinel

    def __init__(self, soft: int, hard: int, events: list):
        self._soft = soft
        self._hard = hard
        self._events = events

    def _exceeds_hard(self, value: int) -> bool:
        """True when ``value`` is above the current hard cap, treating
        ``RLIM_INFINITY`` (-1) as +infinity on either side."""
        if self._hard == self.RLIM_INFINITY:
            return False  # no ceiling
        if value == self.RLIM_INFINITY:
            return True   # asking for unlimited under a finite cap
        return value > self._hard

    def getrlimit(self, which):
        assert which == self.RLIMIT_NOFILE
        self._events.append(("getrlimit", (self._soft, self._hard)))
        return (self._soft, self._hard)

    def setrlimit(self, which, limits):
        assert which == self.RLIMIT_NOFILE
        soft, hard = limits
        # Unprivileged rule: soft may rise up to the hard cap; the hard cap
        # may be lowered but not raised. RLIM_INFINITY is treated as +inf.
        if self._exceeds_hard(soft) or (hard != self._hard and self._exceeds_hard(hard)):
            raise ValueError("current limit exceeds maximum limit")
        self._soft = soft
        self._hard = hard
        self._events.append(("setrlimit", (soft, hard)))


class _Proc:
    returncode = 0


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

    This is the unprivileged-container case: only ``docker --ulimit
    nofile=...`` at launch can lift the hard cap, so the runtime cannot
    fully fix it — but it must still raise soft as high as allowed and
    surface the shortfall loudly."""
    events: list = []
    fake = _FakeResource(soft=1024, hard=4096, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)

    warnings: list = []
    monkeypatch.setattr(
        ray_runtime, "_fd_limit_warn", lambda msg: warnings.append(msg),
        raising=False,
    )

    soft, hard = ray_runtime.ensure_fd_limit(TARGET_NOFILE)

    assert ("setrlimit", (4096, 4096)) in events
    assert soft == 4096 and hard == 4096
    assert warnings, "must warn when hard cap is below the raylet fd target"


def test_ensure_fd_limit_unlimited_hard_targets_min_soft_without_warning(monkeypatch):
    """An unlimited hard cap (RLIM_INFINITY = -1) must be treated as "no
    ceiling": raise soft to exactly ``min_soft`` (NOT min(min_soft, -1) = -1)
    and emit NO warning. Regressing this re-introduces the issue #433
    boundary bug where -1 was mistaken for a tiny cap."""
    events: list = []
    fake = _FakeResource(soft=1024, hard=_FakeResource.RLIM_INFINITY, events=events)
    monkeypatch.setattr(ray_runtime, "resource", fake, raising=False)

    warnings: list = []
    monkeypatch.setattr(
        ray_runtime, "_fd_limit_warn", lambda msg: warnings.append(msg),
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
        ray_runtime, "_fd_limit_warn", lambda msg: warnings.append(msg),
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
        "ensure_ray_cluster must run an RLIMIT_NOFILE preflight before "
        "starting Ray (issue #433)"
    )
    assert "ray_start" in kinds
    assert kinds.index("setrlimit") < kinds.index("ray_start"), (
        "fd-limit preflight must run BEFORE 'ray start' so the raylet "
        "inherits the raised limit"
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
        "force_restart_local_cluster must run an RLIMIT_NOFILE preflight "
        "before starting Ray (issue #433)"
    )
    assert "ray_start" in kinds
    assert kinds.index("setrlimit") < kinds.index("ray_start")
