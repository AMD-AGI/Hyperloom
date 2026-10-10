# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A pre-existing Ray head without ``serving_slot`` is repaired only when it is a lone, idle, local head."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import _ray_backend as rb
from hyperloom.orchestrator.actions.executors import _ray_runtime as rr
from hyperloom.orchestrator.actions.executors import _ray_serving as rs

_LOCAL_NODE = "node-local"


class _FakeRay:
    """The slice of the ``ray`` API the feasibility check and the restart guard read."""

    def __init__(
        self,
        *,
        has_serving_slot: bool = False,
        nodes: list[dict[str, Any]] | None = None,
        available: dict[str, float] | None = None,
    ) -> None:
        self.has_serving_slot = has_serving_slot
        self._nodes = nodes if nodes is not None else [{"NodeID": _LOCAL_NODE, "Alive": True}]
        self._available = available
        self.shutdowns = 0

    def cluster_resources(self) -> dict[str, float]:
        res = {"CPU": 64.0, "GPU": 8.0, "memory": 1e12, f"node:{_LOCAL_NODE}": 1.0}
        if self.has_serving_slot:
            res["serving_slot"] = 1.0
        return res

    def available_resources(self) -> dict[str, float]:
        if self._available is not None:
            return dict(self._available)
        return self.cluster_resources()

    def nodes(self) -> list[dict[str, Any]]:
        return list(self._nodes)

    def get_runtime_context(self) -> Any:
        class _Ctx:
            def get_node_id(self) -> str:
                return _LOCAL_NODE

        return _Ctx()

    def shutdown(self) -> None:
        self.shutdowns += 1


class _StubBackend:
    """Stands in for the process-wide backend; a restart makes the fake head declare the slot."""

    def __init__(self, fake: _FakeRay) -> None:
        self.fake = fake
        self.restarts = 0

    def ensure(self, **_kw: Any) -> None:
        return None

    def restart_local_head(self, log_path: Any = None) -> None:
        self.restarts += 1
        self.fake.has_serving_slot = True


@pytest.fixture
def single_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", lambda: False)


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeRay) -> _StubBackend:
    backend = _StubBackend(fake)
    monkeypatch.setitem(sys.modules, "ray", fake)
    monkeypatch.setattr(rb, "get_ray_backend", lambda: backend)
    return backend


def test_local_head_without_slot_is_restarted_then_feasible(monkeypatch, single_node):
    fake = _FakeRay(has_serving_slot=False)
    backend = _install(monkeypatch, fake)

    rs._ensure_cluster_feasible(num_gpus=1, serving_slot=True)

    assert backend.restarts == 1
    assert fake.has_serving_slot


def test_head_with_slot_is_never_restarted(monkeypatch, single_node):
    backend = _install(monkeypatch, _FakeRay(has_serving_slot=True))

    rs._ensure_cluster_feasible(num_gpus=1, serving_slot=True)

    assert backend.restarts == 0


def test_lease_that_needs_no_slot_never_restarts(monkeypatch, single_node):
    backend = _install(monkeypatch, _FakeRay(has_serving_slot=False))

    rs._ensure_cluster_feasible(num_gpus=1, serving_slot=False)

    assert backend.restarts == 0


@pytest.mark.parametrize(
    "setup, why",
    [
        (lambda mp: mp.setenv("RAY_ADDRESS", "10.0.0.5:6379"), "explicit cluster"),
        (
            lambda mp: mp.setattr(
                "hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", lambda: True
            ),
            "multi-node",
        ),
    ],
)
def test_external_or_multi_node_cluster_keeps_the_error(monkeypatch, single_node, setup, why):
    setup(monkeypatch)
    backend = _install(monkeypatch, _FakeRay(has_serving_slot=False))

    with pytest.raises(rs.RayMissingServingSlotError) as excinfo:
        rs._ensure_cluster_feasible(num_gpus=1, serving_slot=True)

    assert backend.restarts == 0
    assert why in str(excinfo.value)
    assert rs.RAY_INFEASIBLE_MARKER in str(excinfo.value)


def test_cluster_with_a_second_live_node_keeps_the_error(monkeypatch, single_node):
    fake = _FakeRay(
        has_serving_slot=False,
        nodes=[{"NodeID": _LOCAL_NODE, "Alive": True}, {"NodeID": "node-remote", "Alive": True}],
    )
    backend = _install(monkeypatch, fake)

    with pytest.raises(rs.RayMissingServingSlotError, match="2 live nodes"):
        rs._ensure_cluster_feasible(num_gpus=1, serving_slot=True)
    assert backend.restarts == 0


def test_head_on_another_host_keeps_the_error(monkeypatch, single_node):
    fake = _FakeRay(has_serving_slot=False, nodes=[{"NodeID": "node-remote", "Alive": True}])
    backend = _install(monkeypatch, fake)

    with pytest.raises(rs.RayMissingServingSlotError, match="not this host"):
        rs._ensure_cluster_feasible(num_gpus=1, serving_slot=True)
    assert backend.restarts == 0


def test_head_with_gpus_in_use_keeps_the_error(monkeypatch, single_node):
    fake = _FakeRay(has_serving_slot=False, available={"CPU": 64.0, "GPU": 6.0, "memory": 1e12})
    backend = _install(monkeypatch, fake)

    with pytest.raises(rs.RayMissingServingSlotError, match="in use"):
        rs._ensure_cluster_feasible(num_gpus=1, serving_slot=True)
    assert backend.restarts == 0


def test_serving_lease_round_runs_after_the_repair(monkeypatch, single_node):
    """End to end through ``ServingLease.ensure``: the repair happens before the actor is made."""
    fake = _FakeRay(has_serving_slot=False)
    backend = _install(monkeypatch, fake)
    made: list[tuple[float, bool]] = []
    monkeypatch.setattr(rs, "make_serving_actor", lambda n, *, serving_slot: made.append((n, serving_slot)) or object())

    rs.ServingLease(num_gpus=2, serving_slot=True).ensure()

    assert backend.restarts == 1
    assert made == [(2.0, True)]


def test_restart_stops_then_starts_with_the_slot_and_reconnects(monkeypatch):
    """The runtime repair is the installer's: ray stop --force, then a head that declares serving_slot."""
    fake = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    calls: list[Any] = []
    monkeypatch.setattr(
        rr, "force_restart_local_cluster", lambda **kw: calls.append(("restart", kw["num_gpus"], kw["reason"]))
    )
    monkeypatch.setattr(rr, "ray_status_ok", lambda: True)
    monkeypatch.setattr(rr, "quiet_ray_init", lambda **kw: calls.append(("init", kw["num_gpus"])))

    rr.restart_local_head_with_serving_slot(num_gpus=4)

    assert fake.shutdowns == 1
    assert calls[0][0] == "restart" and calls[0][1] == 4 and "serving_slot" in calls[0][2]
    assert calls[1] == ("init", 4)


def test_force_restart_start_command_declares_the_slot(monkeypatch):
    starts: list[list[str]] = []

    def _run(cmd, **_kw):
        if cmd[:2] == ["ray", "start"]:
            starts.append(list(cmd))

        class _P:
            returncode = 0

        return _P()

    monkeypatch.setattr(rr.subprocess, "run", _run)
    monkeypatch.setattr(rr, "ensure_fd_limit", lambda **_kw: (65536, 65536))

    rr.force_restart_local_cluster(num_gpus=2, reason="test")

    assert len(starts) == 1
    assert "--resources" in starts[0]
    assert '"serving_slot": 1' in starts[0][starts[0].index("--resources") + 1]
