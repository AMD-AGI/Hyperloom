# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ray lease and slot probes absorb Ray and OS failures; any other failure surfaces."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import _ray_backend as rb
from hyperloom.orchestrator.actions.executors import _ray_serving as rs


class _RayError(Exception):
    """Stands in for ``ray.exceptions.RayError``."""


class _GetTimeoutError(_RayError):
    """Stands in for ``ray.exceptions.GetTimeoutError``."""


def _install_ray(monkeypatch: pytest.MonkeyPatch, **api: Any) -> None:
    exceptions = SimpleNamespace(RayError=_RayError, GetTimeoutError=_GetTimeoutError)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(exceptions=exceptions, **api))


def _raising(error: BaseException):
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return _raise


_EXPECTED = [_RayError("ray down"), OSError("socket closed")]


def _actor(**methods: Any) -> SimpleNamespace:
    return SimpleNamespace(**{name: SimpleNamespace(remote=lambda *a, _v=value: _v) for name, value in methods.items()})


@pytest.fixture
def lease() -> rs.ServingLease:
    return rs.ServingLease(num_gpus=1)


@pytest.fixture
def specialist() -> rs.GpuSpecialistLease:
    held = rs.GpuSpecialistLease(num_gpus=1)
    held._actor = _actor(is_alive=True, exit_code=3)
    return held


@pytest.mark.parametrize("error", _EXPECTED)
def test_slot_probe_reads_an_unreachable_cluster_as_not_busy(monkeypatch, error) -> None:
    monkeypatch.setattr(rb, "ray_gpu_specialist_exec_enabled", lambda: True)
    _install_ray(monkeypatch, is_initialized=lambda: True, available_resources=_raising(error))

    assert rb.serving_slot_busy() is False


def test_slot_probe_lets_an_unexpected_failure_propagate(monkeypatch) -> None:
    monkeypatch.setattr(rb, "ray_gpu_specialist_exec_enabled", lambda: True)
    _install_ray(monkeypatch, is_initialized=lambda: True, available_resources=_raising(RuntimeError("probe bug")))

    with pytest.raises(RuntimeError, match="probe bug"):
        rb.serving_slot_busy()


@pytest.mark.parametrize("error", _EXPECTED)
def test_abandoning_a_round_logs_a_failed_cancel(monkeypatch, caplog, lease, error) -> None:
    _install_ray(monkeypatch, cancel=_raising(error))

    with caplog.at_level(logging.WARNING, logger=rs.log.name):
        assert lease._abandon_ref(object()) is None

    assert [r.getMessage() for r in caplog.records] == [
        f"ServingLease: could not cancel the abandoned round: {error!r}"
    ]


def test_abandoning_a_round_lets_an_unexpected_failure_propagate(monkeypatch, lease) -> None:
    _install_ray(monkeypatch, cancel=_raising(TypeError("not an object ref")))

    with pytest.raises(TypeError, match="not an object ref"):
        lease._abandon_ref(object())


@pytest.mark.parametrize("error", _EXPECTED)
def test_an_unreachable_actor_cannot_cancel_its_round(monkeypatch, lease, error) -> None:
    _install_ray(monkeypatch, get=_raising(error))
    lease._actor = _actor(cancel_round="ref")

    assert lease._ask_actor_to_cancel("stop") is False


def test_cancelling_a_round_lets_an_unexpected_failure_propagate(monkeypatch, lease) -> None:
    _install_ray(monkeypatch, get=_raising(RuntimeError("cancel bug")))
    lease._actor = _actor(cancel_round="ref")

    with pytest.raises(RuntimeError, match="cancel bug"):
        lease._ask_actor_to_cancel("stop")


@pytest.mark.parametrize("error", _EXPECTED)
def test_close_kills_the_actor_when_the_stop_request_fails(monkeypatch, lease, error) -> None:
    killed: list[object] = []
    _install_ray(monkeypatch, get=_raising(error), kill=killed.append)
    actor = _actor(stop="ref")
    lease._actor = actor

    lease.close()

    assert killed == [actor]
    assert lease._actor is None


def test_close_lets_an_unexpected_stop_failure_propagate(monkeypatch, lease) -> None:
    killed: list[object] = []
    _install_ray(monkeypatch, get=_raising(RuntimeError("stop bug")), kill=killed.append)
    lease._actor = _actor(stop="ref")

    with pytest.raises(RuntimeError, match="stop bug"):
        lease.close()


@pytest.mark.parametrize("error", _EXPECTED)
def test_kill_clears_the_handle_when_ray_fails(monkeypatch, lease, error) -> None:
    _install_ray(monkeypatch, kill=_raising(error))
    lease._actor = object()

    lease._kill_actor()

    assert lease._actor is None


def test_kill_lets_an_unexpected_failure_propagate(monkeypatch, lease) -> None:
    _install_ray(monkeypatch, kill=_raising(ValueError("not an actor handle")))
    lease._actor = object()

    with pytest.raises(ValueError, match="not an actor handle"):
        lease._kill_actor()


@pytest.mark.parametrize("error", _EXPECTED)
def test_a_specialist_actor_that_cannot_be_reached_is_not_alive_and_has_no_exit_code(
    monkeypatch, specialist, error
) -> None:
    _install_ray(monkeypatch, get=_raising(error))

    assert (specialist.is_alive(), specialist.exit_code()) == (False, None)


def test_a_specialist_liveness_probe_timeout_reads_as_alive(monkeypatch, specialist) -> None:
    _install_ray(monkeypatch, get=_raising(_GetTimeoutError("slow")))

    assert specialist.is_alive() is True


def test_specialist_liveness_lets_an_unexpected_failure_propagate(monkeypatch, specialist) -> None:
    _install_ray(monkeypatch, get=_raising(RuntimeError("probe bug")))

    with pytest.raises(RuntimeError, match="probe bug"):
        specialist.is_alive()


def test_specialist_exit_code_lets_an_unexpected_failure_propagate(monkeypatch, specialist) -> None:
    _install_ray(monkeypatch, get=_raising(RuntimeError("probe bug")))

    with pytest.raises(RuntimeError, match="probe bug"):
        specialist.exit_code()
