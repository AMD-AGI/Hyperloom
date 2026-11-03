# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ray lease and slot probes absorb Ray and OS failures; any other failure surfaces."""

from __future__ import annotations

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


def _actor(**methods: Any) -> SimpleNamespace:
    return SimpleNamespace(**{name: SimpleNamespace(remote=lambda *a, _v=value: _v) for name, value in methods.items()})


def _specialist() -> rs.GpuSpecialistLease:
    held = rs.GpuSpecialistLease(num_gpus=1)
    held._actor = _actor(is_alive=True, exit_code=3)
    return held


def _slot_busy(monkeypatch, error):
    monkeypatch.setattr(rb, "ray_exec_enabled", lambda: True)
    _install_ray(monkeypatch, is_initialized=lambda: True, available_resources=_raising(error))
    return rb.serving_slot_busy()


def _abandon_ref(monkeypatch, error):
    _install_ray(monkeypatch, cancel=_raising(error))
    return rs.ServingLease(num_gpus=1)._abandon_ref(object())


def _ask_to_cancel(monkeypatch, error):
    _install_ray(monkeypatch, get=_raising(error))
    lease = rs.ServingLease(num_gpus=1)
    lease._actor = _actor(cancel_round="ref")
    return lease._ask_actor_to_cancel("stop")


def _close_after_failed_stop(monkeypatch, error):
    """Returns whether the actor was killed and the handle cleared."""
    killed: list[object] = []
    _install_ray(monkeypatch, get=_raising(error), kill=killed.append)
    lease = rs.ServingLease(num_gpus=1)
    actor = _actor(stop="ref")
    lease._actor = actor
    lease.close()
    return killed == [actor] and lease._actor is None


def _kill_actor(monkeypatch, error):
    _install_ray(monkeypatch, kill=_raising(error))
    lease = rs.ServingLease(num_gpus=1)
    lease._actor = object()
    lease._kill_actor()
    return lease._actor


def _specialist_is_alive(monkeypatch, error):
    _install_ray(monkeypatch, get=_raising(error))
    return _specialist().is_alive()


def _specialist_exit_code(monkeypatch, error):
    _install_ray(monkeypatch, get=_raising(error))
    return _specialist().exit_code()


# site -> (call, result when Ray or the OS fails)
_SITES = {
    "slot_busy": (_slot_busy, False),
    "abandon_ref": (_abandon_ref, None),
    "ask_to_cancel": (_ask_to_cancel, False),
    "close": (_close_after_failed_stop, True),
    "kill_actor": (_kill_actor, None),
    "is_alive": (_specialist_is_alive, False),
    "exit_code": (_specialist_exit_code, None),
}


@pytest.mark.parametrize("error", [_RayError("ray down"), OSError("socket closed")], ids=["ray", "os"])
@pytest.mark.parametrize("site", list(_SITES))
def test_ray_and_os_failures_are_absorbed(monkeypatch, site, error) -> None:
    call, expected = _SITES[site]
    outcome = call(monkeypatch, error)
    assert outcome == expected


@pytest.mark.parametrize("site", list(_SITES))
def test_an_unexpected_failure_propagates(monkeypatch, site) -> None:
    call, _ = _SITES[site]
    with pytest.raises(RuntimeError, match="unexpected ray failure"):
        call(monkeypatch, RuntimeError("unexpected ray failure"))
