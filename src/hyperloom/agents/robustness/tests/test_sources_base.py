# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for sources/base.py."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from hyperloom.agents.robustness.sources.base import (
    DegradeRouter,
    HealthState,
    SourceData,
    SourceUnavailable,
)


@dataclass
class _FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedSource:
    """Source double whose ``fetch`` plays out a queued list of outcomes."""

    def __init__(self, name: str, script: list[object]):
        self.name = name
        self._script = list(script)
        self.calls = 0

    async def fetch(self, ctx: object) -> SourceData:
        self.calls += 1
        if not self._script:
            raise IndexError(f"{self.name}: no scripted outcomes left")
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, SourceData)
        return outcome


def _data(label: str) -> SourceData:
    return SourceData(local_gpu={"from": label})


@pytest.mark.asyncio
async def test_router_happy_path_uses_primary():
    clock = _FakeClock()
    primary = _ScriptedSource("server", [_data("server"), _data("server"), _data("server")])
    router = DegradeRouter(primary, clock=clock)
    for _ in range(3):
        snap = await router.collect(ctx=None)
        assert snap.local_gpu == {"from": "server"}
        assert snap.local_processes_known is True
    assert primary.calls == 3


@pytest.mark.asyncio
async def test_router_degrades_after_threshold_failures(caplog):
    clock = _FakeClock()
    primary = _ScriptedSource(
        "server",
        [
            SourceUnavailable("first"),
            SourceUnavailable("second"),
            SourceUnavailable("third"),
            SourceUnavailable("fourth"),
        ],
    )
    router = DegradeRouter(primary, fail_threshold=3, recheck_interval_s=30, clock=clock)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            _ = await router.collect(ctx=None)
        # After 3rd failure primary is DEGRADED; next tick skips primary within recheck window.
        snap_post = await router.collect(ctx=None)

    assert primary.calls == 3, "primary should not be retried inside recheck window"
    assert router._state is HealthState.DEGRADED
    assert snap_post.local_processes_known is False, "blind tick must mark processes unknown"

    transitions = [r for r in caplog.records if "state healthy -> degraded" in r.getMessage()]
    assert len(transitions) == 1, "single WARN on transition only"


@pytest.mark.asyncio
async def test_router_recovers_after_recheck_window(caplog):
    clock = _FakeClock()
    primary = _ScriptedSource(
        "server",
        [
            SourceUnavailable("a"),
            SourceUnavailable("b"),
            SourceUnavailable("c"),
            _data("server"),
        ],
    )
    router = DegradeRouter(primary, fail_threshold=3, recheck_interval_s=30, clock=clock)

    for _ in range(3):
        await router.collect(ctx=None)
    assert router._state is HealthState.DEGRADED

    # Inside recheck window: primary not probed.
    clock.advance(10.0)
    snap_blind = await router.collect(ctx=None)
    assert primary.calls == 3
    assert snap_blind.local_processes_known is False

    # Past recheck window: primary probed, succeeds, state HEALTHY.
    clock.advance(25.0)
    with caplog.at_level(logging.WARNING):
        snap = await router.collect(ctx=None)
    assert primary.calls == 4
    assert router._state is HealthState.HEALTHY
    assert snap.local_gpu == {"from": "server"}
    assert snap.local_processes_known is True
    transitions = [r for r in caplog.records if "state degraded -> healthy" in r.getMessage()]
    assert len(transitions) == 1


@pytest.mark.asyncio
async def test_router_returns_empty_when_source_unavailable():
    clock = _FakeClock()
    primary = _ScriptedSource("server", [SourceUnavailable("p")])
    router = DegradeRouter(primary, fail_threshold=1, recheck_interval_s=0, clock=clock)
    snap = await router.collect(ctx=None)
    assert snap.local_processes_known is False


@pytest.mark.asyncio
async def test_router_treats_unexpected_exception_as_failure(caplog):
    clock = _FakeClock()

    class Boom(Exception):
        pass

    primary = _ScriptedSource("server", [Boom("oops"), Boom("oops"), Boom("oops")])
    router = DegradeRouter(primary, fail_threshold=3, recheck_interval_s=10, clock=clock)
    with caplog.at_level(logging.ERROR):
        for _ in range(3):
            snap = await router.collect(ctx=None)
            assert snap.local_processes_known is False
    assert router._state is HealthState.DEGRADED
    assert any("source server raised unexpectedly" in r.getMessage() for r in caplog.records)
