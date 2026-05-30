"""Unit tests for sources/base.py.

Covers DegradeRouter state machine + SourceData merge semantics.
"""

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
    """Source double whose ``fetch`` plays out a queued list of outcomes.

    Each entry is either a :class:`SourceData` (returned) or an
    exception instance (raised). Out-of-script calls raise
    ``IndexError`` so tests catch unintended invocations.
    """

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
    return SourceData(session_summary={"from": label})


# ---------------------------------------------------------------------------
# Source data merging
# ---------------------------------------------------------------------------

def test_source_data_merge_preserves_existing_fields():
    primary = SourceData(
        session_pods=[{"pod": "a"}],
        session_metrics={"x": 1},
        sources_used=["server"],
    )
    secondary = SourceData(
        session_pods=[{"pod": "b"}],
        session_metrics={"y": 2},
        cluster_faults=[{"fault": 1}],
        sources_used=["local"],
    )
    primary.merge_from(secondary)
    assert primary.session_pods == [{"pod": "a"}]
    assert primary.session_metrics == {"x": 1}
    assert primary.cluster_faults == [{"fault": 1}]
    assert primary.sources_used == ["server", "local"]


def test_source_data_merge_inherits_degraded_reason_when_missing():
    primary = SourceData()
    secondary = SourceData(degraded_reason="server timeout")
    primary.merge_from(secondary)
    assert primary.degraded_reason == "server timeout"


# ---------------------------------------------------------------------------
# DegradeRouter happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_happy_path_uses_primary_only():
    clock = _FakeClock()
    primary = _ScriptedSource("server", [_data("server"), _data("server"), _data("server")])
    fallback = _ScriptedSource("local", [_data("local")])
    router = DegradeRouter(primary, fallback, clock=clock)
    for _ in range(3):
        snap = await router.collect(ctx=None)
        assert snap.sources_used == ["server"]
        assert snap.session_summary == {"from": "server"}
        assert snap.degraded_reason is None
    assert primary.calls == 3
    assert fallback.calls == 0


# ---------------------------------------------------------------------------
# DegradeRouter degrade after consecutive failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_degrades_after_three_failures(caplog):
    clock = _FakeClock()
    primary = _ScriptedSource(
        "server",
        [
            SourceUnavailable("first"),
            SourceUnavailable("second"),
            SourceUnavailable("third"),
            SourceUnavailable("fourth"),  # not consumed because we degrade
        ],
    )
    fallback = _ScriptedSource("local", [_data("local"), _data("local"), _data("local"), _data("local")])
    router = DegradeRouter(primary, fallback, fail_threshold=3, recheck_interval_s=30, clock=clock)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            snap = await router.collect(ctx=None)
        # After 3rd failure primary is DEGRADED. Subsequent tick skips primary
        # because clock has not advanced past recheck interval.
        snap_post = await router.collect(ctx=None)

    assert primary.calls == 3, "primary should not be retried inside recheck window"
    assert fallback.calls == 4
    assert router.primary_state is HealthState.DEGRADED
    assert snap_post.degraded_reason and "degraded" in snap_post.degraded_reason
    assert snap_post.sources_used == ["local"]

    transitions = [r for r in caplog.records if "state healthy -> degraded" in r.getMessage()]
    assert len(transitions) == 1, "single WARN on transition only"


# ---------------------------------------------------------------------------
# DegradeRouter recovery after recheck interval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_recovers_after_recheck_window(caplog):
    clock = _FakeClock()
    primary = _ScriptedSource(
        "server",
        [
            SourceUnavailable("a"),
            SourceUnavailable("b"),
            SourceUnavailable("c"),  # degrade after this
            _data("server"),         # recovery probe
        ],
    )
    fallback = _ScriptedSource("local", [_data("local"), _data("local"), _data("local")])
    router = DegradeRouter(primary, fallback, fail_threshold=3, recheck_interval_s=30, clock=clock)

    for _ in range(3):
        await router.collect(ctx=None)
    assert router.primary_state is HealthState.DEGRADED

    # Inside recheck window: primary not probed, fallback served.
    clock.advance(10.0)
    await router.collect(ctx=None)
    assert primary.calls == 3

    # Past recheck window: primary probed, succeeds, state goes HEALTHY.
    clock.advance(25.0)
    with caplog.at_level(logging.WARNING):
        snap = await router.collect(ctx=None)
    assert primary.calls == 4
    assert router.primary_state is HealthState.HEALTHY
    assert snap.sources_used == ["server"]
    assert snap.degraded_reason is None
    transitions = [r for r in caplog.records if "state degraded -> healthy" in r.getMessage()]
    assert len(transitions) == 1


# ---------------------------------------------------------------------------
# Fallback failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_returns_empty_when_both_unavailable():
    clock = _FakeClock()
    primary = _ScriptedSource("server", [SourceUnavailable("p")] * 3)
    fallback = _ScriptedSource("local", [SourceUnavailable("f")] * 3)
    router = DegradeRouter(primary, fallback, fail_threshold=1, recheck_interval_s=0, clock=clock)
    snap = await router.collect(ctx=None)
    assert snap.sources_used == []
    assert snap.degraded_reason and "both sources unavailable" in snap.degraded_reason


# ---------------------------------------------------------------------------
# Unexpected exceptions in primary count as failures but do not crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_treats_unexpected_exception_as_failure(caplog):
    clock = _FakeClock()

    class Boom(Exception):
        pass

    primary = _ScriptedSource("server", [Boom("oops"), Boom("oops"), Boom("oops")])
    fallback = _ScriptedSource("local", [_data("local"), _data("local"), _data("local")])
    router = DegradeRouter(primary, fallback, fail_threshold=3, recheck_interval_s=10, clock=clock)
    with caplog.at_level(logging.ERROR):
        for _ in range(3):
            snap = await router.collect(ctx=None)
            assert snap.sources_used == ["local"]
    assert router.primary_state is HealthState.DEGRADED
    assert any("primary source server raised unexpectedly" in r.getMessage() for r in caplog.records)
