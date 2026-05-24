"""F3-3 / N31 — auto-enqueue final roofline on CLOSE entry.

Exercises :meth:`Coordinator._maybe_enqueue_final_roofline_n31` and
its helper :meth:`Coordinator._snapshot_age_seconds` directly,
bypassing the full Coordinator constructor (which needs a live
SQLite, agent registry, knowledge plane, etc.). The helper only
touches three Coordinator surfaces — ``shared_state``, ``tasks``,
``_record_close_step`` — so a lightweight stub class is enough.

Reference: ``plan_roofline_framework/F3_policygate_advisory.MD`` §F3-3.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState


class _StubTaskRegistry:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.next_was_existing = False

    async def create_or_return_existing(self, **kwargs: Any):
        self.calls.append(kwargs)
        from types import SimpleNamespace
        return SimpleNamespace(task_id="t-stub"), self.next_was_existing


class _N31Coordinator(Coordinator):
    """Bypass __init__ — we only need the two helpers under test."""

    def __init__(self, state: SharedState) -> None:
        # Skip the heavy Coordinator.__init__: we only invoke the two
        # methods under test, both of which read self.shared_state and
        # self.tasks and call self._record_close_step.
        self.shared_state = state
        self.tasks = _StubTaskRegistry()
        self.close_steps: list[dict] = []

    async def _record_close_step(self, name: str, **kwargs: Any) -> None:
        self.close_steps.append({"name": name, **kwargs})


def _iso_offset(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)) \
        .isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# _snapshot_age_seconds — pure helper
# ---------------------------------------------------------------------------


def test_snapshot_age_returns_none_for_missing_ts():
    assert Coordinator._snapshot_age_seconds(None) is None
    assert Coordinator._snapshot_age_seconds("") is None
    assert Coordinator._snapshot_age_seconds("not-a-date") is None


def test_snapshot_age_parses_iso_with_tz():
    age = Coordinator._snapshot_age_seconds(_iso_offset(120.0))
    assert age is not None
    assert 100.0 <= age <= 200.0


def test_snapshot_age_handles_naive_iso_as_utc():
    """``_now_iso()`` always emits an offset, but tests / legacy
    snapshots may have written naive strings — treat them as UTC."""
    naive = (datetime.now(timezone.utc) - timedelta(seconds=60)) \
        .replace(tzinfo=None) \
        .isoformat(timespec="microseconds")
    age = Coordinator._snapshot_age_seconds(naive)
    assert age is not None
    assert 30.0 <= age <= 120.0


# ---------------------------------------------------------------------------
# _maybe_enqueue_final_roofline_n31 — gate behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n31_skipped_when_composite_off():
    s = SharedState()
    s.use_roofline_composite = False
    coord = _N31Coordinator(s)
    out = await coord._maybe_enqueue_final_roofline_n31()
    assert out is None
    assert coord.tasks.calls == []
    assert coord.close_steps == []


@pytest.mark.asyncio
async def test_n31_enqueues_when_no_snapshot():
    s = SharedState()
    s.use_roofline_composite = True
    s.last_trace_analyze = {}
    coord = _N31Coordinator(s)
    out = await coord._maybe_enqueue_final_roofline_n31()
    assert out is not None
    assert len(coord.tasks.calls) == 1
    call = coord.tasks.calls[0]
    assert call["kind"] == "roofline"
    assert call["idempotency_key"] == "internal-roofline-n31_close"
    assert call["params"]["reason"] == "n31_no_snapshot"
    assert call["params"]["source"] == "coordinator_internal"
    assert coord.close_steps[-1]["name"] == "n31_final_roofline"
    assert coord.close_steps[-1]["status"] == "enqueued"


@pytest.mark.asyncio
async def test_n31_skipped_when_snapshot_fresh():
    s = SharedState()
    s.use_roofline_composite = True
    s.cumulative_gain_validated = 12.0
    s.last_trace_analyze = {
        "analysis_md_text": "# body",
        "ts": _iso_offset(60.0),               # 1 min — fresh
        "roofline_baseline_gain_at_snapshot": 11.5,   # drift < 3%
    }
    coord = _N31Coordinator(s)
    out = await coord._maybe_enqueue_final_roofline_n31()
    assert out is None
    assert coord.tasks.calls == []


@pytest.mark.asyncio
async def test_n31_enqueues_on_stale_age():
    s = SharedState()
    s.use_roofline_composite = True
    s.cumulative_gain_validated = 12.0
    s.last_trace_analyze = {
        "analysis_md_text": "# body",
        "ts": _iso_offset(900.0),              # 15 min ago — > 5 min threshold
        "roofline_baseline_gain_at_snapshot": 12.0,   # drift = 0
    }
    coord = _N31Coordinator(s)
    await coord._maybe_enqueue_final_roofline_n31()
    assert len(coord.tasks.calls) == 1
    assert coord.tasks.calls[0]["params"]["reason"] == "n31_stale_age"


@pytest.mark.asyncio
async def test_n31_enqueues_on_gain_drift():
    s = SharedState()
    s.use_roofline_composite = True
    s.cumulative_gain_validated = 18.0
    s.last_trace_analyze = {
        "analysis_md_text": "# body",
        "ts": _iso_offset(60.0),               # fresh in time
        "roofline_baseline_gain_at_snapshot": 11.0,   # drift = 7% > 3%
    }
    coord = _N31Coordinator(s)
    await coord._maybe_enqueue_final_roofline_n31()
    assert len(coord.tasks.calls) == 1
    assert coord.tasks.calls[0]["params"]["reason"] == "n31_gain_drift"


@pytest.mark.asyncio
async def test_n31_idempotent_on_re_entry():
    s = SharedState()
    s.use_roofline_composite = True
    s.last_trace_analyze = {}
    coord = _N31Coordinator(s)
    coord.tasks.next_was_existing = True
    out = await coord._maybe_enqueue_final_roofline_n31()
    assert out is not None
    # Existing task returned — close_step still recorded so the
    # sequencer audit trail is consistent.
    assert coord.close_steps[-1]["name"] == "n31_final_roofline"
