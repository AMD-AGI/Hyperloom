# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for hyperloom.agents.robustness.signals.stall."""

from __future__ import annotations

from hyperloom.common.coerce import to_unix
from hyperloom.agents.robustness.role.prompt_inputs import InboxItem, ReactorContext
from hyperloom.agents.robustness.signals.stall import (
    StallConfig,
    _collect_last_seen,
    evaluate_stall_signals,
)
from hyperloom.agents.robustness.sources.base import SourceData
from hyperloom.agents.robustness.signals.symptom import SymptomSeverity


def _item(from_agent: str, ts=None) -> InboxItem:
    payload = {"ts": ts} if ts is not None else {}
    return InboxItem(seq=1, msg_id="m", from_agent=from_agent, topic="heartbeat", payload=payload)


def test_coerce_unix_variants() -> None:
    assert to_unix(None) is None
    assert to_unix(5) == 5.0
    assert to_unix("2026-01-01T00:00:00Z") > 0
    assert to_unix("123.5") == 123.5
    assert to_unix("not-a-time") is None


def test_collect_last_seen_branches() -> None:
    inbox = [
        _item("user", ts=999.0),  # untracked -> skip
        _item("kernel_agent", ts=50.0),  # tracked -> set
        _item("kernel_agent"),  # no ts -> skip
    ]
    events = [
        {"agent": "critic"},  # no ts -> continue
        {"agent": "orchestration", "ts": 200.0},
        {"agent": "user", "ts": 5.0},  # untracked
    ]
    last = _collect_last_seen(inbox, events)
    assert last == {"kernel_agent": 50.0, "orchestration": 200.0}


def test_evaluate_stall_emits_high() -> None:
    ctx = ReactorContext(inbox=[_item("orchestration", ts=100.0)], now_unix=10_000.0)
    data = SourceData(coordinator_events=[])
    symptoms = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=300.0, severity_high_after_s=900.0))
    assert len(symptoms) == 1
    assert symptoms[0].name == "agent_stall"
    assert symptoms[0].severity == SymptomSeverity.HIGH


def test_evaluate_stall_no_activity_no_symptom() -> None:
    # No ground truth for any agent -> no symptom.
    ctx = ReactorContext(inbox=[], now_unix=10_000.0)
    data = SourceData(coordinator_events=[])
    assert evaluate_stall_signals(ctx, data) == []


def _progress(agent: str, *, unix: float, task: str) -> dict:
    """A ``local_task_progress`` snapshot holding one agent's heartbeat."""
    return {"running": 1, "by_agent": {agent: {"last_progress_unix": unix, "task": task}}}


def test_work_still_reporting_units_withholds_the_accusation() -> None:
    """A phase whose work is one long deterministic task has no turn to emit."""
    ctx = ReactorContext(inbox=[_item("orchestration", ts=9_800.0)], now_unix=10_000.0)
    data = SourceData(local_task_progress=_progress("orchestration", unix=9_950.0, task="roofline"))
    out = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=100.0))
    assert [s.severity for s in out] == [SymptomSeverity.LOW]
    assert out[0].evidence["accusation_withheld"] is True
    assert out[0].evidence["in_flight_work"] == "roofline"


def test_a_withheld_accusation_still_leaves_a_trace() -> None:
    """``return []`` made the near-miss unobservable; the operator needs to see it."""
    ctx = ReactorContext(inbox=[_item("orchestration", ts=9_800.0)], now_unix=10_000.0)
    data = SourceData(local_task_progress=_progress("orchestration", unix=9_950.0, task="explore"))
    out = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=100.0))
    assert out[0].name == "agent_stall"
    assert out[0].subject == {"agent": "orchestration"}
    assert "explore" in out[0].summary and "withheld" in out[0].summary


def test_busy_work_of_one_agent_does_not_silence_another() -> None:
    """One busy task used to vouch for every agent at once, with no attribution."""
    ctx = ReactorContext(
        inbox=[_item("orchestration", ts=9_800.0), _item("critic", ts=9_800.0)],
        now_unix=10_000.0,
    )
    data = SourceData(local_task_progress=_progress("orchestration", unix=9_950.0, task="explore"))
    by_agent = {
        s.subject["agent"]: s.severity
        for s in evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=100.0))
    }
    assert by_agent["orchestration"] is SymptomSeverity.LOW
    assert by_agent["critic"] is SymptomSeverity.MEDIUM


def test_an_hour_long_warmup_that_keeps_reporting_is_never_accused() -> None:
    """A measured warmup runs 3941s; a ceiling would alert on every healthy one."""
    ctx = ReactorContext(inbox=[_item("orchestration", ts=10_000.0)], now_unix=13_941.0)
    data = SourceData(local_task_progress=_progress("orchestration", unix=13_900.0, task="warmup"))
    out = evaluate_stall_signals(
        ctx,
        data,
        config=StallConfig(stall_timeout_s=300.0, severity_high_after_s=900.0),
    )
    assert out[0].evidence["accusation_withheld"] is True
    assert out[0].severity is SymptomSeverity.MEDIUM


def test_a_long_wait_raises_the_withheld_note_without_turning_it_into_an_accusation() -> None:
    """Operators need the long ones to stand out; they are still not stalls."""
    ctx = ReactorContext(inbox=[_item("orchestration", ts=10_000.0)], now_unix=13_941.0)
    data = SourceData(local_task_progress=_progress("orchestration", unix=13_900.0, task="warmup"))
    cfg = StallConfig(stall_timeout_s=300.0, severity_high_after_s=900.0)
    long_wait = evaluate_stall_signals(ctx, data, config=cfg)[0]
    short_ctx = ReactorContext(inbox=[_item("orchestration", ts=13_500.0)], now_unix=13_941.0)
    short_wait = evaluate_stall_signals(short_ctx, data, config=cfg)[0]
    assert short_wait.severity is SymptomSeverity.LOW
    assert long_wait.severity is SymptomSeverity.MEDIUM
    assert short_wait.severity is not SymptomSeverity.HIGH


def test_the_moment_the_work_stops_reporting_the_next_tick_accuses() -> None:
    """Freshness, not the clock, is what bounds the suppression."""
    ctx = ReactorContext(inbox=[_item("orchestration", ts=10_000.0)], now_unix=13_941.0)
    stale = SourceData(local_task_progress=_progress("orchestration", unix=13_600.0, task="warmup"))
    out = evaluate_stall_signals(
        ctx,
        stale,
        config=StallConfig(stall_timeout_s=300.0, severity_high_after_s=900.0),
    )
    assert [s.severity for s in out] == [SymptomSeverity.HIGH]
    assert "accusation_withheld" not in out[0].evidence


def test_work_that_has_gone_quiet_too_lets_the_stall_through() -> None:
    """Silent agents plus silent work is the case the signal exists for."""
    ctx = ReactorContext(inbox=[_item("orchestration", ts=100.0)], now_unix=10_000.0)
    data = SourceData(local_task_progress=_progress("orchestration", unix=1_000.0, task="explore"))
    out = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=300.0))
    assert [s.name for s in out] == ["agent_stall"]
    assert out[0].evidence["in_flight_work"] == "explore"
    assert out[0].evidence["in_flight_work_idle_seconds"] == 9_000


def test_running_work_that_never_reported_is_no_evidence_either_way() -> None:
    """Absent a heartbeat the signal must fall back to agent silence, not trust."""
    ctx = ReactorContext(inbox=[_item("orchestration", ts=100.0)], now_unix=10_000.0)
    data = SourceData(local_task_progress={"running": 1})
    out = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=300.0))
    assert [s.name for s in out] == ["agent_stall"]
    assert "in_flight_work_idle_seconds" not in out[0].evidence
