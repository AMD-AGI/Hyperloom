"""Tests for the dispatch_history.jsonl closed-schema writer +
per-event schema audit + telemetry.json rollup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.dynamic_action_history import (
    ABANDONED_FIELDS,
    CRITIC_VERDICT_FIELDS,
    DISPATCHED_FIELDS,
    DispatchHistoryEvent,
    DispatchHistoryRowError,
    INTEGRATE_RESULT_FIELDS,
    SUB_AGENT_DONE_FIELDS,
    SUB_AGENT_TERMINATED_FIELDS,
    TELEMETRY_FIELDS,
    TelemetryRowError,
    append_dispatch_history_row,
    event_field_set,
    write_dynamic_action_telemetry,
)
from inference_optimizer.orchestrator.dynamic_action_proposal import (
    DynamicActionStatus,
)
from inference_optimizer.orchestrator.dynamic_action_resume import (
    ABANDONED_HISTORY_FIELDS as RESUME_ABANDONED_FIELDS,
)
from inference_optimizer.session_paths import (
    dynamic_action_dispatch_history_path,
    dynamic_action_telemetry_path,
)


DYN_ID = "dyn-0-1"


def _read_rows(session_dir: Path, dyn_id: str = DYN_ID) -> list[dict]:
    path = dynamic_action_dispatch_history_path(session_dir, dyn_id)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dispatched_payload() -> dict:
    return {
        "round_index": 3,
        "scope_domains": ["serving_specialist", "kernel_switch_specialist"],
        "side_effects_declared": ["framework_source"],
        "budget_hint": "medium",
        "degraded_dispatch": False,
        "seed_kit_tokens": 1234,
    }


def _sub_agent_done_payload(proposal_count: int = 1) -> dict:
    return {
        "terminal_state": "COMPLETED",
        "reason": "emit_proposal",
        "turns_used": 4,
        "journal_path": "/sd/dyn-0-1/sub_agent_journal.md",
        "proposal_count": proposal_count,
    }


def _critic_verdict_payload(verdict: str = "approve") -> dict:
    return {
        "verdict": verdict,
        "reason_codes": [] if verdict == "approve" else ["x"],
        "applied_rules": ["provenance_literal"],
        "cross_domain_flag": True,
        "mechanical_floor_blocked": False,
    }


def _integrate_result_payload(lifecycle: str = "KEPT") -> dict:
    return {
        "integrate_status": "kept",
        "lifecycle": lifecycle,
        "delta_pct": 3.2,
        "task_id": "task-int-1",
        "patches_applied": ["p.patch"],
        "patches_reverted": [],
    }


def _abandoned_payload() -> dict:
    return {
        "previous_status": "SUB_AGENT_RUNNING",
        "coordinator_session_id": "coord-1",
        "worktree_cleanup_outcome": "success",
        "artifact_missing": False,
    }


# ===========================================================================
# Schema closure
# ===========================================================================
class TestSchemaClosure:

    def test_resume_abandoned_alias_is_canonical_set(self):
        """The resume module re-exports the canonical schema."""
        assert RESUME_ABANDONED_FIELDS is ABANDONED_FIELDS

    @pytest.mark.parametrize("event,field_set", [
        (DispatchHistoryEvent.DISPATCHED, DISPATCHED_FIELDS),
        (DispatchHistoryEvent.SUB_AGENT_DONE, SUB_AGENT_DONE_FIELDS),
        (DispatchHistoryEvent.SUB_AGENT_TERMINATED,
         SUB_AGENT_TERMINATED_FIELDS),
        (DispatchHistoryEvent.CRITIC_VERDICT, CRITIC_VERDICT_FIELDS),
        (DispatchHistoryEvent.INTEGRATE_RESULT, INTEGRATE_RESULT_FIELDS),
        (DispatchHistoryEvent.ABANDONED_ON_RESUME, ABANDONED_FIELDS),
    ])
    def test_event_field_set_lookup(self, event, field_set):
        assert event_field_set(event) == field_set
        assert event_field_set(event.value) == field_set

    @pytest.mark.parametrize("field_set", [
        DISPATCHED_FIELDS, SUB_AGENT_DONE_FIELDS,
        CRITIC_VERDICT_FIELDS, INTEGRATE_RESULT_FIELDS, ABANDONED_FIELDS,
    ])
    def test_every_event_carries_header_fields(self, field_set):
        assert "event" in field_set
        assert "ts" in field_set


# ===========================================================================
# Writer behaviour
# ===========================================================================
class TestWriter:

    def test_writes_one_dispatched_row(self, tmp_path):
        append_dispatch_history_row(
            session_dir=tmp_path,
            dyn_id=DYN_ID,
            event=DispatchHistoryEvent.DISPATCHED,
            payload=_dispatched_payload(),
        )
        rows = _read_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["event"] == "dispatched"
        assert rows[0]["round_index"] == 3
        assert set(rows[0].keys()) == DISPATCHED_FIELDS

    def test_string_event_label_accepted(self, tmp_path):
        append_dispatch_history_row(
            session_dir=tmp_path, dyn_id=DYN_ID,
            event="critic_verdict",
            payload=_critic_verdict_payload(),
        )
        assert _read_rows(tmp_path)[0]["event"] == "critic_verdict"

    def test_appends_multiple_rows(self, tmp_path):
        append_dispatch_history_row(
            session_dir=tmp_path, dyn_id=DYN_ID,
            event=DispatchHistoryEvent.DISPATCHED,
            payload=_dispatched_payload(),
        )
        append_dispatch_history_row(
            session_dir=tmp_path, dyn_id=DYN_ID,
            event=DispatchHistoryEvent.SUB_AGENT_DONE,
            payload=_sub_agent_done_payload(),
        )
        rows = _read_rows(tmp_path)
        assert [r["event"] for r in rows] == ["dispatched", "sub_agent_done"]

    @pytest.mark.parametrize("forbidden_header", ["event", "ts"])
    def test_payload_cannot_carry_header_fields(self, tmp_path, forbidden_header):
        bad = _dispatched_payload()
        bad[forbidden_header] = "x"
        with pytest.raises(DispatchHistoryRowError):
            append_dispatch_history_row(
                session_dir=tmp_path, dyn_id=DYN_ID,
                event=DispatchHistoryEvent.DISPATCHED, payload=bad,
            )

    def test_payload_with_extra_field_rejected(self, tmp_path):
        bad = _dispatched_payload()
        bad["surprise"] = 42
        with pytest.raises(DispatchHistoryRowError):
            append_dispatch_history_row(
                session_dir=tmp_path, dyn_id=DYN_ID,
                event=DispatchHistoryEvent.DISPATCHED, payload=bad,
            )

    def test_payload_missing_field_rejected(self, tmp_path):
        bad = _dispatched_payload()
        del bad["seed_kit_tokens"]
        with pytest.raises(DispatchHistoryRowError):
            append_dispatch_history_row(
                session_dir=tmp_path, dyn_id=DYN_ID,
                event=DispatchHistoryEvent.DISPATCHED, payload=bad,
            )

    def test_unknown_event_label_raises(self, tmp_path):
        with pytest.raises(ValueError):
            append_dispatch_history_row(
                session_dir=tmp_path, dyn_id=DYN_ID,
                event="not_a_real_event", payload={},
            )

    def test_write_failure_is_non_fatal_warning(
        self, tmp_path, monkeypatch, caplog,
    ):
        """OSError on disk → log warning + continue; never raise out."""
        def _boom(self, *a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(Path, "mkdir", _boom)
        append_dispatch_history_row(
            session_dir=tmp_path, dyn_id=DYN_ID,
            event=DispatchHistoryEvent.DISPATCHED,
            payload=_dispatched_payload(),
        )

    def test_round_trip_every_event(self, tmp_path):
        """Write one row per event type; reload and verify the closed
        schema is honoured everywhere."""
        cases = [
            (DispatchHistoryEvent.DISPATCHED, _dispatched_payload(),
             DISPATCHED_FIELDS),
            (DispatchHistoryEvent.SUB_AGENT_DONE,
             _sub_agent_done_payload(), SUB_AGENT_DONE_FIELDS),
            (DispatchHistoryEvent.SUB_AGENT_TERMINATED,
             _sub_agent_done_payload(proposal_count=0),
             SUB_AGENT_TERMINATED_FIELDS),
            (DispatchHistoryEvent.CRITIC_VERDICT,
             _critic_verdict_payload(), CRITIC_VERDICT_FIELDS),
            (DispatchHistoryEvent.INTEGRATE_RESULT,
             _integrate_result_payload(), INTEGRATE_RESULT_FIELDS),
            (DispatchHistoryEvent.ABANDONED_ON_RESUME,
             _abandoned_payload(), ABANDONED_FIELDS),
        ]
        for event, payload, _ in cases:
            append_dispatch_history_row(
                session_dir=tmp_path, dyn_id=DYN_ID,
                event=event, payload=payload,
            )
        rows = _read_rows(tmp_path)
        assert len(rows) == len(cases)
        for row, (event, _payload, field_set) in zip(rows, cases):
            assert row["event"] == event.value
            assert set(row.keys()) == field_set


# ===========================================================================
# Invariant — full lifecycle reconstruction
# ===========================================================================
class TestInvariantLifecycle:
    """A successful KEPT run produces the four lifecycle rows in order
    (DISPATCHED → SUB_AGENT_DONE → CRITIC_VERDICT → INTEGRATE_RESULT)."""

    def test_inv_dispatch_history_complete_lifecycle(self, tmp_path):
        sequence = [
            (DispatchHistoryEvent.DISPATCHED, _dispatched_payload()),
            (DispatchHistoryEvent.SUB_AGENT_DONE,
             _sub_agent_done_payload()),
            (DispatchHistoryEvent.CRITIC_VERDICT,
             _critic_verdict_payload()),
            (DispatchHistoryEvent.INTEGRATE_RESULT,
             _integrate_result_payload()),
        ]
        for event, payload in sequence:
            append_dispatch_history_row(
                session_dir=tmp_path, dyn_id=DYN_ID,
                event=event, payload=payload,
            )
        events = [r["event"] for r in _read_rows(tmp_path)]
        assert events == [e.value for e, _ in sequence]


# ===========================================================================
# telemetry.json per-dyn_id rollup
# ===========================================================================
class TestTelemetry:

    def test_telemetry_field_set_pinned(self):
        assert TELEMETRY_FIELDS == frozenset({
            "dyn_id", "rolled_up_at", "lifecycle",
            "kept", "reverted", "integrate_failed", "critic_rejected",
            "timed_out", "failed", "completed_empty", "abandoned",
            "gain_pct", "round_index",
        })

    def test_telemetry_kept_writes_one_counter_set(self, tmp_path):
        write_dynamic_action_telemetry(
            session_dir=tmp_path, dyn_id=DYN_ID,
            lifecycle=DynamicActionStatus.KEPT,
            gain_pct=3.7, round_index=2,
        )
        path = dynamic_action_telemetry_path(tmp_path, DYN_ID)
        body = json.loads(path.read_text(encoding="utf-8"))
        assert set(body.keys()) == TELEMETRY_FIELDS
        assert body["lifecycle"] == "KEPT"
        assert body["kept"] == 1
        assert body["reverted"] == 0
        assert body["gain_pct"] == 3.7
        assert body["round_index"] == 2

    @pytest.mark.parametrize("lifecycle,counter", [
        (DynamicActionStatus.KEPT, "kept"),
        (DynamicActionStatus.REVERTED, "reverted"),
        (DynamicActionStatus.INTEGRATE_FAILED, "integrate_failed"),
        (DynamicActionStatus.CRITIC_REJECTED, "critic_rejected"),
        (DynamicActionStatus.TIMED_OUT, "timed_out"),
        (DynamicActionStatus.FAILED, "failed"),
        (DynamicActionStatus.COMPLETED_EMPTY, "completed_empty"),
        (DynamicActionStatus.ABANDONED, "abandoned"),
    ])
    def test_telemetry_every_terminal_state_sets_one_counter(
        self, tmp_path, lifecycle, counter,
    ):
        write_dynamic_action_telemetry(
            session_dir=tmp_path, dyn_id=DYN_ID, lifecycle=lifecycle,
        )
        body = json.loads(
            dynamic_action_telemetry_path(tmp_path, DYN_ID)
            .read_text(encoding="utf-8"),
        )
        assert body[counter] == 1
        # Exactly one of the eight counters must be set.
        all_counters = {
            "kept", "reverted", "integrate_failed", "critic_rejected",
            "timed_out", "failed", "completed_empty", "abandoned",
        }
        assert sum(body[c] for c in all_counters) == 1

    def test_telemetry_non_terminal_rejected(self, tmp_path):
        with pytest.raises(TelemetryRowError):
            write_dynamic_action_telemetry(
                session_dir=tmp_path, dyn_id=DYN_ID,
                lifecycle=DynamicActionStatus.SUB_AGENT_RUNNING,
            )

    def test_telemetry_overwrite_idempotent(self, tmp_path):
        """Second write replaces the first — needed when the resume
        sweep re-rolls up a dyn_id that already had a terminal write."""
        write_dynamic_action_telemetry(
            session_dir=tmp_path, dyn_id=DYN_ID,
            lifecycle=DynamicActionStatus.KEPT, gain_pct=5.0,
        )
        write_dynamic_action_telemetry(
            session_dir=tmp_path, dyn_id=DYN_ID,
            lifecycle=DynamicActionStatus.ABANDONED,
        )
        body = json.loads(
            dynamic_action_telemetry_path(tmp_path, DYN_ID)
            .read_text(encoding="utf-8"),
        )
        assert body["lifecycle"] == "ABANDONED"
        assert body["abandoned"] == 1
        assert body["kept"] == 0

    def test_telemetry_string_lifecycle_accepted(self, tmp_path):
        write_dynamic_action_telemetry(
            session_dir=tmp_path, dyn_id=DYN_ID, lifecycle="REVERTED",
        )
        body = json.loads(
            dynamic_action_telemetry_path(tmp_path, DYN_ID)
            .read_text(encoding="utf-8"),
        )
        assert body["lifecycle"] == "REVERTED"


# ===========================================================================
# Telemetry invariant — file present on every terminal dyn_id
# ===========================================================================
class TestInvariantTelemetry:

    def test_inv_telemetry_present_on_terminal_state(self, tmp_path):
        """Every terminal lifecycle write must produce a telemetry
        file on disk."""
        for status in (
            DynamicActionStatus.KEPT,
            DynamicActionStatus.REVERTED,
            DynamicActionStatus.INTEGRATE_FAILED,
            DynamicActionStatus.CRITIC_REJECTED,
            DynamicActionStatus.TIMED_OUT,
            DynamicActionStatus.FAILED,
            DynamicActionStatus.COMPLETED_EMPTY,
            DynamicActionStatus.ABANDONED,
        ):
            dyn = f"dyn-{status.value.lower()}"
            write_dynamic_action_telemetry(
                session_dir=tmp_path, dyn_id=dyn, lifecycle=status,
            )
            assert dynamic_action_telemetry_path(
                tmp_path, dyn,
            ).is_file()
