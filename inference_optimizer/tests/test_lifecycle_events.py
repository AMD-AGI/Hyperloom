# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""#266 method-1 lifecycle-event infrastructure.

Covers the operator-facing phase/step boundary log:

* ``phase_state.lifecycle_label`` resolves the human-friendly names used
  in issue #266 (TraceLens / GEAK / Integrate / Report), falls back to the
  phase-label table, then to the verbatim name.
* ``phase_state.make_lifecycle_event`` produces a canonical row (seq / ts /
  phase upper-cased / step / label default / status upper-cased / artifact
  filtering / duration rounding).
* ``SharedState.record_lifecycle_event`` appends, defaults the phase to the
  current coordinator phase, keeps ``seq`` monotonic across the cap, and
  enforces ``_LIFECYCLE_CAP``.
* ``lifecycle`` round-trips through ``save`` / ``load_or_init``.
* ``lifecycle`` is a Coordinator-only (``CORE_STATE_FIELDS``) field so an
  LLM ``update_state`` cannot forge events.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.phase_state import (
    LIFECYCLE_STATUSES,
    PHASE_KERNEL,
    lifecycle_label,
    make_lifecycle_event,
)
from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
from inference_optimizer.orchestrator.shared_state import (
    _LIFECYCLE_CAP,
    SharedState,
)


# ===========================================================================
# pure helpers
# ===========================================================================
def test_lifecycle_label_resolves_human_labels():
    # The five #266 pipeline names map to the internal step/handler names.
    assert lifecycle_label("trace_analyze") == "TraceLens"
    assert lifecycle_label("roofline") == "TraceLens"
    assert lifecycle_label("run_optimization") == "GEAK"
    assert lifecycle_label("integrate") == "Integrate"
    assert lifecycle_label("apply_patch") == "Integrate"
    assert lifecycle_label("report") == "Report"


def test_lifecycle_label_falls_back_to_phase_then_verbatim():
    # A bare phase name resolves to its human label ...
    assert lifecycle_label("KERNEL") == "Kernel optimization"
    assert lifecycle_label("kernel") == "Kernel optimization"
    # ... and an unmapped name is returned verbatim (never empty).
    assert lifecycle_label("some_new_step") == "some_new_step"
    assert lifecycle_label("") == ""


def test_make_lifecycle_event_shape():
    event = make_lifecycle_event(
        step="trace_analyze",
        status="start",  # lower-cased on purpose
        phase="kernel",  # lower-cased on purpose
        label=None,      # default from step
        artifacts={"out_dir": "/tmp/run", "empty": "", "none": None},
        detail="  starting  ",
        duration_s=1.23456,
        seq=3,
        ts="2026-06-09T00:00:00+00:00",
    )
    assert event["seq"] == 3
    assert event["ts"] == "2026-06-09T00:00:00+00:00"
    assert event["phase"] == "KERNEL"          # upper-cased
    assert event["step"] == "trace_analyze"
    assert event["label"] == "TraceLens"       # defaulted from step
    assert event["status"] == "START"          # upper-cased
    assert event["detail"] == "starting"       # stripped
    # Empty / None artifact values are dropped; survivors are str-coerced.
    assert event["artifacts"] == {"out_dir": "/tmp/run"}
    assert event["duration_s"] == 1.235        # rounded to 3dp


def test_make_lifecycle_event_omits_duration_when_none():
    event = make_lifecycle_event(
        step="report", status="START", phase="CLOSE", label=None,
        artifacts=None, detail="", duration_s=None, seq=0,
        ts="2026-06-09T00:00:00+00:00",
    )
    assert "duration_s" not in event
    assert event["artifacts"] == {}
    assert event["label"] == "Report"


def test_lifecycle_statuses_enum():
    assert LIFECYCLE_STATUSES == frozenset({"START", "END", "ERROR"})


# ===========================================================================
# SharedState writer
# ===========================================================================
def test_record_lifecycle_event_appends_and_defaults_phase():
    s = SharedState(session_id="abc")
    s.phase = PHASE_KERNEL
    row = s.record_lifecycle_event(
        step="run_optimization",
        status="START",
        artifacts={"workspace": "/tmp/ws"},
        detail="kernel tg001",
    )
    assert len(s.lifecycle) == 1
    assert row is s.lifecycle[0]
    assert row["phase"] == "KERNEL"            # defaulted from current phase
    assert row["label"] == "GEAK"              # defaulted from step
    assert row["status"] == "START"
    assert row["artifacts"] == {"workspace": "/tmp/ws"}
    assert row["seq"] == 0


def test_record_lifecycle_event_explicit_phase_and_label_override():
    s = SharedState(session_id="abc")
    s.phase = PHASE_KERNEL
    row = s.record_lifecycle_event(
        step="custom", status="END", phase="EXPLORE",
        label="Custom Label", duration_s=2.0,
    )
    assert row["phase"] == "EXPLORE"
    assert row["label"] == "Custom Label"
    assert row["duration_s"] == 2.0


def test_record_lifecycle_event_monotonic_seq_and_cap():
    s = SharedState(session_id="abc")
    total = _LIFECYCLE_CAP + 25
    for i in range(total):
        s.record_lifecycle_event(step="trace_analyze", status="END", detail=f"#{i}")
    # Cap is enforced ...
    assert len(s.lifecycle) == _LIFECYCLE_CAP
    # ... but seq stays monotonic and reflects the true emission order even
    # after the oldest rows are trimmed.
    seqs = [e["seq"] for e in s.lifecycle]
    assert seqs == sorted(seqs)
    assert seqs[-1] == total - 1
    # The very first surviving row is the (total - cap)-th emitted event.
    assert seqs[0] == total - _LIFECYCLE_CAP


def test_lifecycle_persists_round_trip(tmp_path):
    s = SharedState(session_id="abc")
    s.phase = PHASE_KERNEL
    s.record_lifecycle_event(
        step="trace_analyze", status="END",
        artifacts={"candidates": "/tmp/kc.json"}, duration_s=42.0,
    )
    s.save(tmp_path)

    s2 = SharedState.load_or_init(tmp_path)
    assert len(s2.lifecycle) == 1
    ev = s2.lifecycle[0]
    assert ev["step"] == "trace_analyze"
    assert ev["label"] == "TraceLens"
    assert ev["artifacts"] == {"candidates": "/tmp/kc.json"}
    assert ev["duration_s"] == 42.0


def test_lifecycle_is_core_state_field():
    # Defense-in-depth: an LLM update_state intent must not be able to
    # forge lifecycle events (would let it fake "phase finished" lines).
    assert "lifecycle" in CORE_STATE_FIELDS
