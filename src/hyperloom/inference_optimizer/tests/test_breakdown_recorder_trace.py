# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recorder's write trace: what was written, from where, over what."""

from __future__ import annotations

import logging

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import trace as trace_mod
from hyperloom.inference_optimizer.breakdown.recorder.instrument import record_singleton_section
from hyperloom.inference_optimizer.breakdown.recorder.recorder import Recorder
from hyperloom.orchestrator.kernel._recorder_trace import trace_recording_skipped


def _record(session_dir, payload=None):
    """Record through the SDK, so the trace sees both sides of the call."""
    record_singleton_section(
        session_dir,
        "session",
        {"session_id": "s-1", "phase": "CLOSE"} if payload is None else payload,
        producer="coordinator",
    )


@pytest.fixture
def traced(caplog):
    """Turn the write trace on for one test and capture what it emits."""
    trace_mod.enable_trace(True)
    caplog.set_level(trace_mod.TRACE, logger=trace_mod.log.name)
    try:
        yield caplog
    finally:
        trace_mod.enable_trace(False)


def test_the_trace_is_silent_until_it_is_asked_for(tmp_path, caplog):
    """A recorder is called from everywhere, so this has to default to off."""
    caplog.set_level(trace_mod.TRACE, logger=trace_mod.log.name)
    trace_mod.enable_trace(False)

    _record(tmp_path)

    assert not trace_mod.trace_enabled()
    assert caplog.records == []


def test_a_write_says_what_it_wrote_and_who_asked(tmp_path, traced):
    """The two questions a doubted number raises, answered on one line."""
    _record(tmp_path, {"session_id": "s-1", "tick_count": 12})

    line = "\n".join(record.getMessage() for record in traced.records)

    assert "section=session" in line
    assert "outcome=created" in line
    # Both sides of the SDK: the helper that built the payload, and the code
    # that decided to record something.
    assert "via=instrument.py:" in line
    assert "record_singleton_section" in line
    assert f"from={__file__.rsplit('/', 1)[-1]}:" in line


def test_an_overwritten_reading_is_named_with_both_values(tmp_path, traced):
    """The gemma overwrite, as it would have looked while it was happening.

    A fragment id is stable per entity, so a second write of the same id merges
    into the first. That is what lets a later re-measure land on the readings an
    earlier decision was made on, and it is invisible in the archive that
    results. Here it is a line saying so, at the moment it happens.
    """
    recorder = Recorder(tmp_path, producer="kernel_agent")
    for value in (5081.0100767, 5100.763142143991):
        recorder.record_upsert_item(
            "kernel_rebench_attempt",
            {"kernel_id": "k001", "tput": value},
            key="k001",
        )

    first, second = (record.getMessage() for record in traced.records)

    assert "outcome=created" in first
    assert "outcome=replaced" in second
    assert "changed=tput:5081.0100767->5100.763142143991" in second


def test_a_rewrite_that_changes_nothing_says_nothing_changed(tmp_path, traced):
    """Recording the same fact twice is normal, and worth telling apart."""
    recorder = Recorder(tmp_path, producer="kernel_agent")
    for _ in range(2):
        recorder.record_upsert_item(
            "kernel_rebench_attempt",
            {"kernel_id": "k001", "tput": 5081.01},
            key="k001",
        )

    second = list(traced.records)[-1].getMessage()

    assert "outcome=replaced" in second
    assert "changed=none" in second


def test_a_nested_field_is_named_rather_than_dumped(tmp_path, traced):
    """A line long enough to hold two nested payloads is one nobody reads."""
    recorder = Recorder(tmp_path, producer="kernel_agent")
    for status in ("succeeded", "needs_review"):
        recorder.record_upsert_item(
            "kernel_lane_run",
            {
                "kernel_id": "k001",
                "status": status,
                "evidence": {"decision": status, "nested": {"deep": [1, 2, 3]}},
            },
            key="k001",
        )

    second = list(traced.records)[-1].getMessage()

    assert "status:succeeded->needs_review" in second
    assert "evidence:changed" in second
    assert "deep" not in second


def test_a_failed_write_is_traced_and_still_raises(tmp_path, traced, monkeypatch):
    """A write that never landed is the most important one to hear about."""

    def explode(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.breakdown.recorder.recorder.atomic_write_text",
        explode,
    )
    recorder = Recorder(tmp_path, producer="kernel_agent")

    with pytest.raises(OSError):
        recorder.record_item("kernel_lane_run", {"kernel_id": "k001"})

    line = list(traced.records)[-1].getMessage()

    assert "outcome=failed" in line
    assert "error=OSError:no space left on device" in line


def test_a_trace_that_breaks_does_not_break_the_write(tmp_path, traced, monkeypatch):
    """A broken trace must never be why a fact went unrecorded."""

    def explode(*_args, **_kwargs):
        raise RuntimeError("trace is broken")

    monkeypatch.setattr(trace_mod, "_entity", explode)
    recorder = Recorder(tmp_path, producer="kernel_agent")

    target = recorder.record_item("kernel_lane_run", {"kernel_id": "k001"}, key="k001")

    assert target.exists()
    assert any("trace failed" in record.getMessage() for record in traced.records)


def test_a_credential_in_a_recorded_value_is_masked(tmp_path, traced):
    """Producers record diagnostic text, and text is where credentials hide.

    The values a merging write names are the ones worth reading in full, which
    is exactly what makes this line the widest copy of whatever a producer put
    in the payload.
    """
    recorder = Recorder(tmp_path, producer="kernel_agent")
    for detail in ("Authorization: Bearer tok-aaaaaaaaaaaa", "retry failed"):
        recorder.record_upsert_item(
            "kernel_lane_run",
            {"kernel_id": "k001", "error": detail},
            key="k001",
        )

    line = list(traced.records)[-1].getMessage()

    assert "tok-aaaaaaaaaaaa" not in line
    assert "[REDACTED]" in line


def test_a_credential_in_an_entity_id_is_masked(tmp_path, traced):
    """An id can be built from an error excerpt, and the trace names the id."""
    recorder = Recorder(tmp_path, producer="kernel_agent")
    recorder.record_item("kernel_lane_run", {"kernel_id": "ak-liveseecret123"})

    line = list(traced.records)[-1].getMessage()

    assert "liveseecret123" not in line
    assert "[REDACTED]" in line


def test_the_trace_level_stays_below_debug():
    """A per-write firehose cannot share a level with output read for anything else."""
    assert trace_mod.TRACE < logging.DEBUG
    assert logging.getLevelName(trace_mod.TRACE) == "TRACE"


def test_turning_the_trace_off_is_not_the_same_as_leaving_it_unset(tmp_path, caplog):
    """Clearing the level hands the decision to whatever the root happens to be.

    A root logger at NOTSET enables everything, so a caller that explicitly
    asked for silence kept getting a firehose.
    """
    caplog.set_level(trace_mod.TRACE, logger=trace_mod.log.name)
    trace_mod.enable_trace(True)
    trace_mod.enable_trace(False)
    root = logging.getLogger()
    restore = root.level
    root.setLevel(logging.NOTSET)
    try:
        _record(tmp_path)
    finally:
        root.setLevel(restore)

    assert not trace_mod.trace_enabled()
    assert caplog.records == []


def test_a_write_that_only_adds_fields_still_admits_what_it_left_out():
    """Both halves of the line are capped, so both have to be counted."""
    previous = {"session_id": "s-1"}
    merged = {"session_id": "s-1", **{f"f{index}": index for index in range(20)}}

    summary = trace_mod._changed(previous, merged)

    assert "(+" in summary and "more)" in summary


def test_a_record_that_was_never_attempted_says_why(tmp_path, traced):
    """The gap this fills: nothing written, and nothing said about it either."""
    _record(None)
    _record(tmp_path, {})

    lines = [record.getMessage() for record in traced.records]

    assert any("outcome=skipped" in line and "no session_dir" in line for line in lines)
    assert any("outcome=skipped" in line and "empty payload" in line for line in lines)
    assert all("via=instrument.py:" in line for line in lines if "skipped" in line)


def test_a_swallowed_writer_failure_is_traced_and_still_swallowed(tmp_path, traced, monkeypatch):
    """A producer keeps running; the record it lost should not go unmentioned."""

    def explode(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.breakdown.recorder.recorder.atomic_write_text",
        explode,
    )

    _record(tmp_path)

    line = list(traced.records)[-1].getMessage()

    assert "outcome=skipped" in line
    assert "reason=writer raised" in line
    assert "error=OSError:no space left on device" in line


def test_a_credential_in_a_skipped_record_is_masked(tmp_path, traced, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("Authorization: Bearer tok-aaaaaaaaaaaa")

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.breakdown.recorder.recorder.atomic_write_text",
        explode,
    )

    _record(tmp_path)

    line = list(traced.records)[-1].getMessage()

    assert "tok-aaaaaaaaaaaa" not in line
    assert "[REDACTED]" in line


def test_a_call_that_never_reached_the_recorder_says_so(traced):
    """The recorder can only trace calls that arrive.

    Producers call it from inside ``try`` blocks wider than the call itself, so
    a failure in the import, the arguments, or a pre-condition means none of
    the recorder's own guards ever rule and the loss is silent.
    """
    trace_recording_skipped(
        "kernel_invocations",
        reason="caller raised before the recorder",
        entity="k001",
        error=RuntimeError("boom"),
    )

    line = list(traced.records)[-1].getMessage()

    assert "section=kernel_invocations" in line
    assert "outcome=skipped" in line
    assert "id=k001" in line
    assert "error=RuntimeError:boom" in line
