# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recorder's write trace: what was written, from where, over what."""

from __future__ import annotations

import logging

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import trace as trace_mod
from hyperloom.inference_optimizer.breakdown.recorder.instrument import (
    record_measurement,
    record_operation,
)
from hyperloom.inference_optimizer.breakdown.recorder.recorder import Recorder


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

    record_measurement(tmp_path, measurement_id="m-1", name="final_throughput", value=5081.01)

    assert not trace_mod.trace_enabled()
    assert caplog.records == []


def test_a_write_says_what_it_wrote_and_who_asked(tmp_path, traced):
    """The two questions a doubted number raises, answered on one line."""
    record_measurement(
        tmp_path,
        measurement_id="m-1",
        name="final_throughput",
        value=5081.0100767,
        unit="tok/s",
    )

    line = "\n".join(record.getMessage() for record in traced.records)

    assert "section=measurements" in line
    assert "measurement_id=m-1" in line
    assert "outcome=created" in line
    # Both sides of the SDK: the helper that built the payload, and the code
    # that decided to record something.
    assert "via=instrument.py:" in line
    assert "record_measurement" in line
    assert f"from={__file__.rsplit('/', 1)[-1]}:" in line


def test_an_overwritten_reading_is_named_with_both_values(tmp_path, traced):
    """The gemma overwrite, as it would have looked while it was happening.

    A fragment id is stable per entity, so a second write of the same id merges
    into the first. That is what lets a later re-measure land on the readings an
    earlier decision was made on, and it is invisible in the archive that
    results. Here it is a line saying so, at the moment it happens.
    """
    for value in (5081.0100767, 5100.763142143991):
        record_measurement(tmp_path, measurement_id="m-final", name="final_throughput", value=value)

    first, second = (record.getMessage() for record in traced.records)

    assert "outcome=created" in first
    assert "outcome=replaced" in second
    assert "changed=value:5081.0100767->5100.763142143991" in second


def test_a_rewrite_that_changes_nothing_says_nothing_changed(tmp_path, traced):
    """Recording the same fact twice is normal, and worth telling apart."""
    for _ in range(2):
        record_measurement(tmp_path, measurement_id="m-1", name="final_throughput", value=5081.01)

    second = list(traced.records)[-1].getMessage()

    assert "outcome=replaced" in second
    assert "changed=none" in second


def test_a_nested_field_is_named_rather_than_dumped(tmp_path, traced):
    """A line long enough to hold two nested payloads is one nobody reads."""
    for status in ("succeeded", "needs_review"):
        record_operation(
            tmp_path,
            operation_id="op-1",
            kind="kernel_optimization",
            status=status,
            outputs={"decision": status, "nested": {"deep": [1, 2, 3]}},
        )

    second = list(traced.records)[-1].getMessage()

    assert "status:succeeded->needs_review" in second
    assert "outputs:changed" in second
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
        recorder.record_item("measurements", {"measurement_id": "m-1"})

    line = list(traced.records)[-1].getMessage()

    assert "outcome=failed" in line
    assert "error=OSError:no space left on device" in line


def test_a_trace_that_breaks_does_not_break_the_write(tmp_path, traced, monkeypatch):
    """A broken trace must never be why a fact went unrecorded."""

    def explode(*_args, **_kwargs):
        raise RuntimeError("trace is broken")

    monkeypatch.setattr(trace_mod, "_entity", explode)
    recorder = Recorder(tmp_path, producer="kernel_agent")

    target = recorder.record_item("measurements", {"measurement_id": "m-1"}, key="m-1")

    assert target.exists()
    assert any("trace failed" in record.getMessage() for record in traced.records)


def test_a_credential_in_a_recorded_value_is_masked(tmp_path, traced):
    """Producers record diagnostic text, and text is where credentials hide.

    The values a merging write names are the ones worth reading in full, which
    is exactly what makes this line the widest copy of whatever a producer put
    in the payload.
    """
    for detail in ("Authorization: Bearer tok-aaaaaaaaaaaa", "retry failed"):
        record_operation(tmp_path, operation_id="op-1", kind="integrate", error=detail)

    line = list(traced.records)[-1].getMessage()

    assert "tok-aaaaaaaaaaaa" not in line
    assert "[REDACTED]" in line


def test_a_credential_in_an_entity_id_is_masked(tmp_path, traced):
    """An id can be built from an error excerpt, and it names the fragment file."""
    record_operation(tmp_path, operation_id="op-ak-liveseecret123", kind="integrate")

    line = list(traced.records)[-1].getMessage()

    assert "liveseecret123" not in line
    assert "[REDACTED]" in line


def test_the_trace_level_stays_below_debug():
    """A per-write firehose cannot share a level with output read for anything else."""
    assert trace_mod.TRACE < logging.DEBUG
    assert logging.getLevelName(trace_mod.TRACE) == "TRACE"
