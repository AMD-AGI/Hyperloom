# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recorder's write trace: what was written, from where, over what."""

from __future__ import annotations

import logging

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import trace as trace_mod
from hyperloom.inference_optimizer.breakdown.recorder.instrument import (
    record_gemm_tuning_operation,
    record_measurement,
    record_operation,
)
from hyperloom.inference_optimizer.breakdown.recorder.recorder import Recorder
from hyperloom.orchestrator.kernel._recorder_trace import trace_recording_skipped
from hyperloom.orchestrator.state.shared_state import SharedState


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
        record_measurement(tmp_path, measurement_id="m-1", name="throughput", value=1.0)
    finally:
        root.setLevel(restore)

    assert not trace_mod.trace_enabled()
    assert caplog.records == []


def test_a_write_that_only_adds_fields_still_admits_what_it_left_out():
    """Both halves of the line are capped, so both have to be counted."""
    previous = {"operation_id": "op-1"}
    merged = {"operation_id": "op-1", **{f"f{index}": index for index in range(20)}}

    summary = trace_mod._changed(previous, merged)

    assert "(+" in summary and "more)" in summary


def test_a_record_that_was_never_attempted_says_why(tmp_path, traced):
    """The gap this fills: nothing written, and nothing said about it either."""
    record_operation(None, operation_id="op-1", kind="integrate_patch")
    record_measurement(tmp_path, name="throughput", value=1.0)

    lines = [record.getMessage() for record in traced.records]

    assert any("outcome=skipped" in line and "no session_dir" in line for line in lines)
    assert any("outcome=skipped" in line and "no measurement_id" in line for line in lines)
    assert all("via=instrument.py:" in line for line in lines if "skipped" in line)


def test_a_swallowed_writer_failure_is_traced_and_still_swallowed(tmp_path, traced, monkeypatch):
    """A producer keeps running; the record it lost should not go unmentioned."""

    def explode(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.breakdown.recorder.recorder.atomic_write_text",
        explode,
    )

    record_operation(tmp_path, operation_id="op-1", kind="integrate_patch")

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

    record_operation(tmp_path, operation_id="op-1", kind="integrate_patch")

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
        "kernel_e2e",
        reason="caller raised before the recorder",
        entity="k001",
        error=RuntimeError("boom"),
    )

    line = list(traced.records)[-1].getMessage()

    assert "section=kernel_e2e" in line
    assert "outcome=skipped" in line
    assert "id=k001" in line
    assert "error=RuntimeError:boom" in line


def test_an_integrate_the_orchestrator_could_not_record_is_named(traced):
    """The path a lost adoption actually takes out of the run.

    ``record_kernel_e2e`` is called under a condition checked before the
    recorder is reached, so on a KEEP with no session directory the adoption
    that credits the integrate is never written and nothing rules on it.
    """

    state = SharedState()
    assert state._session_dir is None

    state.record_kernel_integrate_result(
        {
            "decision": "KEEP",
            "kernel_id": "k001",
            "patch_path": "/artifacts/k001.py",
            "target_file": "/repo/k001.py",
            "gain_pct": 4.0,
        }
    )

    lines = [record.getMessage() for record in traced.records]

    assert any("section=kernel_e2e" in line and "reason=no session_dir" in line for line in lines)


def test_a_keep_that_no_e2e_confirmed_is_not_mistaken_for_a_lost_record(tmp_path, traced):
    """Not every missing adoption is a missing write, and the two must differ.

    A GEMM keep that end-to-end validation never confirmed records its
    operation and deliberately records no adoption. Downstream that is the same
    absence a dropped write leaves, so the reason has to be stated here.
    """
    record_gemm_tuning_operation(
        tmp_path,
        payload={"task_id": "gemm-1"},
        result={"decision": "KEEP", "e2e_validated": False},
    )

    lines = [record.getMessage() for record in traced.records]

    assert any("section=adoptions" in line and "not an e2e-validated keep" in line for line in lines)
