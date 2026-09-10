# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A recording that fails has to say so.

Recording is best-effort: the run outranks its own record. That is only
tolerable if the loss is reported, otherwise the export looks complete while
missing facts, and nothing downstream can tell the difference.
"""

from __future__ import annotations

import logging

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import recorder_warnings as rw
from hyperloom.inference_optimizer.session import sbd_v6
from hyperloom.inference_optimizer.session.session_binding import session_scope


@pytest.fixture(autouse=True)
def _forget_earlier_notes():
    rw.reset_for_tests()
    yield
    rw.reset_for_tests()


def test_a_lost_write_is_parked_where_the_export_reads_it(tmp_path):
    with session_scope(tmp_path):
        rw.note_failure(section="phase_event", error=OSError("disk full"), detail="recording the phase entry")

    parked = sbd_v6.read_write_warnings(tmp_path)
    assert any("recorder.phase_event" in w for w in parked)
    assert any("disk full" in w for w in parked)


def test_a_lost_write_reaches_the_exported_breakdown(tmp_path):
    with session_scope(tmp_path):
        rw.note_failure(section="stack_event", error=RuntimeError("spool gone"))

    from hyperloom.inference_optimizer.breakdown import exporter

    reported = exporter.build(tmp_path)["metadata"]["warnings"]
    assert any("recorder.stack_event" in w for w in reported)


def test_the_same_failure_is_reported_once(tmp_path, caplog):
    with session_scope(tmp_path), caplog.at_level(logging.WARNING):
        for _ in range(5):
            rw.note_failure(section="phase_event", error=OSError("disk full"))

    assert sum("phase_event" in r.message for r in caplog.records) == 1


def test_a_different_failure_of_the_same_section_is_still_reported(tmp_path, caplog):
    with session_scope(tmp_path), caplog.at_level(logging.WARNING):
        rw.note_failure(section="phase_event", error=OSError("disk full"))
        rw.note_failure(section="phase_event", error=ValueError("bad row"))

    assert sum("phase_event" in r.message for r in caplog.records) == 2


def test_a_session_that_is_not_bound_still_logs_the_loss(caplog):
    """There is nowhere to park it, so the log is the whole record."""
    with caplog.at_level(logging.WARNING):
        rw.note_failure(section="close", error=OSError("nope"))

    assert any("close" in r.message for r in caplog.records)


def test_noting_a_failure_never_raises(tmp_path, monkeypatch):
    """It runs inside the handler for a failure, so it cannot add one."""

    def _boom(*args, **kwargs):
        raise RuntimeError("the sidecar is gone too")

    monkeypatch.setattr(sbd_v6, "record_write_warning", _boom)
    with session_scope(tmp_path):
        rw.note_failure(section="phase_event", error=OSError("disk full"))
