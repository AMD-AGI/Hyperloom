# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``warm_start`` event.

The event this replaces was projected, and these tests pin the three defects
that projection had.

The status answered the wrong question: a miss was published as
``not_matched``, so a first-ever session for a workload -- an empty KB
answering correctly -- was filed under a word that reads as a malfunction.

The reads block described the wrong window. It aggregated the session's whole
recipe audit log inside an event that covers T0 alone, which pulled in every
successful write (the log carries those with ``hit: True``) and every
mid-session amendment read.

And two of its fields could never be populated: ``by_source`` and
``best_config_by_source`` read keys no producer has ever written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import EVENT_STATUS_INTERRUPTED
from hyperloom.inference_optimizer.breakdown.recorder.warm_start_event import (
    MATCH_HIT,
    MATCH_MISS,
    MATCH_SEED_ONLY,
    make_warm_start_recorder,
    matched_block,
    record_read,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope

CID = "inference:qwen3-8b:mi355x:sglang:qwen3:qwen3forcausallm:0.5.17:bf16"


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "warm_start"]


def _event(session_dir: Path) -> dict[str, Any]:
    events = _events(session_dir)
    assert len(events) == 1, f"expected one warm_start event, got {len(events)}"
    return events[0]


def _recorder(**overrides: Any):
    """Open the event the way T0 opens it, once the identity is resolved."""
    kwargs: dict[str, Any] = {
        "macro_cycle": 0,
        "requested_canonical_id": CID,
        "scope": {"kernel_optimizer": "geak", "tp": 1, "conc": 64, "isl": 8192, "osl": 1024},
        "start_time": "2026-08-20T07:25:00+00:00",
    }
    kwargs.update(overrides)
    recorder = make_warm_start_recorder(**kwargs)
    assert recorder is not None
    return recorder


def _recipe(**overrides: Any) -> dict[str, Any]:
    row = {
        "canonical_id": CID,
        "best_throughput": 3239.9,
        "validated_gain_pct": 36.16,
        "replayable": True,
        "replay_material_available": True,
        "view_source": "current",
        "workload_shape": {"tp": 1, "conc": 64, "isl": 8192, "osl": 1024},
        "remote_session_id": "20260818T063226Z",
        "sessions": [{"session_id": "20260818T063226Z", "gain_pct": 36.23}],
        "provenance": {"session_id": "20260818T063226Z"},
    }
    row.update(overrides)
    return row


def _matched(*, tier: str = "exact") -> dict[str, Any]:
    return matched_block(
        tier=tier,
        confidence=1.0 if tier == "exact" else 0.72,
        source="kb-store",
        canonical_id=CID,
        recipe=_recipe(),
        expected_gain_pct=36.23,
        lessons=[{"a": 1}, {"b": 2}],
        pitfalls=[{"c": 3}],
    )


def _read(**overrides: Any) -> dict[str, Any]:
    """One audit event shaped the way ``RecipeKB._read_event`` shapes them."""
    event = {
        "op": "read",
        "method": "get_recipe",
        "mode": "local",
        "backend": "local-json",
        "remote": "none",
        "resolution": "local",
        "hit": True,
        "candidates": 1,
        "request": {"canonical_id": CID},
        "result": {
            "canonical_id": CID,
            "exact": True,
            "best_throughput": 3239.9,
            "best_config_nonempty": True,
        },
    }
    event.update(overrides)
    return event


# ---------------------------------------------------------------------------
# the lookup's own outcome, kept apart from what it found
# ---------------------------------------------------------------------------


def test_an_exact_hit_reports_identity_gain_and_origin(_bound_session):
    recorder = _recorder()
    recorder.finish(match_status=MATCH_HIT, matched=_matched())

    event = _event(_bound_session)
    assert event["status"] == "succeeded"
    assert event["ext"]["match_status"] == MATCH_HIT
    matched = event["ext"]["matched"]
    assert matched["match_type"] == "exact"
    assert matched["canonical_id"] == CID
    assert matched["optimized_throughput"] == pytest.approx(3239.9)
    assert matched["expected_gain_pct"] == pytest.approx(36.23)
    assert matched["origin"] == {"session_id": "20260818T063226Z", "gain_pct": pytest.approx(36.23)}
    assert matched["experience"] == {"lessons_count": 2, "pitfalls_count": 1}


def test_a_miss_is_a_completed_lookup_and_not_a_failure(_bound_session):
    """The projection published this as ``not_matched``, its own event status.

    A workload nobody has optimized yet has an empty KB, and an empty KB
    answering "nothing here" is the lookup working. Reporting that in the
    status field made every cold start read as though a step had gone wrong.
    """
    _recorder().finish(match_status=MATCH_MISS)

    event = _event(_bound_session)
    assert event["status"] == "succeeded"
    assert event["ext"]["match_status"] == MATCH_MISS
    assert event["ext"]["matched"] is None


def test_a_seed_only_match_is_a_third_state_and_not_a_bad_status(_bound_session):
    """A record was found and cannot be executed, which is neither of the two.

    It rides ``match_status`` rather than the event status, because T0 stamps
    its own anchor row before searching and then matches it -- so grading
    ``seed_only`` in the status would mark every cold start as unhealthy.
    """
    recorder = _recorder()
    recorder.finish(match_status=MATCH_SEED_ONLY, matched=_matched())

    event = _event(_bound_session)
    assert event["status"] == "succeeded"
    assert event["ext"]["match_status"] == MATCH_SEED_ONLY
    # Described, not dropped: what was found is why this is not a plain miss.
    assert event["ext"]["matched"]["canonical_id"] == CID


def test_a_lookup_that_broke_is_separated_from_one_that_found_nothing(_bound_session):
    _recorder().finish(match_status=MATCH_MISS, error="RecipeStoreTimeout")

    event = _event(_bound_session)
    assert event["status"] == "failed"
    assert event["ext"]["failure"] == {"error_class": "RecipeStoreTimeout"}


def test_a_degraded_tier_is_not_reported_as_exact(_bound_session):
    recorder = _recorder()
    recorder.finish(match_status=MATCH_HIT, matched=_matched(tier="compatible_framework_version"))

    matched = _event(_bound_session)["ext"]["matched"]
    assert matched["match_type"] == "degraded"
    assert matched["tier"] == "compatible_framework_version"


def test_the_queried_identity_is_recorded_as_it_is_queried(_bound_session):
    """Read rather than rebuilt: the hardware dimension is topology-aware."""
    recorder = _recorder()
    recorder.finish(match_status=MATCH_MISS)

    request = _event(_bound_session)["ext"]["request"]
    assert request["canonical_id"] == CID
    assert request["scope"]["conc"] == 64


# ---------------------------------------------------------------------------
# reads: T0's own, one row each
# ---------------------------------------------------------------------------


def test_each_read_is_kept_as_its_own_row_with_the_tallies_over_them(_bound_session):
    recorder = _recorder()
    record_read(_bound_session, _read())
    record_read(_bound_session, _read(method="search", resolution="local", hit=False, result=None))
    recorder.finish(match_status=MATCH_HIT, matched=_matched())

    reads = _event(_bound_session)["ext"]["reads"]
    assert reads["count"] == 2
    assert reads["hits"] == 1
    assert reads["by_method"] == {"get_recipe": 1, "search": 1}
    assert reads["by_resolution"] == {"local": 2}
    assert len(reads["rows"]) == 2
    assert reads["rows"][0]["matched_canonical_id"] == CID
    assert reads["rows"][0]["exact"] is True
    # A miss has no result to describe, so it says nothing rather than
    # reporting a row with everything in it zeroed.
    assert reads["rows"][1]["matched_canonical_id"] is None
    assert reads["rows"][1]["exact"] is None


def test_a_write_is_not_counted_as_a_read(_bound_session):
    """``put_recipe`` logs its success with ``hit: True``.

    The projection aggregated the audit log without filtering by operation, so
    every recipe the session wrote inflated both the read count and the hit
    count of a lookup that had already finished.
    """
    recorder = _recorder()
    record_read(_bound_session, _read())
    record_read(
        _bound_session,
        {
            "op": "write",
            "method": "put_recipe",
            "resolution": "local_write",
            "success": True,
            "hit": True,
            "request": {"canonical_id": CID},
            "result": {"canonical_id": CID, "version": 3, "created": False},
        },
    )
    recorder.finish(match_status=MATCH_HIT, matched=_matched())

    reads = _event(_bound_session)["ext"]["reads"]
    assert reads["count"] == 1
    assert reads["hits"] == 1
    assert reads["by_method"] == {"get_recipe": 1}


def test_a_read_served_after_the_lookup_settled_belongs_to_nobody(_bound_session):
    """``_kb_amend_recipe`` consults the same store mid-session.

    Its reads are real reads through the same audit hook, and the projection
    counted them into the anchor's tally. The event's own open interval is what
    separates them.
    """
    recorder = _recorder()
    record_read(_bound_session, _read())
    recorder.finish(match_status=MATCH_HIT, matched=_matched())

    record_read(_bound_session, _read(method="get_authoritative_recipe"))

    reads = _event(_bound_session)["ext"]["reads"]
    assert reads["count"] == 1
    assert reads["by_method"] == {"get_recipe": 1}


def test_a_read_served_before_the_lookup_opened_belongs_to_nobody(_bound_session):
    record_read(_bound_session, _read(method="get_authoritative_recipe"))
    recorder = _recorder()
    record_read(_bound_session, _read())
    recorder.finish(match_status=MATCH_HIT, matched=_matched())

    assert _event(_bound_session)["ext"]["reads"]["by_method"] == {"get_recipe": 1}


def test_a_lookup_left_open_by_one_session_cannot_claim_another_s_reads(tmp_path, _bound_session):
    """T0 raising before it settles leaves its window open.

    The window is scoped to the session it was opened in, so the leak cannot
    cross into the next session and attribute its reads to an anchor that
    belongs to a different run and never finished.
    """
    _recorder()  # opened and never settled

    other = tmp_path / "other-session"
    other.mkdir()
    with session_scope(other):
        live = _recorder()
        record_read(other, _read())
        live.finish(match_status=MATCH_HIT, matched=_matched())
        assert _event(other)["ext"]["reads"]["count"] == 1

    # The leaked window claims nothing served outside its own session.
    record_read(other, _read(method="search"))
    assert _event(other)["ext"]["reads"]["count"] == 1


def test_a_lookup_that_read_nothing_carries_no_reads_block(_bound_session):
    _recorder().finish(match_status=MATCH_MISS)

    assert _event(_bound_session)["ext"]["reads"] is None


def test_the_retired_by_source_fields_are_gone(_bound_session):
    """They read ``result.sources`` / ``result.best_config_source``.

    No producer in the codebase has ever written either key, so both maps were
    always empty -- which reads as "no source was involved" rather than as
    "this was never recorded".
    """
    recorder = _recorder()
    record_read(_bound_session, _read())
    recorder.finish(match_status=MATCH_HIT, matched=_matched())

    reads = _event(_bound_session)["ext"]["reads"]
    assert "by_source" not in reads
    assert "best_config_by_source" not in reads


# ---------------------------------------------------------------------------
# a lookup that never settled
# ---------------------------------------------------------------------------


def test_a_session_killed_mid_lookup_keeps_the_identity_it_asked_for(_bound_session):
    """The request rides on the open shell, so the event is not anonymous."""
    _recorder()

    event = _event(_bound_session)
    assert event["status"] == "running"
    assert event["ext"]["request"]["canonical_id"] == CID


def test_finalize_closes_a_lookup_that_outlived_its_process(_bound_session):
    recorder = _recorder()
    record_read(_bound_session, _read())
    del recorder

    finalize_events(_bound_session)

    event = _event(_bound_session)
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    # The reads it did get to make say how far it got.
    assert event["ext"]["reads"]["count"] == 1
