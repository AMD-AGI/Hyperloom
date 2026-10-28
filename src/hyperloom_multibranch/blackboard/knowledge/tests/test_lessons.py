# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The lists a finished session's knowledge carries, read from the scripted session."""

from __future__ import annotations

import dataclasses

from hyperloom_multibranch.contracts.blackboard.peers import Validity, Verdict
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion
from hyperloom_multibranch.blackboard.knowledge.lessons import (
    MAX_ITEM_CHARS,
    MAX_ITEMS,
    LessonKind,
    lesson_of,
    listed,
    per_test_lessons,
    roofline_gaps,
    session_lessons,
)
from hyperloom_multibranch.blackboard.tests.factories import judgement, profile, request, result
from hyperloom_multibranch.blackboard.tests.sessions import GEMM


def test_every_finished_test_teaches_one_lesson(snapshot: BlackboardSnapshot) -> None:
    lessons = {
        lesson.request_id: (lesson.kind, lesson.direction, lesson.parent_commit)
        for lesson in per_test_lessons(snapshot)
    }

    assert lessons == {
        "r1": (LessonKind.WORKED, "moe gemm / tuned kernel", "c0"),
        "r2": (LessonKind.FAILED, "attention / split kv", "c0"),
        "r3": (LessonKind.PITFALL, "scheduler / chunked prefill", "c0"),
        "r5": (LessonKind.PITFALL, "kv cache / fp8 kv cache", "c0"),
        "r4": (LessonKind.FAILED, "moe gemm / tuned kernel", "c1"),
        "m1": (LessonKind.WORKED, "merge of c1", "c0"),
    }


def test_the_session_lists_put_the_best_and_the_worst_first(snapshot: BlackboardSnapshot) -> None:
    lessons = session_lessons(snapshot, "i1")

    assert lessons.worked == (
        "merge of c1 on c0: +4.10%, merged as i1",
        "moe gemm / tuned kernel on c0: +4.00%, committed as c1",
    )
    assert lessons.failed == (
        "attention / split kv on c0: regressed at -2.00%",
        "moe gemm / tuned kernel on c1: flat at +0.30%",
    )
    assert lessons.pitfalls == (
        "scheduler / chunked prefill on c0: the candidate failed (server exited during warmup)",
        "kv cache / fp8 kv cache on c0: +6.00% but failed the accuracy check",
    )
    assert lessons.learned == (
        "confirmed: the MoE GEMM is bound by memory bandwidth",
        "falsified: splitting the KV cache helps at this concurrency",
    )
    assert lessons.gaps == ("output_tps: 1041.0 is 69% of the roofline bound 1500.0",)
    assert session_lessons(snapshot).gaps == ()


def test_a_test_in_flight_unmeasured_unverified_or_improved_without_a_commit_teaches_nothing(
    snapshot: BlackboardSnapshot,
) -> None:
    in_flight = TestRecord(request("r9", GEMM))
    unmeasured = TestRecord(request("r9", GEMM), result("r9", validity=Validity.MEASUREMENT_FAILED))
    regressed = judgement(-2.0, ("r9",), verdict=Verdict.REGRESSED)
    unverified = TestRecord(request("r9", GEMM), result("r9"), regressed, Conclusion.UNVERIFIED)
    kept_aside = TestRecord(request("r9", GEMM), result("r9"), judgement(1.0, ("r9",)), Conclusion.RECORDED_ONLY)

    assert lesson_of(in_flight, snapshot) is None
    assert lesson_of(unmeasured, snapshot) is None
    assert lesson_of(unverified, snapshot) is None
    assert lesson_of(kept_aside, snapshot) is None


def test_an_unsettled_judgement_teaches_nothing_and_a_blocked_gain_is_a_pitfall(snapshot: BlackboardSnapshot) -> None:
    def recorded(verdict: Verdict, refused_by: tuple[str, ...] = ()) -> TestRecord:
        judged = dataclasses.replace(judgement(5.0, ("r9",), verdict=verdict), refused_by=refused_by)
        return TestRecord(request("r9", GEMM), result("r9"), judged, Conclusion.RECORDED_ONLY)

    assert lesson_of(recorded(Verdict.NEEDS_CONFIRM), snapshot) is None
    assert lesson_of(recorded(Verdict.INCONCLUSIVE), snapshot) is None
    blocked = lesson_of(recorded(Verdict.BLOCKED, ("p90_regressed",)), snapshot)
    assert blocked is not None
    assert (blocked.kind, blocked.outcome) == (LessonKind.PITFALL, "+5.00% but held back by p90_regressed")


def test_a_test_without_an_intent_is_named_by_its_purpose_and_kind(snapshot: BlackboardSnapshot) -> None:
    unknown = dataclasses.replace(snapshot, intents={}, committed=())
    slower = judgement(-1.0, ("r9",), verdict=Verdict.REGRESSED)
    record = TestRecord(request("r9", GEMM), result("r9"), slower, Conclusion.RECORDED_ONLY)

    lesson = lesson_of(record, unknown)

    assert lesson is not None and lesson.direction == "explore test r9"


def test_lists_drop_repeats_clip_long_items_and_stop_at_the_cap() -> None:
    long = "x" * (MAX_ITEM_CHARS + 50)

    assert listed(["a", "a", long]) == ("a", "x" * (MAX_ITEM_CHARS - 3) + "...")
    assert len(listed(str(index) for index in range(MAX_ITEMS + 5))) == MAX_ITEMS


def test_gaps_need_a_profile_and_a_positive_bound() -> None:
    assert roofline_gaps(None, {"output_tps": 1.0}) == ()
    assert roofline_gaps(profile(roofline_ceiling={"output_tps": 0.0, "ttft": 2.0}), {"output_tps": 1.0}) == ()
