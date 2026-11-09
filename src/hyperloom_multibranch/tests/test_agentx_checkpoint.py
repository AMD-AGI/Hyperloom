# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX checkpoint in a multi-branch session: its mode, the judgement of a stopped candidate, the CLOSE report."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hyperloom.common.agentx_early_stop import EARLY_STOP_ENV, EarlyStopMode
from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_INTVTY_P50_CHECKPOINT
from hyperloom.contracts.measurement import (
    AgentxCheckpointRuling,
    AttachmentRef,
    FailureClass,
    Purpose,
    TestResult,
    Validity,
)
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.codec import decode, encode
from hyperloom_multibranch.blackboard.tests.factories import intent, judgement, request, result
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Verdict
from hyperloom_multibranch.rulegate.agentx import CHECKPOINT_TRAILED, partial_window
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID
from hyperloom_multibranch.rulegate.rules import AgentXRule
from hyperloom_multibranch.scheduling.checkpoint_tally import CheckpointScope, checkpoint_tally
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.specialist.judging import BranchJudge, Judging
from hyperloom_multibranch.tests import test_specialist_judging as judging
from hyperloom_multibranch.tests.test_cli import WORKLOAD_ARGS
from hyperloom_multibranch import cli


def _settings() -> RunSettings:
    return cli.settings_from(cli.build_parser().parse_args(["run", *WORKLOAD_ARGS]), lambda tp: 4)


def _intent():  # type: ignore[no-untyped-def]
    return intent("s1", "c0")


P50, TOTAL = AgentXObjective.P50_INTERACTIVITY, AgentXObjective.TOTAL_TOKEN_THROUGHPUT
TRAILS = AgentxCheckpointRuling(1800.0, 50.0, 45.0, -10.0, 2.0, trails=True, stopped=False)
KEEPS_UP = replace(TRAILS, candidate_p50=50.5, gain_pct=1.0, trails=False)
STOPPED = replace(TRAILS, stopped=True)
CASE = AttachmentRef(FIX_CASE_ID, "dir", "b" * 64)


# -- the mode --------------------------------------------------------------------------------------------------------


def test_a_session_only_records_the_checkpoint_unless_the_operator_enforces_it(monkeypatch) -> None:
    monkeypatch.delenv(EARLY_STOP_ENV, raising=False)
    assert _settings().agentx_early_stop is EarlyStopMode.SHADOW
    monkeypatch.setenv(EARLY_STOP_ENV, "enforce")
    assert _settings().agentx_early_stop is EarlyStopMode.ENFORCE
    monkeypatch.setenv(EARLY_STOP_ENV, "enforced")
    with pytest.raises(ValueError, match="is not one of enforce, shadow, off"):
        _settings()


def test_the_manifest_keeps_the_mode_so_a_resumed_session_judges_as_it_began(monkeypatch) -> None:
    monkeypatch.setenv(EARLY_STOP_ENV, "enforce")
    settings = _settings()
    monkeypatch.setenv(EARLY_STOP_ENV, "off")
    assert decode(RunSettings, encode(settings)).agentx_early_stop is EarlyStopMode.ENFORCE
    older = encode(settings)
    del older["agentx_early_stop"]  # type: ignore[index]
    assert decode(RunSettings, older).agentx_early_stop is EarlyStopMode.SHADOW


# -- a stopped candidate ---------------------------------------------------------------------------------------------


def _stopped(record: TestRecord, ruling: AgentxCheckpointRuling = STOPPED) -> TestRecord:
    failed = replace(
        result(record.request_id, validity=Validity.CANDIDATE_FAILED, failure_reason="agentx_early_stop"),
        agentx_checkpoint=ruling,
    )
    return replace(record, result=failed)


async def test_a_candidate_the_checkpoint_stopped_regresses_on_its_partial_gain_and_records_no_value() -> None:
    record = _stopped(TestRecord(request("r1", _intent())))
    judged = await AgentXRule(P50).first(record, {GRADED_INTVTY_P50: 100.0})
    assert (judged.verdict, judged.refused_by, judged.gain_pct) == (Verdict.REGRESSED, (CHECKPOINT_TRAILED,), -10.0)
    assert judged.recorded_value == {} and judged.accuracy is AccuracyVerdict.NOT_MEASURED
    assert partial_window(judged) and not partial_window(judgement(5.0, ("r2",)))


class _StoppingBlackboard(LocalBlackboard):
    """The toolbox's blackboard, on which the checkpoint stopped every explore of ``SLOW``."""

    async def write_result(self, result: TestResult):  # type: ignore[no-untyped-def]
        record = await self.record(result.request_id)
        if record.request.purpose is Purpose.EXPLORE and record.request.delta == judging.SLOW:
            result = replace(
                result,
                validity=Validity.CANDIDATE_FAILED,
                failure_class=FailureClass.CANDIDATE,
                failure_reason="agentx_early_stop",
                rounds=(),
                accuracy=(),
                agentx_checkpoint=STOPPED,
            )
        return await super().write_result(result)


async def test_the_judge_records_a_stopped_explore_as_regressed_without_a_retest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(judging, "_Checked", _StoppingBlackboard)
    session = judging._Session(tmp_path / "blackboard", checked=True)
    await session.prelude()
    session.judge = BranchJudge(
        judging.BRANCH,
        session.runtime,
        Judging(session.gate, session.toolbox, AgentXRule(P50), session.blobs, session.fixes),
    )

    first = await session.explore(judging.SLOW)
    await session.step()

    record = await session.runtime.record(first)
    assert record.result.validity is Validity.CANDIDATE_FAILED
    assert record.judgement.verdict is Verdict.REGRESSED and record.judgement.refused_by == (CHECKPOINT_TRAILED,)
    assert record.conclusion is Conclusion.RECORDED_ONLY and record.commit_id is None
    assert session.retests() == []


# -- the CLOSE report ------------------------------------------------------------------------------------------------


def _record(request_id: str, ruling: AgentxCheckpointRuling | None, **fields: object) -> TestRecord:
    measured = replace(result(request_id), agentx_checkpoint=ruling)
    return TestRecord(request(request_id, _intent()), measured, **fields)  # type: ignore[arg-type]


def _snapshot(*records: TestRecord, root_median: bool = True) -> SimpleNamespace:
    values = {GRADED_INTVTY_P50: 100.0, **({GRADED_INTVTY_P50_CHECKPOINT: 50.0} if root_median else {})}
    root = _intent().parent_commit
    return SimpleNamespace(records=records, baselines=(BaselineRecord(root, values, {}, {}, ("b0",)),))


def test_the_report_counts_what_a_shadow_checkpoint_would_have_stopped_and_how_those_ended() -> None:
    kept = judgement(3.0, ("r1",))
    snapshot = _snapshot(
        _record("r1", TRAILS, judgement=kept, conclusion=Conclusion.COMMITTED, commit_id="c1"),
        _record("r2", TRAILS, judgement=judgement(-3.0, ("r2",), verdict=Verdict.REGRESSED)),
        _record("r3", TRAILS, judgement=judgement(0.5, ("r3",), verdict=Verdict.FLAT)),
        _record("r4", KEEPS_UP),
        _record("r5", None),
        TestRecord(replace(request("r6", _intent()), attachments=(CASE,)), result("r6")),
        TestRecord(request("r7", _intent(), purpose=Purpose.FINAL), result("r7")),
    )

    tally = checkpoint_tally(snapshot, CheckpointScope(EarlyStopMode.SHADOW, P50))  # type: ignore[arg-type]

    assert (tally.judged, tally.trailing, tally.no_reference, tally.no_ruling) == (4, 3, 0, 1)
    assert tally.outcomes == {"KEEP": 1, "REGRESSED": 1, "FLAT": 1}
    assert tally.lines()[:2] == [
        "Mode shadow; explore tests judged: 4; the checkpoint would have stopped 3.",
        "How the 3 it would have stopped ended: FLAT 1, KEEP 1, REGRESSED 1.",
    ]


def test_an_explore_on_a_parent_without_a_median_is_counted_unjudged_for_want_of_a_reference() -> None:
    snapshot = _snapshot(_record("r1", None), root_median=False)
    tally = checkpoint_tally(snapshot, CheckpointScope(EarlyStopMode.SHADOW, P50))  # type: ignore[arg-type]
    assert (tally.judged, tally.no_reference, tally.no_ruling) == (0, 1, 0)


def test_a_session_on_total_token_throughput_reports_that_no_test_is_judged() -> None:
    tally = checkpoint_tally(_snapshot(_record("r1", None)), CheckpointScope(EarlyStopMode.SHADOW, TOTAL))  # type: ignore[arg-type]
    assert tally.judged == tally.no_reference == tally.no_ruling == 0
    assert tally.lines() == ["Mode shadow; no test is judged under total_token_throughput."]
