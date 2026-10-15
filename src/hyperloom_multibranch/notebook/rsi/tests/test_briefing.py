# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The full startup brief: what was tried, this session's lessons, and what other sessions left, under a cap."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from hyperloom_kb import RemoteReadResult

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.peers import Verdict
from hyperloom_multibranch.contracts.notebook.records import (
    BaselineRecord,
    Learning,
    LineageEvent,
    NotebookSnapshot,
    StartupBrief,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LearningKind, LineageKind
from hyperloom_multibranch.notebook.knowledge.learning_recall import LearningRecall
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.knowledge.recall import IdentityIndex
from hyperloom_multibranch.notebook.knowledge.tests.fakes import (
    FakeExperienceReader,
    FakeExport,
    completed_read,
    experience_summary,
)
from hyperloom_multibranch.notebook.knowledge.warm_start import WarmLead
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.notebook.rsi.briefing import CrossSessionBriefing
from hyperloom_multibranch.notebook.tests.factories import FakeClock, judgement, profile, request, result
from hyperloom_multibranch.notebook.tests.sessions import METRIC, SPLIT_KV, scripted_session

SAME = {
    "model": "deepseek-ai/DeepSeek-R1",
    "gpu": "mi355x",
    "framework": "vllm",
    "tp": 8,
    "conc": 64,
    "bottleneck": "attention",
    "isl": 1024,
    "osl": 1024,
}
LEAD = WarmLead(
    canonical_id="inference:older",
    session_id="older",
    scope={},
    gain_pct=10.0,
    best_throughput=1100.0,
    server_args="--enable-expert-parallel",
    envs={},
    replayable=True,
    pitfalls=("fp8 kv cache fails accuracy",),
    what_failed=("block size 32 regressed",),
)


def _recalled(experience_id: str, decision: str, outcome: float, gpu: str = "mi355x") -> dict[str, object]:
    identity = {**SAME, "gpu": gpu}
    return experience_summary(experience_id, identity, decision=decision, outcome=outcome, change="--block-size 16")


def _reader(*experiences: dict[str, object]) -> FakeExperienceReader:
    return FakeExperienceReader(completed_read(*experiences))


class _WithExtraFailure:
    """The notebook, with one more failed split-kv test on ``c1``."""

    def __init__(self, notebook: LocalNotebook) -> None:
        self.notebook = notebook

    async def startup_brief(self, start_commit: str) -> StartupBrief:
        return await self.notebook.startup_brief(start_commit)

    async def snapshot(self) -> NotebookSnapshot:
        snapshot = await self.notebook.snapshot()
        again = dataclasses.replace(SPLIT_KV, parent_commit="c1")
        slower = judgement(-1.0, ("r9",), verdict=Verdict.REGRESSED)
        extra = TestRecord(request("r9", again), result("r9"), slower, Conclusion.RECORDED_ONLY)
        return dataclasses.replace(snapshot, records=(*snapshot.records, extra))


async def test_the_brief_holds_this_sessions_lessons_and_both_kinds_of_recall(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    reader = _reader(
        _recalled("exp-faster", "keep", 1100.0),
        _recalled("exp-slower", "revert", 950.0),
        _recalled("exp-elsewhere", "keep", 1200.0, gpu="mi300x"),
    )
    briefing = CrossSessionBriefing(
        _WithExtraFailure(notebook), warm=(LEAD,), experiences=reader, metric=METRIC, focus="attention", branch="s3"
    )

    brief = await briefing.startup_brief("c1")

    assert {item.request_id for item in brief.tried} == {"r1", "r2", "r3", "r4", "r5"}
    assert brief.lessons == (
        "this session: attention / split kv failed 2 times on the states this one builds on",
        "prior: --block-size 16: revert -5.00% [exp-slower]",
        "to re-verify: --block-size 16: keep +20.00% (gpu differ) [exp-elsewhere]",
        "inference:older: fp8 kv cache fails accuracy",
        "inference:older: block size 32 regressed",
    )
    assert brief.leads == (
        "recipe: inference:older: +10.00% `--enable-expert-parallel` [older]",
        "prior: --block-size 16: keep +10.00% [exp-faster]",
    )


async def test_the_cap_keeps_the_most_specific_items(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    briefing = CrossSessionBriefing(_WithExtraFailure(notebook), warm=(LEAD,), max_chars=150)

    brief = await briefing.startup_brief("c1")

    assert len(brief.lessons) == 1 and brief.lessons[0].startswith("this session:")
    assert brief.leads == ("recipe: inference:older: +10.00% `--enable-expert-parallel` [older]",)


async def test_the_first_batch_brief_is_cross_session_only(tmp_path: Path) -> None:
    clock = FakeClock()

    def as_role(role: Role) -> LocalNotebook:
        return LocalNotebook(tmp_path / "fresh", Writer(role, role.value), clock=clock)

    await as_role(Role.SCHEDULER).write_session_profile(profile())
    await as_role(Role.SCHEDULER).record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await as_role(Role.GATE).write_baseline(BaselineRecord("c0", {METRIC: 1000.0}, {METRIC: 0.8}, {METRIC: 1.6}, ()))
    briefing = CrossSessionBriefing(
        as_role(Role.OBSERVER), warm=(LEAD,), experiences=_reader(_recalled("exp-faster", "keep", 1100.0))
    )

    brief = await briefing.startup_brief("c0")

    assert brief.tried == ()
    assert brief.lessons == ("inference:older: fp8 kv cache fails accuracy", "inference:older: block size 32 regressed")
    assert brief.leads == (
        "recipe: inference:older: +10.00% `--enable-expert-parallel` [older]",
        "prior: --block-size 16: keep +10.00% [exp-faster]",
    )


async def test_a_recalled_summary_without_conditions_is_sorted_by_the_index_the_briefing_holds(
    tmp_path: Path,
) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    bare = {key: value for key, value in _recalled("exp-faster", "keep", 1100.0).items() if key != "identity"}
    index = IdentityIndex(FakeExport([{"id": "exp-faster", "identity": SAME}]))

    with_index = CrossSessionBriefing(notebook, experiences=_reader(bare), identities=index)
    without = CrossSessionBriefing(notebook, experiences=_reader(bare))

    assert (await with_index.startup_brief("c0")).leads == ("prior: --block-size 16: keep +10.00% [exp-faster]",)
    assert without.identities is None
    assert (await without.startup_brief("c0")).leads == ()


async def test_the_lessons_hold_what_this_session_and_past_sessions_learnt_about_the_focus(tmp_path: Path) -> None:
    clock = FakeClock()
    notebook = await scripted_session(tmp_path / "notebook", clock)
    s2 = LocalNotebook(tmp_path / "notebook", Writer(Role.SPECIALIST_RUNTIME, "s2", "s2"), clock=clock)
    learnt = await s2.append_learning(
        Learning(
            "s2", LearningKind.PITFALL, "split kv regresses when attention is not the bottleneck", request_ids=("r2",)
        )
    )
    past = {
        "experience_id": "exp-past",
        "source_run_id": "sess-0",
        "identity": {**SAME, "tp": 4},
        "change_summary": "attention split kv pays only past conc 128",
        "decision": "measured",
    }
    log = tmp_path / LEARNINGS_LOG
    recall = LearningRecall(FakeExperienceReader(completed_read(past)), log=RetrievalLog(log))

    briefing = CrossSessionBriefing(notebook, focus="attention split kv", branch="s3", learnings=recall)
    brief = await briefing.startup_brief("c0")
    alone = await CrossSessionBriefing(notebook, focus="attention split kv").startup_brief("c0")
    (logged,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    assert (logged["branch"], logged["source"], logged["query"]) == ("s3", "brief", "attention split kv")
    assert (logged["here"], logged["past"]) == ([learnt.entry_id], ["exp-past"])

    assert (
        f"learnt here: [s2, pitfall] split kv regresses when attention is not the bottleneck "
        f"(r2: regressed -2.00%, recorded_only) [{learnt.entry_id}]"
    ) in brief.lessons
    assert (
        "learnt before: [sess-0, tp differ, measured] attention split kv pays only past conc 128 [exp-past]"
        in brief.lessons
    )
    assert not any(lesson.startswith("learnt before") for lesson in alone.lessons)
    assert any(lesson.startswith("learnt here") for lesson in alone.lessons)


async def test_without_a_kb_or_a_profile_or_a_completed_read_nothing_is_recalled(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    failed = FakeExperienceReader(RemoteReadResult("", "unavailable", "", (), ("planner_timeout",)))

    class _NoProfile(_WithExtraFailure):
        async def snapshot(self) -> NotebookSnapshot:
            return dataclasses.replace(await self.notebook.snapshot(), profile=None)

    unavailable = await CrossSessionBriefing(notebook, experiences=failed).startup_brief("c1")
    no_profile = await CrossSessionBriefing(_NoProfile(notebook), experiences=_reader()).startup_brief("c1")
    no_kb = await CrossSessionBriefing(notebook).startup_brief("c1")

    assert unavailable.lessons == unavailable.leads == no_profile.leads == no_kb.leads == ()
