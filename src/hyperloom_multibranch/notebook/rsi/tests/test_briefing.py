# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The full startup brief: what was tried, this session's lessons, and what other sessions left, under a cap."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from hyperloom_kb import RemoteReadResult

from hyperloom.contracts.recipe import ConfigDelta, LaunchConfig, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, UnknownRequest, Writer
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer, Verdict
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
from hyperloom_multibranch.notebook.knowledge.recall import (
    INTERACTIVITY_OBJECTIVE,
    THROUGHPUT_OBJECTIVE,
    IdentityIndex,
)
from hyperloom_multibranch.notebook.knowledge.tests.fakes import (
    FakeExperienceReader,
    FakeExport,
    completed_read,
    experience_summary,
)
from hyperloom_multibranch.contracts.notebook.records import PeerReplay, WarmReplay
from hyperloom_multibranch.notebook.knowledge.warm_start import PreparedReplay, WarmLead
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.notebook.rsi.briefing import CrossSessionBriefing, SessionBriefs
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
    index = IdentityIndex(FakeExport([{"id": "exp-faster", "identity": SAME, "objective": THROUGHPUT_OBJECTIVE}]))

    with_index = CrossSessionBriefing(notebook, experiences=_reader(bare), identities=index)
    without = CrossSessionBriefing(notebook, experiences=_reader(bare))

    assert (await with_index.startup_brief("c0")).leads == ("prior: --block-size 16: keep +10.00% [exp-faster]",)
    assert without.identities is None
    assert (await without.startup_brief("c0")).leads == ()


async def test_an_experience_of_the_other_kind_of_session_is_only_a_lesson_to_reverify_naming_its_objective(
    tmp_path: Path,
) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    agentx = {"id": "exp-agentx", "identity": SAME, "objective": INTERACTIVITY_OBJECTIVE}
    faster = {"id": "exp-faster", "identity": SAME, "objective": THROUGHPUT_OBJECTIVE}
    reader = _reader(_recalled("exp-agentx", "keep", 1120.0), _recalled("exp-faster", "keep", 1100.0))

    briefing = CrossSessionBriefing(
        notebook, experiences=reader, identities=IdentityIndex(FakeExport([agentx, faster]))
    )
    brief = await briefing.startup_brief("c0")

    assert brief.leads == ("prior: --block-size 16: keep +10.00% [exp-faster]",)
    assert brief.lessons[-1] == (
        "to re-verify: --block-size 16: keep +12.00% on e2e_interactivity@v1 (objective differ) [exp-agentx]"
    )


@dataclasses.dataclass
class _Repo:
    """The experiment repository as a replay reads it: each branch's head, and the recipe of each commit."""

    heads: dict[str, str]
    recipes: dict[str, Recipe]

    async def head(self, branch: str) -> str:
        return self.heads[branch]

    async def read_recipe(self, commit: str) -> Recipe:
        return self.recipes[commit]


async def test_a_lead_with_a_replay_says_how_to_replay_it_and_the_briefing_replays_it_on_the_branch_head(
    tmp_path: Path,
) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    unpacked = "the record holds no patch or kernel files"
    prepared = PreparedReplay(LEAD.canonical_id, None, LaunchConfig("--enable-expert-parallel"), unpacked)
    head = Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 128"))
    repo = _Repo({"s3": "c3"}, {"c3": head})
    warm = (dataclasses.replace(LEAD, replay=prepared),)
    briefing = CrossSessionBriefing(notebook, warm=warm, branch="s3", repo=repo)

    brief = await briefing.startup_brief("c1")
    replayed = await briefing.warm_replay(LEAD.canonical_id)

    assert (
        "recipe: inference:older: +10.00% `--enable-expert-parallel` [older]; "
        f"replay on your head with notebook_replay: its launch config only, {unpacked}"
    ) in brief.leads
    assert replayed == WarmReplay(
        LEAD.canonical_id,
        "c3",
        RecipeDelta(config=ConfigDelta(server_args="--enable-expert-parallel", remove_args=("--max-num-seqs",))),
        IntentLayer.CONFIG,
        unpacked,
    )
    assert await briefing.warm_replay("inference:none") is None
    assert await CrossSessionBriefing(notebook, warm=warm, branch="s3").warm_replay(LEAD.canonical_id) is None


async def test_another_branchs_tested_change_is_replayed_on_the_branch_head_as_it_was_tested(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    head = Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 128"))
    briefing = CrossSessionBriefing(notebook, branch="s3", repo=_Repo({"s3": "c3"}, {"c3": head}))

    replayed = await briefing.peer_replay("r1")

    tested = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))
    assert replayed == PeerReplay("r1", "s1", "c3", tested, IntentLayer.CONFIG)


async def test_a_peer_replay_says_why_when_there_is_nothing_to_replay_on_the_branch_head(tmp_path: Path) -> None:
    root = tmp_path / "notebook"
    notebook = await scripted_session(root, FakeClock())
    toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"), clock=FakeClock())
    dropped = RecipeDelta(remove_kernels=("kernel/gemm/old.csv",))
    await toolbox.write_request(request("r8", SPLIT_KV, dropped))
    await toolbox.write_request(request("r9", SPLIT_KV, RecipeDelta()))
    plain = Recipe("sha256:" + "a" * 64, LaunchConfig("--max-num-seqs 128"))
    tuned = dataclasses.replace(plain, launch=LaunchConfig("--max-num-seqs 256"))

    def replaying(branch: str, recipe: Recipe | None = plain) -> CrossSessionBriefing:
        repo = _Repo({branch: "c3"}, {"c3": recipe}) if recipe is not None else None
        return CrossSessionBriefing(notebook, branch=branch, repo=repo)

    for briefing, request_id, why in (
        (replaying("s3"), "r8", "no longer applies to your head c3: the recipe loads no kernel artifacts"),
        (replaying("s3"), "r9", "r9 was measured without a change"),
        (replaying("s1"), "r1", "r1 is this branch's own test"),
        (replaying("s3", tuned), "r1", "your head c3 already has the change r1 tested"),
        (replaying("s3", None), "r1", "reads no experiment repository"),
    ):
        with pytest.raises(ValueError, match=re.escape(why)):
            await briefing.peer_replay(request_id)
    with pytest.raises(UnknownRequest):
        await replaying("s3").peer_replay("r404")


async def test_recalled_leads_take_turns_by_the_layer_they_change(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())

    def kept(experience_id: str, outcome: float, family: str) -> dict[str, object]:
        summary = experience_summary(experience_id, SAME, outcome=outcome, change=f"change {experience_id}")
        return {**summary, "change_family": family}

    reader = _reader(
        kept("exp-c1", 1100.0, "config_variant"),
        kept("exp-c2", 1080.0, "config_variant"),
        kept("exp-c3", 1060.0, "config_variant"),
        kept("exp-s1", 1050.0, "source_patch"),
    )

    brief = await CrossSessionBriefing(notebook, experiences=reader).startup_brief("c0")

    assert brief.leads == (
        "prior (config): change exp-c1: keep +10.00% [exp-c1]",
        "prior (source): change exp-s1: keep +5.00% [exp-s1]",
        "prior (config): change exp-c2: keep +8.00% [exp-c2]",
        "prior (config): change exp-c3: keep +6.00% [exp-c3]",
    )


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


async def test_the_brief_logs_the_leads_it_kept_under_its_cap_with_the_change_each_suggests(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    log = tmp_path / LEARNINGS_LOG
    recall = LearningRecall(FakeExperienceReader(completed_read()), log=RetrievalLog(log))
    full = CrossSessionBriefing(
        notebook, warm=(LEAD,), experiences=_reader(_recalled("exp-faster", "keep", 1100.0)), learnings=recall
    )

    await full.startup_brief("c0")
    await dataclasses.replace(full, max_chars=100).startup_brief("c0")
    first, capped = [json.loads(line)["leads"] for line in log.read_text(encoding="utf-8").splitlines()]

    recipe = {"ref": "inference:older", "change": "--enable-expert-parallel"}
    assert first == [recipe, {"ref": "exp-faster", "change": "--block-size 16"}]
    assert capped == [recipe]


async def test_the_brief_logs_the_leads_it_shows_and_only_those(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    log = tmp_path / LEARNINGS_LOG
    recall = LearningRecall(FakeExperienceReader(completed_read()), log=RetrievalLog(log))
    reader = _reader(_recalled("exp-faster", "keep", 1100.0))
    full = CrossSessionBriefing(notebook, warm=(LEAD,), experiences=reader, branch="s3", learnings=recall)

    await full.startup_brief("c0")
    await dataclasses.replace(full, max_chars=100).startup_brief("c0")
    first, capped = [json.loads(line)["leads"] for line in log.read_text(encoding="utf-8").splitlines()]

    recipe = {"ref": "inference:older", "change": "--enable-expert-parallel"}
    assert first == [recipe, {"ref": "exp-faster", "change": "--block-size 16"}]
    assert capped == [recipe]


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


class _Changing(_WithExtraFailure):
    """The notebook, which gets the extra failure once ``extra`` is set."""

    extra = False

    async def snapshot(self) -> NotebookSnapshot:
        return await super().snapshot() if self.extra else await self.notebook.snapshot()


async def test_a_brief_built_again_shows_the_notebook_then_and_reads_the_kb_once(tmp_path: Path) -> None:
    changing = _Changing(await scripted_session(tmp_path / "notebook", FakeClock()))
    reader = _reader(_recalled("exp-faster", "keep", 1100.0))
    briefs = SessionBriefs(changing, experiences=reader, metric=METRIC)
    repeated = "this session: attention / split kv failed 2 times on the states this one builds on"

    first = await briefs.text("s3", "c1", focus="attention")
    changing.extra = True
    later = await briefs.text("s3", "c1")

    assert repeated not in first and repeated in later
    assert "prior: --block-size 16: keep +10.00% [exp-faster]" in later
    ((_, context, _),) = reader.calls
    assert (context["observations"]["bottleneck"], context["observations"]["branch"]) == ("attention", "s3")


async def test_a_read_that_did_not_complete_is_read_again_at_the_next_brief(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    failed = FakeExperienceReader(RemoteReadResult("", "unavailable", "", (), ("planner_timeout",)))
    briefs = SessionBriefs(notebook, experiences=failed)

    await briefs.text("s3", "c1")
    await briefs.text("s3", "c1")

    assert len(failed.calls) == 2


async def test_each_branch_keeps_its_focus_and_every_branch_shares_one_index_and_cache(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    briefs = SessionBriefs(notebook, experiences=_reader())

    briefs.for_branch("s3", "attention")
    s3, s4 = briefs.for_branch("s3"), briefs.for_branch("s4")

    assert (s3.focus, s4.focus) == ("attention", "")
    assert s3.recalls is s4.recalls and s3.identities is s4.identities
