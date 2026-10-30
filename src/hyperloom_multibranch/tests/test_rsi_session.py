# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The RSI's scheduling playbook end to end: sessions run through the run entry on the fake toolbox and peers, as
:mod:`~hyperloom_multibranch.tests.test_skeleton` runs them, with a judge that plays the playbook as written.

A session with an archive plays its champion: the judge reads the playbook, the open areas and each branch's stall
evidence, a branch that stalls past the bar is a stop candidate the judge stops, one drawn for probation runs its
probation tests through a stop the envelope refuses, and CLOSE keeps the session as a trial of the playbook with its
stop audit. A challenger the command line asks for plays the oldest playbook on trial until its judge orders a stop
the envelope refuses, then the champion. A session without an archive, or under ``HYPERLOOM_RSI_WRITE_ONLY=1``, plays
none and runs as one without the RSI. Where the playbook meets the judge's own rules, its grace holds a stalled
branch, a branch behind integration whose commits merge is no stop candidate, and a branch the judge stopped is scored
no more while its tests run on. With too little of EXPLORE left for a new branch's first commit, every refill the
judge asks for is cut.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch import cli
from hyperloom_multibranch.contracts.llm import CompletionReply
from hyperloom_multibranch.contracts.blackboard import Role
from hyperloom_multibranch.blackboard.rsi.archive import (
    ARCHIVE_ENV,
    ARCHIVED,
    NO_ARCHIVE,
    SKIPPED,
    WRITE_ONLY_ENV,
    TreeArchive,
)
from hyperloom_multibranch.blackboard.rsi.frontier import (
    CANDIDATE_COMPONENT,
    EVIDENCE_COMPONENTS,
    NEAR_BAR_COMPONENT,
    PROBATION_COMPONENT,
    SETTLED_TESTS,
    STALL_BAR_COMPONENT,
    STALLED_COMPONENT,
)
from hyperloom_multibranch.blackboard.rsi.ladder import trials
from hyperloom_multibranch.blackboard.rsi.playbook import (
    FALLBACK_KEY,
    H0,
    PLAYBOOK_KEY,
    Crowned,
    Playbook,
    PlaybookPool,
    replaced,
)
from hyperloom_multibranch.blackboard.rsi.session import NO_PLAYBOOK_PLAYED, STOP_AUDIT_REPORT, WRITTEN
from hyperloom_multibranch.blackboard.tools import CODE_TITLE, SETTINGS_TITLE
from hyperloom_multibranch.scheduling.explore import PLAYBOOK_FALLBACK_REASON, PLAYBOOK_OPEN_CUT_REASON
from hyperloom_multibranch.scheduling.judge import GRACE_KEY, HoldReason
from hyperloom_multibranch.scheduling.judge_prompt import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    PLAYBOOK_JUDGE_PROMPT,
    PLAYBOOK_JUDGE_PROMPT_VERSION,
)
from hyperloom_multibranch.scheduling.lifecycle import JUDGE_PROMPT_KEY
from hyperloom_multibranch.scheduling.manifest import read_manifest
from hyperloom_multibranch.scheduling.settings import PlaybookMode
from hyperloom_multibranch.scheduling.tests.child_sessions import child_with_patch
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    MAIN_POINT,
    VIRTUAL_POLL_SECONDS,
    ScriptedCompletion,
    SessionStep,
    tagged,
)
from hyperloom_multibranch.tests.test_reuse_prelude import WORKLOAD_ARGS
from hyperloom_multibranch.tests.test_skeleton import (
    CONC,
    CURVES,
    EXPLORED,
    SETTINGS,
    Stage,
    _decisions,
    _judge_prompts,
    _judge_state,
    _stage,
    child_framework,
)

#: ``child_framework`` is test_skeleton's autouse fixture, imported so that it applies here too: each session's PRELUDE
#: child roots on the framework checkout it was built on, as the root the session commits before it starts does.
__all__ = ["child_framework"]

#: Changes that measure the root's value, so a branch testing them commits nothing.
FLAT = tuple(tagged(f"flat{n}") for n in range(9))
#: A change that gains 0.45%: over half the 0.6% bar the noiseless baseline sets, under the bar, so a near miss.
NEAR = tagged("near")
NEAR_CURVES = {**EXPLORED, frozenset({"near"}): {MAIN_POINT: 1004.5, CONC: 1004.5}}
#: The settings :data:`~hyperloom_multibranch.tests.test_skeleton.SETTINGS` holds, as the command line spells them.
SESSION_ARGS = [
    "--max-hours", "0.5", "--max-branches", "3", "--max-k-per-branch", "2", "--max-parallel-tests", "8",
    "--judge-interval-sec", "600",
]  # fmt: skip
STALLED = "stalled past the bar"
INSISTED = "stop it whatever holds it"


@pytest.fixture(autouse=True)
def no_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    """No archive unless a test names one, and none read-only."""

    monkeypatch.delenv(ARCHIVE_ENV, raising=False)
    monkeypatch.delenv(WRITE_ONLY_ENV, raising=False)


@dataclass
class PlaybookJudge(ScriptedCompletion):
    """A judge that plays the playbook as written: the opening call opens ``opening`` branches at K 2, and every call
    after it asks for ``refill`` more, stops each live specialist branch its stall evidence makes a stop candidate,
    and asks to stop each one of ``insist`` whatever holds it; a test double, not a placeholder."""

    opening: int = 2
    refill: int = 0
    insist: frozenset[str] = frozenset()

    async def __call__(self, system: str, user: str) -> CompletionReply:
        self.calls.append((system, user))
        state = _judge_state(user)
        if state["opening"]:
            return self._answer({"open": self.opening, "open_k": 2, "branches": [], "stop": []})
        stops = [
            {"branch": row["branch"], "reason": STALLED if _candidate(row) else INSISTED}
            for row in state["branches"]
            if row["kind"] == "specialist"
            and "stopped" not in row
            and (_candidate(row) or row["branch"] in self.insist)
        ]
        return self._answer({"open": self.refill, "open_k": 2, "branches": [], "stop": stops})

    def _answer(self, document: dict[str, object]) -> CompletionReply:
        return CompletionReply(json.dumps(document), self.tokens, self.cost_usd)


def _candidate(row: dict[str, Any]) -> bool:
    return bool(row.get("components", {}).get(CANDIDATE_COMPONENT))


def _rounds(stage: Stage) -> list[dict[str, dict[str, Any]]]:
    """Each steering judge call's branch rows, by branch."""

    states = [_judge_state(prompt) for prompt in _judge_prompts(stage)]
    return [{row["branch"]: row for row in state["branches"]} for state in states if not state["opening"]]


def _archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TreeArchive:
    root = tmp_path / "rsi"
    monkeypatch.setenv(ARCHIVE_ENV, str(root))
    return TreeArchive(root)


def _crowned(archive: TreeArchive, playbook: Playbook) -> Playbook:
    """``playbook`` registered beside ``archive`` and crowned its champion."""

    pool = PlaybookPool.beside(archive)
    pool.register(playbook)
    pool.crown(Crowned(playbook.label, 1.0, "crowned for the test"))
    return playbook


def _knowledge(stage: Stage) -> dict[str, dict[str, Any]]:
    return json.loads((stage.layout.reports / "session_breakdown.json").read_text())["close"]["knowledge"]


async def _stops(stage: Stage) -> dict[str, str]:
    return {node.branch: node.stop_reason for node in (await stage.handle(Role.OBSERVER).lineage()).branches}


def _once_on_probation(stage: Stage, changes: Sequence[RecipeDelta]) -> SessionStep:
    """A step that waits until the branch is on probation, then tests ``changes`` on its head, as a specialist goes on
    exploring through its probation."""

    async def step(branch: str) -> None:
        stage.watchers.append(asyncio.ensure_future(_after_probation(stage, branch, changes)))

    return step


async def _after_probation(stage: Stage, branch: str, changes: Sequence[RecipeDelta]) -> None:
    observer = stage.handle(Role.OBSERVER)
    while not any(node.branch == branch and node.probations for node in (await observer.lineage()).branches):
        await stage.time.sleep(VIRTUAL_POLL_SECONDS)
    head = await stage.repo.head(branch)
    for change in changes:
        await stage.toolbox.submit_test(TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, head, change))


def _then_test(stage: Stage, changes: Sequence[RecipeDelta]) -> SessionStep:
    """A step that, once the branch committed, tests ``changes`` on its new head."""

    async def test(branch: str, head: str) -> None:
        for change in changes:
            await stage.toolbox.submit_test(TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, head, change))

    return stage.once_committed(test)


async def test_a_session_with_an_archive_plays_its_champion_stops_a_stalled_branch_and_close_keeps_the_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, monkeypatch)
    numbers = replaced(H0, stall_tests=4, grace_tests=3, probation_share=0.0).numbers
    _crowned(archive, Playbook("stall-4", numbers, parent=H0.label))
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = PlaybookJudge()
    stage.scripts = {"b01": (tagged("c1"),), "b02": FLAT[:6]}

    outcome = await stage.run()

    assert outcome.exit_code == 0
    [fingerprint] = await stage.handle(Role.OBSERVER).manifest_fingerprints()
    versions = fingerprint.policy_versions
    assert (versions[PLAYBOOK_KEY], versions[JUDGE_PROMPT_KEY], versions[GRACE_KEY]) == (
        "stall-4",
        PLAYBOOK_JUDGE_PROMPT_VERSION,
        "3",
    )
    assert FALLBACK_KEY not in versions
    assert {system for system, _ in stage.judge.calls} == {PLAYBOOK_JUDGE_PROMPT}
    prompts = _judge_prompts(stage)
    assert all(
        "Scheduling playbook stall-4:" in prompt and "Open areas for a new branch" in prompt for prompt in prompts
    )
    rounds = _rounds(stage)
    [stopping] = [index for index, rows in enumerate(rounds) if _candidate(rows.get("b02", {}))]
    evidence = rounds[stopping]["b02"]["components"]
    assert evidence[STALL_BAR_COMPONENT] == 4.0 and evidence[STALLED_COMPONENT] >= 4.0
    assert not any(_candidate(rows["b01"]) for rows in rounds if "b01" in rows)
    assert (await _stops(stage))["b02"] == f"judge: {STALLED}"
    assert all("components" not in rows["b02"] for rows in rounds[stopping + 1 :])
    knowledge = _knowledge(stage)
    assert (knowledge["playbook_trial"]["status"], knowledge["stop_audit"]["status"]) == (WRITTEN, WRITTEN)
    assert (stage.layout.reports / STOP_AUDIT_REPORT).is_file()
    [trial] = trials(PlaybookPool.beside(archive))
    assert (trial.session_id, trial.played, trial.champion, trial.fell_back) == (
        stage.layout.session_id,
        "stall-4",
        "stall-4",
        False,
    )


async def test_a_branch_drawn_for_probation_runs_its_probation_tests_through_a_stop_the_envelope_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, monkeypatch)
    numbers = replaced(H0, stall_tests=4, grace_tests=3, probation_share=1.0, probation_tests=4).numbers
    _crowned(archive, Playbook("always-drawn", numbers, parent=H0.label))
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, judge_interval_seconds=300.0)
    stage.judge = PlaybookJudge(insist=frozenset({"b02"}))
    stage.scripts = {"b01": (tagged("c1"),), "b02": FLAT[:5]}
    stage.steps = {"b02": _once_on_probation(stage, FLAT[5:9])}

    outcome = await stage.run()

    assert outcome.exit_code == 0
    [b02] = [node for node in (await stage.handle(Role.OBSERVER).lineage()).branches if node.branch == "b02"]
    assert [(item.tests, item.since_commit) for item in b02.probations] == [(4, "")]
    rounds = [rows["b02"] for rows in _rounds(stage) if "b02" in rows and "components" in rows["b02"]]
    drawn = [row for row in rounds if row["components"].get(PROBATION_COMPONENT)]
    assert drawn and all(row["held"] == HoldReason.PROBATION.value and not _candidate(row) for row in drawn)
    judged = [row["detail"] for row in _decisions(stage) if row["kind"] == "judge"]
    assert any(detail["held"].get("b02") == HoldReason.PROBATION.value for detail in judged)
    [stopped] = [row for row in rounds if _candidate(row)]
    assert stopped["components"][STALLED_COMPONENT] >= 8.0
    assert b02.stop_reason == f"judge: {STALLED}"
    [fingerprint] = await stage.handle(Role.OBSERVER).manifest_fingerprints()
    assert FALLBACK_KEY not in fingerprint.policy_versions


async def test_a_challenger_the_command_line_asks_for_plays_until_its_judge_breaks_the_envelope_then_the_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    pool = PlaybookPool.beside(archive)
    pool.register(Playbook("stall-4", replaced(H0, stall_tests=4).numbers, parent=H0.label, hypothesis="earlier stops"))
    settings = cli.settings_from(
        cli.build_parser().parse_args(["run", *WORKLOAD_ARGS, *SESSION_ARGS, "--playbook", "challenger"])
    )
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = settings
    stage.explored = NEAR_CURVES
    stage.judge = PlaybookJudge(insist=frozenset({"b02"}))
    stage.scripts = {"b01": (tagged("c1"),), "b02": (*FLAT[:4], NEAR)}

    outcome = await stage.run()

    assert outcome.exit_code == 0 and settings.playbook is PlaybookMode.CHALLENGER
    breach = f"stop of b02 refused for {HoldReason.NEAR_BAR.value}"
    first, fell_back = await stage.handle(Role.OBSERVER).manifest_fingerprints()
    assert first.policy_versions[PLAYBOOK_KEY] == "stall-4" and FALLBACK_KEY not in first.policy_versions
    assert (fell_back.policy_versions[PLAYBOOK_KEY], fell_back.policy_versions[FALLBACK_KEY]) == ("stall-4", breach)
    assert fell_back.policy_versions[GRACE_KEY] == str(H0.numbers.grace_tests)
    manifest = read_manifest(stage.layout.manifest).policy_versions
    assert (manifest[FALLBACK_KEY], manifest[GRACE_KEY]) == (breach, str(H0.numbers.grace_tests))
    [event] = [row for row in _decisions(stage) if row["reason"] == PLAYBOOK_FALLBACK_REASON]
    assert event["detail"] == {"breach": breach, "playing": H0.label}
    prompts = _judge_prompts(stage)
    assert "Scheduling playbook stall-4, on trial in this session:" in prompts[1]
    assert "Scheduling playbook h0:" in prompts[-1]
    held = [rows["b02"] for rows in _rounds(stage) if "components" in rows.get("b02", {})]
    assert held and all(row["components"][NEAR_BAR_COMPONENT] == 1.0 and not _candidate(row) for row in held)
    assert (await _stops(stage))["b02"].startswith("session_end")
    [trial] = trials(pool)
    assert (trial.played, trial.champion, trial.fell_back) == ("stall-4", H0.label, True)


@pytest.mark.parametrize("write_only", [False, True], ids=["no-archive", "write-only"])
async def test_a_session_that_plays_no_playbook_runs_as_one_without_the_rsi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_only: bool
) -> None:
    if write_only:
        archive = _archive(tmp_path, monkeypatch)
        monkeypatch.setenv(WRITE_ONLY_ENV, "1")
        _crowned(archive, Playbook("stall-4", replaced(H0, stall_tests=4).numbers, parent=H0.label))
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.explored = NEAR_CURVES
    stage.judge = PlaybookJudge(insist=frozenset({"b02"}))
    stage.scripts = {"b01": (tagged("c1"),), "b02": (*FLAT[:4], NEAR)}

    outcome = await stage.run()

    assert outcome.exit_code == 0
    [fingerprint] = await stage.handle(Role.OBSERVER).manifest_fingerprints()
    versions = fingerprint.policy_versions
    assert PLAYBOOK_KEY not in versions
    assert (versions[JUDGE_PROMPT_KEY], versions[GRACE_KEY]) == (JUDGE_PROMPT_VERSION, str(SETTLED_TESTS))
    assert {system for system, _ in stage.judge.calls} == {JUDGE_SYSTEM_PROMPT}
    for prompt in _judge_prompts(stage):
        assert "Scheduling playbook" not in prompt and "Open areas" not in prompt
        assert "How similar past sessions went" not in prompt
    rows = [row for rows in _rounds(stage) for row in rows.values()]
    assert not any(EVIDENCE_COMPONENTS & set(row.get("components", {})) for row in rows)
    assert not any(row.get("held") in (HoldReason.NEAR_BAR.value, HoldReason.PROBATION.value) for row in rows)
    assert (await _stops(stage))["b02"] == f"judge: {INSISTED}"
    assert not any(node.probations for node in (await stage.handle(Role.OBSERVER).lineage()).branches)
    assert not [row for row in _decisions(stage) if row["reason"] == "scheduling playbook"]
    briefs = stage.sessions.briefs.values()
    assert briefs and not any(title in brief for brief in briefs for title in (SETTINGS_TITLE, CODE_TITLE))
    knowledge = _knowledge(stage)
    assert (knowledge["playbook_trial"]["status"], knowledge["playbook_trial"]["reason"]) == (
        SKIPPED,
        NO_PLAYBOOK_PLAYED,
    )
    if write_only:
        assert knowledge["archive"]["status"] == ARCHIVED
        assert [tree.session_id for tree in TreeArchive(tmp_path / "rsi").load()] == [stage.layout.session_id]
    else:
        assert (knowledge["archive"]["status"], knowledge["archive"]["reason"]) == (SKIPPED, NO_ARCHIVE)


async def test_the_playbook_s_grace_holds_a_stalled_branch_against_the_judge_s_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, monkeypatch)
    numbers = replaced(H0, stall_tests=4, grace_tests=8, probation_share=0.0).numbers
    _crowned(archive, Playbook("patient-grace", numbers, parent=H0.label))
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = PlaybookJudge(insist=frozenset({"b02"}))
    stage.scripts = {"b01": (tagged("c1"),), "b02": FLAT[:6]}

    outcome = await stage.run()

    assert outcome.exit_code == 0
    [fingerprint] = await stage.handle(Role.OBSERVER).manifest_fingerprints()
    assert fingerprint.policy_versions[GRACE_KEY] == "8"
    stalled = [rows["b02"] for rows in _rounds(stage) if "components" in rows.get("b02", {})]
    assert stalled and all(row["held"] == HoldReason.GRACE.value and not _candidate(row) for row in stalled)
    assert max(row["components"][STALLED_COMPONENT] for row in stalled) == 6.0
    judged = [row["detail"] for row in _decisions(stage) if row["kind"] == "judge"]
    assert any(detail["held"].get("b02") == HoldReason.GRACE.value for detail in judged)
    assert (await _stops(stage))["b02"].startswith("session_end")


def _point(value: float) -> dict[int, float]:
    return {MAIN_POINT: value, CONC: value}


async def test_a_branch_behind_integration_is_no_stop_candidate_while_its_commits_merge_and_one_once_they_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge's rule that a branch behind integration still committing is not stalling meets the stall rule
    here: the stall evidence decides, and a commit counts as progress once it merged, not once it was made."""

    archive = _archive(tmp_path, monkeypatch)
    numbers = replaced(H0, stall_tests=4, grace_tests=3, probation_share=0.0).numbers
    _crowned(archive, Playbook("stall-4", numbers, parent=H0.label))
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    # d1 merges onto the root and onto c1; e1 gains on its branch and stacks on nothing. A merge test's confirm reads
    # the explored curves, so the merged recipes are in both.
    merges = {frozenset({"d1"}): _point(1010.0), frozenset({"c1", "d1"}): _point(1050.0)}
    stage.explored = {**EXPLORED, **merges, frozenset({"e1"}): _point(1010.0)}
    stage.measured = {**CURVES, **merges}
    stage.judge = PlaybookJudge(opening=3)
    stage.scripts = {"b01": (tagged("c1"),), "b02": (tagged("d1"),), "b03": (tagged("e1"),)}
    stage.steps = {"b02": _then_test(stage, FLAT[:2]), "b03": _then_test(stage, FLAT[2:5])}

    outcome = await stage.run()

    assert outcome.exit_code == 0
    committed = {item.branch: item for item in await stage.handle(Role.OBSERVER).committed()}
    assert committed["b02"].merged_into is not None and committed["b03"].merged_into is None
    rounds = [rows for rows in _rounds(stage) if "components" in rows.get("b02", {})]
    merged = [rows["b02"] for rows in rounds if rows["b02"].get("behind_integration_pct", 0.0) > 0]
    settled = [row for row in merged if not row.get("held")]
    assert merged and not any(_candidate(row) for row in merged)
    assert settled and max(row["components"][STALLED_COMPONENT] for row in settled) == 2.0
    [failed] = [rows["b03"] for rows in rounds if _candidate(rows.get("b03", {}))]
    assert failed.get("behind_integration_pct", 0.0) > 0
    stops = await _stops(stage)
    assert stops["b02"].startswith("session_end") and stops["b03"] == f"judge: {STALLED}"


async def test_with_too_little_of_explore_left_for_a_first_commit_every_refill_the_judge_asks_for_is_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, monkeypatch)
    _crowned(archive, Playbook("open-early", replaced(H0, last_open_minutes=240.0).numbers, parent=H0.label))
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = PlaybookJudge(refill=1)
    stage.scripts = {"b01": (tagged("c1"),), "b02": FLAT[:2]}

    outcome = await stage.run()

    assert outcome.exit_code == 0
    for prompt in _judge_prompts(stage):
        left = _judge_state(prompt)["explore_remaining_seconds"] / 60
        assert f"Time left: {left:.0f} min, short of the 240 min a new branch takes to its first commit" in prompt
    cuts = [row["detail"] for row in _decisions(stage) if row["reason"] == PLAYBOOK_OPEN_CUT_REASON]
    assert cuts and all(detail["too_late"] and (detail["requested"], detail["opened"]) == (1, 0) for detail in cuts)
    assert "b03" not in await _stops(stage)
