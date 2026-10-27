# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling playbook: bounded numbers and a few rules, checked before a trial, pooled beside the archive."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import BranchProgress, FrontierScore, PendingCandidate
from hyperloom_multibranch.notebook.rsi.archive import TreeArchive
from hyperloom_multibranch.notebook.rsi.frontier import (
    NEAR_BAR_COMPONENT,
    PROBATION_COMPONENT,
    STALLED_COMPONENT,
    STRANDED_COMPONENT,
    StopRule,
)
from hyperloom_multibranch.notebook.rsi.playbook import (
    FALLBACK_KEY,
    H0,
    MAX_RULES,
    PLAYBOOK_KEY,
    SAFETY_ENVELOPE,
    Crowned,
    Governs,
    Playbook,
    PlaybookChoice,
    PlaybookConflict,
    PlaybookPlay,
    PlaybookPool,
    PlaybookRule,
    Retired,
    changes,
    envelope_breaches,
    problems,
    replaced,
    session_playbook,
)

LATER_STOP = replaced(H0, stall_tests=8)
LONGER = Playbook("longer-probation", LATER_STOP.numbers, parent=H0.label, hypothesis="later", registered_at=10.0)
ALIKE = PlaybookRule(Governs.STOP, "Of two branches whose changes are alike, stop the one further behind integration.")


def test_h0_is_the_evidence_rules_with_every_number_at_its_default() -> None:
    assert problems(H0) == ()
    assert H0.stop_rule() == StopRule()
    text = H0.render()
    assert text.startswith("Scheduling playbook h0:")
    assert "what the stall odds set (stall_bar)" in text and "25% of the stop candidates" in text
    assert all(f"- {line}" in text for line in SAFETY_ENVELOPE)


def test_the_numbers_and_rules_set_the_stop_rule_and_the_judges_text() -> None:
    ruled = Playbook("ruled", LATER_STOP.numbers, (ALIKE,), parent=H0.label)

    assert ruled.stop_rule() == StopRule(stall_tests=8)
    assert "stalled 8 judged tests past its last commit" in ruled.render()
    assert f"- rule (stop): {ALIKE.text}" in ruled.render()
    assert ruled.render(challenger=True).startswith("Scheduling playbook ruled, on trial in this session:")


def test_a_challenger_session_plays_the_champion_from_its_first_breach_on_and_records_it() -> None:
    play = PlaybookPlay(PlaybookChoice(LONGER, H0, challenger=True))
    champion = PlaybookPlay(PlaybookChoice(H0, H0))
    breach = "stop of b1, held by its probation"

    assert (play.stop_rule(), play.versions()) == (StopRule(stall_tests=8), {PLAYBOOK_KEY: LONGER.label})
    assert "on trial in this session" in play.render()
    assert not play.fall_back(()) and play.fall_back((breach, "stop of b2, held by a near-bar lead"))
    assert not play.fall_back(("stop of b3, held by a stranded lead",))
    assert (play.playbook, play.stop_rule()) == (H0, StopRule())
    assert play.versions() == {PLAYBOOK_KEY: LONGER.label, FALLBACK_KEY: breach}
    assert play.render() == H0.render()
    assert not champion.fall_back((breach,)) and champion.versions() == {PLAYBOOK_KEY: H0.label}


@pytest.mark.parametrize(
    ("playbook", "found"),
    [
        (replaced(H0, stall_tests=12), "stall_tests 12 is outside 4 to 10"),
        (replaced(H0, probation_share=0.9), "probation_share 0.9 is outside 0.125 to 0.5"),
        (Playbook("many", rules=(ALIKE,) * (MAX_RULES + 1)), f"6 rules, {MAX_RULES} at most"),
        (Playbook("empty", rules=(PlaybookRule(Governs.K, " "),)), "rule 1 is empty"),
        (Playbook("named", rules=(PlaybookRule(Governs.STOP, "Stop b07 first."),)), "rule 1 names a branch"),
        (Playbook("model", rules=(PlaybookRule(Governs.OPEN, "On Qwen3-Next open MoE."),)), "rule 1 names a branch"),
        (Playbook("Bad Label"), "'Bad Label' is no playbook label"),
    ],
)
def test_a_playbook_out_of_its_bounds_or_tied_to_one_session_is_refused(playbook: Playbook, found: str) -> None:
    assert any(problem.startswith(found) for problem in problems(playbook))


def test_a_playbook_changes_one_thing_of_its_parent() -> None:
    both = Playbook("both", replaced(LATER_STOP, grace_tests=6).numbers, (ALIKE,))

    assert changes(H0, LONGER) == ("stall_tests",)
    assert changes(H0, both) == ("stall_tests", "grace_tests", "rules")
    assert problems(LONGER, H0) == ()
    assert problems(both, H0) == ("it changes stall_tests, grace_tests, rules of h0; a playbook changes one thing",)
    assert problems(Playbook("same"), H0) == ("it changes nothing of h0; a playbook changes one thing",)


def test_the_pool_keeps_each_playbook_once_and_refuses_another_under_its_label(tmp_path: Path) -> None:
    pool = PlaybookPool.beside(TreeArchive(tmp_path))

    assert pool.register(LONGER) == LONGER
    assert pool.register(Playbook(LONGER.label, LONGER.numbers, registered_at=99.0)) == LONGER
    with pytest.raises(PlaybookConflict):
        pool.register(Playbook(LONGER.label, H0.numbers))
    with pytest.raises(ValueError, match="no label"):
        pool.register(Playbook("h0"))
    assert pool.playbooks() == (LONGER,) and pool.root == tmp_path / "playbooks"


def test_a_session_plays_the_champion_or_in_a_challenger_session_the_oldest_playbook_on_trial(tmp_path: Path) -> None:
    pool = PlaybookPool(tmp_path)
    later = Playbook("later-grace", replaced(H0, grace_tests=7).numbers, parent=H0.label, registered_at=20.0)
    broken = Playbook("broken", replaced(H0, stall_tests=99).numbers, parent=H0.label, registered_at=5.0)
    for playbook in (later, LONGER, broken):
        pool.register(playbook)

    assert session_playbook(None, challenger=True).playbook == H0
    assert session_playbook(pool, challenger=False).playbook == H0
    assert pool.on_trial() == (LONGER, later)
    trial = session_playbook(pool, challenger=True)
    assert (trial.playbook, trial.champion, trial.challenger) == (LONGER, H0, True)
    assert trial.render().startswith("Scheduling playbook longer-probation, on trial")

    pool.crown(Crowned(LONGER.label, 30.0, "won 3 of 3 sessions"))
    pool.retire(Retired(later.label, 31.0, "behind the champion"))
    pool.retire(Retired(later.label, 32.0, "again"))

    assert pool.champion() == LONGER and pool.on_trial() == ()
    assert session_playbook(pool, challenger=True) == session_playbook(pool, challenger=False)
    assert [item.evidence for item in pool.retired()] == ["behind the champion"]


def _row(name: str, *, waiting: bool = False, **evidence: float) -> BranchProgress:
    pending = (PendingCandidate(f"{name}-x", f"{name}-confirm", "c1", "output_tps", 1050.0, 5.0),) if waiting else ()
    return BranchProgress(name, True, "c0", "c1", 6, 1, 0.0, {}, FrontierScore(0.0, evidence), pending=pending)


def test_the_envelope_forbids_the_stops_of_branches_their_stall_evidence_holds() -> None:
    rows = (
        _row("b1", **{STALLED_COMPONENT: 9.0, NEAR_BAR_COMPONENT: 0.0}),
        _row("b2", waiting=True),
        _row("b3", **{NEAR_BAR_COMPONENT: 1.0, STRANDED_COMPONENT: 4.2}),
        _row("b4", **{PROBATION_COMPONENT: 1.0}),
        _row("b5", **{PROBATION_COMPONENT: 1.0}),
    )

    found = envelope_breaches(rows, {"b1", "b2", "b3", "b4"})

    assert found == (
        "stop of b2, held by a candidate waiting for its confirm",
        "stop of b3, held by a near-bar lead and a stranded lead",
        "stop of b4, held by its probation",
    )
    assert envelope_breaches(rows, ()) == ()
