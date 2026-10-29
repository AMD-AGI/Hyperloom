# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the loop moves between exploiting and diversifying: stalls, flat marginal-gain windows and empty-diff streaks."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import (
    IterationVerdict,
    LoopStanding,
    StallPolicy,
    PHASE_STALLED,
    _RECENT_RESULT_CACHE,
    BestRecord,
    LoopStateStore,
    RunState,
    apply_iteration,
    apply_supervisor_intervention,
    make_event,
)
from kernelforge.workflows.forge_loop.iteration.iteration_types import WindowGain
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop, IterationResult
from kernelforge.workflows.forge_loop.search.search_policy import (
    MARGINAL_GAIN_WINDOW,
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    NO_CHANGES_ESCALATION_THRESHOLD,
    NO_CHANGES_STREAK_WINDOW,
    SEARCH_MODE_DIVERSIFY,
    SEARCH_MODE_EXPLOIT,
)
from kernelforge.workflows.forge_loop.search.supervisor import SupervisionMonitor
from kernelforge.workflows.forge_loop.iteration import iteration_phases
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _make_loop,
    _no_change_agent,
    _reduction_loop,
    _unused_supervisor,
)


def _empty_diff(iteration):
    """One session that ended without a candidate diff at all."""
    return IterationResult(
        iteration=iteration,
        duration_sec=0.01,
        validation_passed=False,
        validation_summary="NO TRACKED CHANGES: agent produced no candidate diff",
        kept=False,
    )


def test_a_new_search_mode_starts_its_own_empty_diff_streak(tmp_path, monkeypatch):
    """Two empty diffs escalate the search, so the count is per search mode.

    Counting over iterations instead means that once the streak is at the
    threshold every later attempt is escalated away on its first empty diff, on
    one datum -- and the attempt hit hardest is the diversification the previous
    escalation just forced.

    Every outcome here records the same ``plan``, so a reset can only come from
    the mode. That is the pairing this test exists for: the end-to-end case
    proves one mode keeps counting across reworded headlines, and this one proves
    a mode change resets even when the headline does not change.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    headline = "rewrite the reduction with warp shuffles"

    def streak():
        return loop._consecutive_no_changes(loop.state_store.read_events())

    for iteration in (1, 2):
        loop._record_iteration_outcome(
            _empty_diff(iteration),
            plan=headline,
            decision_label="NO_CHANGES",
        )
    exhausted_streak = streak()
    loop.run_state.search_mode = SEARCH_MODE_DIVERSIFY
    loop._record_iteration_outcome(
        _empty_diff(3),
        plan=headline,
        decision_label="NO_CHANGES",
    )
    after_first_attempt = streak()
    loop._record_iteration_outcome(
        _empty_diff(4),
        plan=headline,
        decision_label="NO_CHANGES",
    )

    assert exhausted_streak == NO_CHANGES_ESCALATION_THRESHOLD
    assert after_first_attempt == 1
    assert streak() == NO_CHANGES_ESCALATION_THRESHOLD


def test_warm_start_search_policy_is_exploit_and_persisted(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(
        best=BestRecord(
            iteration=0,
            mean_case_speedup=1.2,
            commit_hash="warm-commit",
            source="warm_start",
        )
    )
    loop.handoff_store = runner_module.HandoffStore(str(workspace))

    decision = loop._update_search_policy(1)
    persisted = loop.state_store.load()

    assert decision.mode == "EXPLOIT"
    assert decision.reason_codes == ("KB_WARM_START_EXPLOIT",)
    assert persisted.search_mode == "EXPLOIT"
    assert persisted.search_reason_codes == ["KB_WARM_START_EXPLOIT"]


def test_completed_diversify_cycle_enters_bounded_exploit_residence(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=3,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(
        search_mode="DIVERSIFY",
        diversification_cycle_completed=True,
        best=BestRecord(
            iteration=0,
            mean_case_speedup=1.2,
            commit_hash="warm-commit",
            source="warm_start",
        ),
    )
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    loop.run_state.stall.unresolved_stall_iters = 1

    exploit = loop._update_search_policy(2)
    residence = loop._update_search_policy(3)

    assert exploit.mode == "EXPLOIT"
    assert exploit.reason_codes == ("DIVERSIFY_PLAN_CREATED",)
    assert exploit.residence_iterations_remaining == 2
    assert residence.mode == "EXPLOIT"
    assert residence.reason_codes == ("MODE_RESIDENCE",)
    assert residence.residence_iterations_remaining == 1


def test_search_policy_uses_run_state_when_handoff_is_unavailable(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=3,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.handoff_store = None
    loop.run_state = RunState(search_mode="DIVERSIFY")

    loop._apply_iteration_planning_state(
        optimization_plan_created=True,
    )
    decision = loop._update_search_policy(2)

    assert decision.reason_codes == ("DIVERSIFY_PLAN_CREATED",)
    assert decision.mode == "EXPLOIT"


def test_unsuccessful_diversify_cycle_stays_in_diversify(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=3,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(search_mode="DIVERSIFY")
    loop.run_state.stall.unresolved_stall_iters = 5

    loop._apply_iteration_planning_state(
        optimization_plan_created=False,
    )
    decision = loop._update_search_policy(4)

    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)
    assert decision.mode == "DIVERSIFY"


def test_a_supervisor_intervention_no_longer_erases_the_stall_it_answers(
    tmp_path,
    monkeypatch,
):
    """The mla_decode sequence: three REVERTs, an intervention, then DIVERSIFY.

    While both mechanisms read one counter, the intervention zeroed it and
    ``_update_search_policy`` read the zero fourteen lines later, so the
    no-improvement route into DIVERSIFY could never fire: four and seven
    interventions in the 2026-08-18 batch produced no mode switch at all.
    Asking for advice and changing search direction are now simultaneous.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.monitor = SupervisionMonitor(supervise_after=3, cooldown=3)

    for iteration in (1, 2, 3):
        apply_iteration(
            loop.run_state,
            IterationVerdict(
                iteration=iteration,
                decision="REVERT_PERF",
                kept=False,
                wall_ms=1.1,
                commit_hash="",
                plan="another pass over the same loop body",
            ),
            LoopStanding(baseline_wall_ms=1.0, best_wall_ms=1.0),
            StallPolicy(stall_threshold=3),
        )
        loop.monitor.record(kept=False)

    should, reason = loop.monitor.should_intervene(4)
    assert should is True
    assert "3 consecutive" in reason

    loop.monitor.mark_intervened(4)
    apply_supervisor_intervention(
        loop.run_state,
        iteration=4,
        stall_threshold=3,
    )

    # The supervisor's own trigger is throttled exactly as before.
    assert loop.monitor.no_improve_streak == 0
    assert loop.run_state.stall.no_improvement_iters == 0
    assert loop.monitor.should_intervene(5) == (False, "")

    decision = loop._update_search_policy(4)

    assert loop.run_state.stall.unresolved_stall_iters == 3
    assert decision.mode == "DIVERSIFY"
    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)
    assert decision.objective_kind == OBJECTIVE_DISCOVER_NEW_MECHANISM
    assert loop.run_state.phase == PHASE_STALLED


def _kept_outcome(iteration, best_after, *, mode="EXPLOIT"):
    """One measured iteration that left the incumbent at ``best_after``."""
    return make_event(
        "iteration_result",
        iteration,
        decision="KEEP",
        search_mode=mode,
        best_after_mean_case_speedup=best_after,
    )


def test_a_flat_window_of_keeps_diversifies_a_campaign_that_never_stalled(
    tmp_path,
    monkeypatch,
):
    """Every iteration improved, so nothing else in the policy would fire.

    The stall streak is 0 and stays 0 for as long as the ladder produces any
    gain at all, which is exactly the campaign that refines one direction to the
    end of its budget.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(
        best=BestRecord(
            iteration=MARGINAL_GAIN_WINDOW + 1,
            mean_case_speedup=1.53,
            commit_hash="kept",
            source="iteration",
        )
    )
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    for offset in range(MARGINAL_GAIN_WINDOW + 1):
        loop.state_store.append_event(_kept_outcome(offset + 1, 1.50 + 0.005 * offset))

    decision = loop._update_search_policy(MARGINAL_GAIN_WINDOW + 2)
    persisted = loop.state_store.load()
    recorded = [event for event in loop.state_store.read_events() if event.get("type") == "search_policy_decision"]

    assert loop.run_state.stall.no_improvement_iters == 0
    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("DIMINISHING_RETURNS",)
    assert persisted.search_reason_codes == ["DIMINISHING_RETURNS"]
    # The ratio the decision was taken on is part of its audit trail: without it
    # the log says a window was flat but not how flat.
    assert recorded[-1]["window_gain_ratio"] == pytest.approx(0.02, abs=1e-9)


def test_a_diversification_starts_the_marginal_gain_window_again():
    """The round that acted on a flat window cannot be inside the next one.

    Otherwise the same flat outcomes are still in reach once mode residence
    expires, and the campaign diversifies again on evidence it already spent.
    """
    older = [_kept_outcome(offset + 1, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW + 1)]
    diversified = _kept_outcome(
        MARGINAL_GAIN_WINDOW + 2,
        1.54,
        mode=SEARCH_MODE_DIVERSIFY,
    )
    after = [_kept_outcome(MARGINAL_GAIN_WINDOW + 3 + offset, 1.54) for offset in range(MARGINAL_GAIN_WINDOW - 1)]

    assert IterationLoop._exploit_window_gain(
        older,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ).ratio == pytest.approx(0.02, abs=1e-9)
    assert IterationLoop._exploit_window_gain(
        [*older, diversified, *after],
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ) == WindowGain(ratio=None, unavailable="short_window")


@pytest.mark.parametrize(
    "failed_decision",
    ["AGENT_ERROR", "API_ERROR", "ORCHESTRATION_ERROR"],
)
def test_a_diversification_that_concluded_nothing_is_still_a_boundary(
    failed_decision,
):
    """A round that failed still separates two directions.

    The outcomes that fired the trigger are on the far side of it. If the failed
    round is skipped before its mode is read, the scan walks back into them and
    the campaign diversifies again on evidence it already spent -- with only
    mode residence left to brake it.
    """
    older = [_kept_outcome(offset + 1, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW + 1)]
    failed = make_event(
        "iteration_result",
        MARGINAL_GAIN_WINDOW + 2,
        decision=failed_decision,
        search_mode=SEARCH_MODE_DIVERSIFY,
    )
    after = [_kept_outcome(MARGINAL_GAIN_WINDOW + 3 + offset, 1.54) for offset in range(MARGINAL_GAIN_WINDOW - 1)]

    assert IterationLoop._exploit_window_gain(
        [*older, failed, *after],
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ) == WindowGain(ratio=None, unavailable="short_window")


def test_an_exploit_outcome_that_concluded_nothing_is_transparent():
    """An infrastructure failure inside one direction is not a boundary.

    Only a mode change is. Otherwise a single gateway outage would keep the
    window from ever filling on a campaign that never left EXPLOIT.
    """
    events = [
        _kept_outcome(1, 1.50),
        make_event(
            "iteration_result",
            2,
            decision="AGENT_ERROR",
            search_mode=SEARCH_MODE_EXPLOIT,
        ),
    ] + [_kept_outcome(offset + 3, 1.505 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW)]

    assert IterationLoop._exploit_window_gain(
        events,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ).ratio == pytest.approx(0.02, abs=1e-9)


def test_a_supervisor_direction_gets_a_window_of_its_own():
    """Outcomes recorded before an intervention cannot judge what it injected."""
    events = [_kept_outcome(offset + 1, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW + 1)]

    assert IterationLoop._exploit_window_gain(
        events,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=2,
    ) == WindowGain(ratio=None, unavailable="short_window")


@pytest.mark.parametrize(
    ("anchor", "reason"),
    [
        (None, "non_numeric_score"),
        (0.0, "non_positive_score"),
        ("1.5", "non_numeric_score"),
        (True, "non_numeric_score"),
        (float("nan"), "non_finite_score"),
    ],
)
def test_a_window_without_a_usable_anchor_names_why(anchor, reason):
    """A ratio needs a score to divide by, and a missing one is not a flat one.

    Each of these would otherwise become a gain: a missing or non-numeric score
    read as zero, a bool read as a speedup of 1.0, and a zero or NaN anchor
    turning the division into an exception or an infinity that compares false
    against every floor. They are different facts and are reported as such.
    """
    events = [_kept_outcome(1, anchor)] + [
        _kept_outcome(offset + 2, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW)
    ]

    assert IterationLoop._exploit_window_gain(
        events,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ) == WindowGain(ratio=None, unavailable=reason)


def test_a_window_gain_is_never_both_a_ratio_and_a_reason():
    """The two fields are exclusive, so neither can be read as the other."""
    with pytest.raises(ValueError, match="either a ratio or a reason"):
        WindowGain(ratio=None, unavailable=None)
    with pytest.raises(ValueError, match="either a ratio or a reason"):
        WindowGain(ratio=0.02, unavailable="short_window")


def test_an_unevaluable_window_says_so_in_the_decision_event(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A trigger that cannot run must not log like one that ran and found gain.

    ``make_event`` drops None fields, so a bare ratio of None leaves the event
    identical whether the window was short, the score series unusable, or the
    ladder healthy. A campaign whose incumbent score is never recorded has this
    trigger disabled for its whole life; that fact has to be legible.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    for offset in range(MARGINAL_GAIN_WINDOW + 1):
        loop.state_store.append_event(_kept_outcome(offset + 1, None))

    decision = loop._update_search_policy(MARGINAL_GAIN_WINDOW + 2)
    recorded = [event for event in loop.state_store.read_events() if event.get("type") == "search_policy_decision"]

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert "window_gain_ratio" not in recorded[-1]
    assert recorded[-1]["window_gain_unavailable"] == "non_numeric_score"
    assert "non_numeric_score" in capsys.readouterr().out


def test_a_short_window_says_so_rather_than_saying_nothing(
    tmp_path,
    monkeypatch,
):
    """The ordinary young-campaign case is still recorded as a named absence."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    loop.state_store.append_event(_kept_outcome(1, 1.50))

    loop._update_search_policy(2)
    recorded = [event for event in loop.state_store.read_events() if event.get("type") == "search_policy_decision"]

    assert recorded[-1]["window_gain_unavailable"] == "short_window"


def test_repeated_empty_diffs_escalate_before_the_generic_stall_threshold(
    tmp_path,
    monkeypatch,
):
    """Consecutive empty diffs must force a new direction on their own.

    A direction the Implementer cannot express as an edit costs a whole session
    per attempt, so waiting for ``supervise_after`` no-improvement iterations
    lets the same fruitless direction be retried at full price.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=3)
    observed_modes = []

    async def stuck_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = "rewrite the reduction with warp shuffles"
        observed_modes.append(loop.run_state.search_mode)
        return "No source change was needed."

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=stuck_agent, supervisor_fn=_unused_supervisor)))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "NO_CHANGES", "NO_CHANGES"]
    assert state.stall.unresolved_stall_iters < loop.ic.supervise_after
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]


def test_single_empty_diff_does_not_escalate(tmp_path, monkeypatch):
    """One session that found nothing to change is not proof of a dead end."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    state = LoopStateStore(str(workspace)).load()

    assert state.search_mode == "EXPLOIT"
    assert state.search_reason_codes == ["CANONICAL_GAIN_AVAILABLE"]


def test_the_empty_diff_streak_survives_a_reworded_plan_headline(
    tmp_path,
    monkeypatch,
):
    """Rewording the same direction must not reset the streak.

    Every session is asked to close with a fresh one-line ``PLAN:`` headline in
    plain prose, so two sessions handed the same direction never word it the
    same way. Counting the streak against that sentence therefore reset it on
    every iteration and the escalation could only ever fire in a test that
    pinned one literal across sessions.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=3)
    headlines = [
        "fuse the two reduction passes",
        "merge both reduction passes into one",
        "collapse the reduction into a single pass",
    ]
    observed_modes = []

    async def reworded_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = headlines[len(loop.results)]
        observed_modes.append(loop.run_state.search_mode)
        return "No source change was needed."

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=reworded_agent, supervisor_fn=_unused_supervisor)))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "NO_CHANGES", "NO_CHANGES"]
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]


def test_api_outage_does_not_break_the_empty_diff_streak(tmp_path, monkeypatch):
    """An outage measured nothing, so it neither extends nor resets the streak.

    Both halves are visible in the mode observed per iteration: the third
    session still explores because the outage did not count as an empty diff,
    and the fourth diversifies because the outage did not discard the first one
    either. Skipping the outage outright is what makes the second half work: it
    ran under EXPLOIT like its neighbours, but reading its mode at all would let
    an outcome that measured nothing speak for the direction.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=4)
    plans = ["fuse the two passes", "", "fuse the two passes", "fuse the two passes"]
    observed_modes = []

    async def flaky_agent(_kernel_path, _history, session_sink):
        attempt = len(loop.results)
        session_sink["plan"] = plans[attempt]
        observed_modes.append(loop.run_state.search_mode)
        if attempt == 1:
            session_sink["end_reason"] = iteration_phases.EXHAUSTED_END_REASON
            return "agent session ended with an API error"
        return "No source change was needed."

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=flaky_agent, supervisor_fn=_unused_supervisor)))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "API_ERROR", "NO_CHANGES", "NO_CHANGES"]
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]


def test_a_crashed_session_does_not_break_the_empty_diff_streak(
    tmp_path,
    monkeypatch,
):
    """A session that died measured nothing either, so it is transparent too.

    AGENT_ERROR is produced on the same empty-diff branch as API_ERROR. It stays
    out of INFRASTRUCTURE_DECISIONS because that set also picks the cumulative
    counter bucket and there is none for an agent error, so it is the streak
    that has to read it as an attempt which never reached the kernel.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=4)
    plan = "fuse the two passes"
    observed_modes = []

    async def crashing_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = plan
        observed_modes.append(loop.run_state.search_mode)
        if len(loop.results) == 1:
            raise RuntimeError("SDK stream died mid-session")
        return "No source change was needed."

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=crashing_agent, supervisor_fn=_unused_supervisor)))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "AGENT_ERROR", "NO_CHANGES", "NO_CHANGES"]
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]
    assert state.cumulative.reverted == 4
    assert state.stall.unresolved_stall_iters == 4


def test_an_outage_run_cannot_push_the_first_empty_diff_out_of_the_window(
    tmp_path,
    monkeypatch,
):
    """The streak window counts outcomes, so outages cannot exhaust it.

    An iteration writes several events, so a window measured in raw log events
    covers only a handful of iterations: five interleaved outages would push the
    first empty diff out of it and silently disable the escalation. The streak
    is rebuilt from that same log rather than carried in a schema field, so what
    each session saw is asserted directly.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=8)
    plan = "fuse the two passes"
    observed_modes = []
    observed_streaks = []

    async def outage_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = plan
        observed_modes.append(loop.run_state.search_mode)
        observed_streaks.append(loop._consecutive_no_changes(loop.state_store.recent_results(NO_CHANGES_STREAK_WINDOW)))
        if 1 <= len(loop.results) <= 5:
            session_sink["end_reason"] = iteration_phases.EXHAUSTED_END_REASON
            return "agent session ended with an API error"
        return "No source change was needed."

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=outage_agent, supervisor_fn=_unused_supervisor)))

    reopened = LoopStateStore(str(workspace))
    live = reopened.load()
    events = reopened.read_events()
    decisions = [event.get("decision") for event in events if event.get("type") == "iteration_result"]
    resumed_streak = loop._consecutive_no_changes(reopened.recent_results(NO_CHANGES_STREAK_WINDOW))

    assert decisions == ["NO_CHANGES"] + ["API_ERROR"] * 5 + ["NO_CHANGES"] * 2
    # Without this the run would not exercise the distinction: a window counted
    # in raw events would still have held every outcome.
    assert len(events) > NO_CHANGES_STREAK_WINDOW
    # The last session is the whole point: it reached two only because the first
    # empty diff was still inside the window after five outages, and because the
    # outages neither reset it nor counted toward it.
    assert observed_streaks == [0, 1, 1, 1, 1, 1, 1, 2]
    assert observed_modes[-1] == "DIVERSIFY"
    assert live.search_reason_codes == ["REPEATED_NO_CHANGES"]
    # Reading the same log back reaches the same verdict for the mode the run
    # now sits in: the escalation moved it to DIVERSIFY, under which only the
    # last iteration's empty diff has been observed.
    assert resumed_streak == 1


def test_the_outcome_window_is_servable_from_the_cache_or_refused(tmp_path):
    """The streak window must be answerable in full, from memory, or fail.

    A short answer is indistinguishable from a short streak, so the cache is
    sized for the window and a wider request is refused instead of truncated.
    """
    store = LoopStateStore(str(tmp_path))
    for iteration in range(1, NO_CHANGES_STREAK_WINDOW + 2):
        store.append_event(make_event("iteration_started", iteration))
        store.append_event(make_event("iteration_result", iteration, decision="NO_CHANGES"))

    window = store.recent_results(NO_CHANGES_STREAK_WINDOW)

    assert NO_CHANGES_STREAK_WINDOW <= _RECENT_RESULT_CACHE
    assert [event["iter"] for event in window] == list(range(2, NO_CHANGES_STREAK_WINDOW + 2))
    assert LoopStateStore(str(tmp_path)).recent_results(NO_CHANGES_STREAK_WINDOW) == window
    with pytest.raises(ValueError, match="exceeds the cached outcome bound"):
        store.recent_results(_RECENT_RESULT_CACHE + 1)
