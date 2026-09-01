# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the AVO self-supervision monitor (loop/supervisor.py).

``SupervisionMonitor`` is pure state logic (no LLM, no I/O): it decides WHEN the
loop should call a supervisor to break a stall. These tests pin that decision
contract — stall threshold, cooldown, unlimited interventions, and
reset-on-improvement. The "is it circling / dead-ended" semantic judgment is NOT
made here (it moved to the LLM supervisor), so there is no cycle-detection test."""

from __future__ import annotations

from kernelforge.loop.supervisor import SupervisionMonitor


def test_monitor_triggers_after_three_stalls_with_no_intervention_cap():
    # Small knobs so the sequence is easy to follow.
    m = SupervisionMonitor(supervise_after=3, cooldown=2)

    # No stall yet -> no intervention.
    assert m.should_intervene(1) == (False, "")

    # Three consecutive no-improvement iterations.
    for _ in range(3):
        m.record(kept=False)
    do_it, reason = m.should_intervene(10)
    assert do_it is True and "no new best" in reason

    # After intervening, the streak resets and the cooldown blocks re-triggering.
    m.mark_intervened(10)
    assert m.intervention_count == 1
    assert m.should_intervene(11) == (False, "")  # 11 - 10 < cooldown(2)

    # Stall again past the cooldown -> second intervention.
    for _ in range(3):
        m.record(kept=False)
    assert m.should_intervene(13)[0] is True
    m.mark_intervened(13)

    # There is no intervention cap: another three-round stall triggers again.
    for _ in range(3):
        m.record(kept=False)
    assert m.should_intervene(20)[0] is True
    m.mark_intervened(20)
    assert m.intervention_count == 3

    # A kept iteration clears the stall streak.
    m2 = SupervisionMonitor(supervise_after=3)
    m2.record(kept=False)
    m2.record(kept=False)
    assert m2.no_improve_streak == 2
    m2.record(kept=True)
    assert m2.no_improve_streak == 0


def test_monitor_default_trigger_is_three_rounds():
    monitor = SupervisionMonitor()
    monitor.record(kept=False)
    monitor.record(kept=False)
    assert monitor.should_intervene(3) == (False, "")
    monitor.record(kept=False)
    assert monitor.should_intervene(4)[0] is True


def test_monitor_does_not_trigger_below_stall_threshold():
    m = SupervisionMonitor(supervise_after=100, cooldown=1)
    m.record(kept=False)
    m.record(kept=False)
    assert m.should_intervene(5) == (False, "")


def test_failed_attempt_anchors_cooldown_without_resetting_stall():
    monitor = SupervisionMonitor(supervise_after=1, cooldown=3)
    monitor.record(kept=False)

    assert monitor.should_intervene(2)[0] is True
    monitor.mark_attempted(2)

    assert monitor.no_improve_streak == 1
    assert monitor.intervention_count == 0
    assert monitor.should_intervene(3) == (False, "")
    assert monitor.should_intervene(4) == (False, "")
    assert monitor.should_intervene(5)[0] is True
