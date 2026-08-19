# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX budget profile, search-scope collapse, and the session-level guards.

An AgentX round costs orders of magnitude more wall-clock than a synthetic one,
so budgets sized for the latter reap healthy variants -- and an overtime kill is
terminal (the variant skips the KEEP ladder entirely). These tests pin the
widened budgets, the scope reductions that make the cost affordable, and the two
guards that stop a session from silently measuring something other than what it
reports: the bypass-backend combination, and a resume that changes benchmark
mode or crosses an AgentX measurement epoch.

Every case asserts the synthetic path is untouched.
"""

from __future__ import annotations

import argparse

import pytest

from hyperloom.inference_optimizer.cli import (
    _apply_agentx_budget_profile,
    _preflight_agentx_backend,
)
from hyperloom.inference_optimizer.cli.bootstrap import (
    AGENTX_MEASUREMENT_EPOCH,
    _flag_explicitly_set,
    agentx_state_is_stale,
)


def _budget_args(**over) -> argparse.Namespace:
    base = dict(
        max_hours=2.0,
        explore_overtime_kill_ratio=2.0,
        conc_sweep_timeout_sec=1800,
        conc_sweep_total_budget_sec=9000,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _off(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _on(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")


# --- budget profile -----------------------------------------------------------


def test_budget_profile_is_noop_without_agentx(monkeypatch):
    _off(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert vars(args) == vars(_budget_args())


def test_budget_profile_widens_conc_sweep_defaults_under_agentx(monkeypatch):
    _on(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.conc_sweep_timeout_sec > 1800
    assert args.conc_sweep_total_budget_sec > 9000


def test_budget_profile_leaves_the_kill_ratio_alone(monkeypatch):
    """A duration-based replay compresses runtime spread, it does not widen it.

    The measurement window is fixed, so only warmup scales with how slow a
    config is: a variant with 3x slower warmup still lands at ~1.8x the baseline
    total (measured: 46 min warmup + 60 min window + 5 min setup). The stock
    2.0x guard is not the thing that needed loosening -- the per-variant hard
    cap was, and raising this instead would invert the two.
    """
    _on(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.explore_overtime_kill_ratio == 2.0


def test_hard_cap_stays_above_the_soft_kill_at_agentx_baselines(monkeypatch):
    """The layering `_compute_explore_variant_timeout` documents must hold.

    At the measured ~111 min baseline the stock 4h ceiling clamps the hard cap
    to 240 min while the soft kill sits at 222 min -- barely intact -- and a
    longer baseline inverts it, so the generic timeout fires and the round loses
    its KILLED_OVERTIME diagnosis. The AgentX ceiling keeps the ordering.
    """
    from hyperloom.orchestrator.actions.executors.explore import (
        AGENTX_EXPLORE_TIMEOUT_CEILING_SEC,
        DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC,
        _compute_explore_variant_timeout,
    )

    measured_baseline = 111 * 60  # the E4 round
    for baseline in (measured_baseline, 2 * 60 * 60):
        soft_kill = baseline * 2.0
        stock = _compute_explore_variant_timeout(
            baseline, 2.0, ceiling_sec=DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC
        )
        agentx = _compute_explore_variant_timeout(
            baseline, 2.0, ceiling_sec=AGENTX_EXPLORE_TIMEOUT_CEILING_SEC
        )
        assert agentx > soft_kill, f"layering inverted at baseline={baseline}"
        assert agentx >= stock


def test_budget_profile_never_touches_max_hours(monkeypatch):
    """``--max-hours`` is the operator's contract with the scheduler.

    Silently extending it gets the job killed from outside the process, which
    is far harder to diagnose than simply running out of budget.
    """
    _on(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.max_hours == 2.0


def test_budget_profile_preserves_operator_values(monkeypatch):
    """A value the operator typed is left exactly as typed."""
    _on(monkeypatch)
    args = _budget_args(
        explore_overtime_kill_ratio=1.5,
        conc_sweep_timeout_sec=600,
        conc_sweep_total_budget_sec=1200,
    )
    _apply_agentx_budget_profile(args)
    assert args.explore_overtime_kill_ratio == 1.5
    assert args.conc_sweep_timeout_sec == 600
    assert args.conc_sweep_total_budget_sec == 1200


# --- explicit-flag detection --------------------------------------------------


def test_flag_explicitly_set_detects_both_polarities(monkeypatch):
    args = argparse.Namespace()
    monkeypatch.setattr("sys.argv", ["optimize"])
    assert _flag_explicitly_set(args, "enable_conc_sweep") is False
    monkeypatch.setattr("sys.argv", ["optimize", "--enable-conc-sweep"])
    assert _flag_explicitly_set(args, "enable_conc_sweep") is True
    monkeypatch.setattr("sys.argv", ["optimize", "--no-enable-conc-sweep"])
    assert _flag_explicitly_set(args, "enable_conc_sweep") is True
    monkeypatch.setattr("sys.argv", ["optimize", "--conc-sweep-timeout-sec=60"])
    assert _flag_explicitly_set(args, "enable_conc_sweep") is False


# --- bypass guard -------------------------------------------------------------


def test_bypass_guard_allows_magpie(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_is_inert_without_agentx(monkeypatch):
    _off(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_rejects_the_silent_combination(monkeypatch):
    """AgentX + bypass runs synthetic work and labels it AgentX."""
    _on(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace())
    assert ei.value.code == 2


def test_guard_rejects_agentx_with_a_scriptable_framework(monkeypatch):
    """The other way for the switch to no-op while every gate still fires.

    ``apply_agentx_switch`` returns early for a scriptable framework, but
    ``agentx_enabled()`` is a bare env read -- so eval is disabled, the conc
    sweep is turned off, the grid is collapsed, budgets are widened and the
    state is stamped ``benchmark_mode="agentx"`` over a workload that never
    touched a trace.
    """
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace(framework="xdit"))
    assert ei.value.code == 2


def test_guard_allows_agentx_with_a_serving_framework(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    for fw in ("vllm", "sglang"):
        _preflight_agentx_backend(argparse.Namespace(framework=fw))  # must not raise


def test_scriptable_guard_is_inert_without_agentx(monkeypatch):
    """A scriptable run on its own is perfectly normal."""
    _off(monkeypatch)
    _preflight_agentx_backend(argparse.Namespace(framework="xdit"))  # must not raise


# --- resume staleness ---------------------------------------------------------


class _St:
    def __init__(self, mode="", epoch=0):
        self.benchmark_mode = mode
        self.agentx_epoch = epoch


def test_resume_accepts_matching_agentx_state(monkeypatch):
    _on(monkeypatch)
    assert agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH)) == ""


def test_resume_accepts_matching_synthetic_state(monkeypatch):
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("synthetic", 0)) == ""


def test_resume_rejects_mode_switch(monkeypatch):
    """The KEEP ledger keys on server args alone, so the rows would collide."""
    _on(monkeypatch)
    assert "benchmark_mode" in agentx_state_is_stale(_St("synthetic", 0))
    _off(monkeypatch)
    assert "benchmark_mode" in agentx_state_is_stale(_St("agentx", 1))


def test_resume_rejects_stale_agentx_epoch(monkeypatch):
    """Same knobs, different workload: the old numbers cannot anchor."""
    _on(monkeypatch)
    reason = agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH - 1))
    assert "epoch" in reason


def test_resume_tolerates_sessions_predating_the_field(monkeypatch):
    """An empty mode means "not asserted", not "mismatch"."""
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("", 0)) == ""


# --- submission verdict gate --------------------------------------------------


def _measurement(**over):
    # Serving shape: positive throughput plus at least one completed request.
    base = {"output_throughput": 100.0, "completed_requests": 42}
    base.update(over)
    return base


def _valid(result):
    from hyperloom.orchestrator.actions.executors.benchmark_result import (
        is_valid_measurement,
    )

    return is_valid_measurement(result)


def test_verdict_gate_rejects_a_failed_submission(monkeypatch):
    """A scenario-rejected run is not comparable and must not reach KEEP."""
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=False)) is False


def test_verdict_gate_rejects_an_unknown_verdict(monkeypatch):
    """None means no scenario, or an aiperf too old to stamp one.

    ``map_aiperf`` writes the key unconditionally, so an unknown verdict still
    arrives as a present key. Treating unknown as valid is exactly how an
    incomparable run reaches the leaderboard-comparable set.
    """
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=None)) is False


def test_verdict_gate_accepts_a_valid_submission(monkeypatch):
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=True)) is True


def test_verdict_gate_is_inert_on_the_synthetic_path(monkeypatch):
    """``is_valid_measurement`` is hot for every synthetic measurement too.

    The synthetic harness runs an InferenceX revision this repo re-pins from
    time to time. Were a future upstream to stamp the key into a synthetic
    ``inferencex_result.json``, a presence-only check would silently invalidate
    every synthetic measurement session-wide while throughput still looked
    healthy. Gating on the mode removes that coupling.
    """
    _off(monkeypatch)
    assert _valid(_measurement(submission_valid=False)) is True
    assert _valid(_measurement(submission_valid=None)) is True


def test_verdict_gate_spares_scriptable_runs_under_agentx(monkeypatch):
    """A scriptable framework skips the aiperf switch entirely.

    ``apply_agentx_switch`` returns early for scriptable frameworks, so their
    results legitimately carry no verdict even with AgentX on. Keying on
    absence rather than presence would reap them.
    """
    _on(monkeypatch)
    assert _valid(_measurement()) is True

