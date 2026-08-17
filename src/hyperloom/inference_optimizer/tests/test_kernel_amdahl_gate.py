"""An integrate run that cannot clear its bar is GPU time spent on a known answer.

A kernel worth ``f`` of GPU time, made ``S`` times faster, cannot make the whole
deployment more than ``1/((1-f) + f/S)`` faster. When that ceiling is under the
threshold the integrate run has to beat, the run is arithmetically incapable of
passing -- and it is a full end-to-end serving benchmark.

Skipping it is the rare kind of pruning that cannot cost an optimization, but
only if the bound is applied in the safe direction. So these tests are mostly
about the cases where the gate must decline to fire: missing inputs, unusable
inputs, and anything close enough to the bar that the trace's own error in
``gpu_pct`` could account for the gap.
"""
from __future__ import annotations

import pytest

from hyperloom.orchestrator.kernel._kernel_decisions import (
    AMDAHL_GPU_PCT_MARGIN,
    _amdahl_gate_enabled,
    amdahl_e2e_ceiling_pct,
    integrate_cannot_pass,
)


# ── the arithmetic ────────────────────────────────────────────────────────────

def test_ceiling_matches_amdahl():
    """10% of GPU time made 2x faster caps the whole thing at 1/(0.9+0.05)."""
    got = amdahl_e2e_ceiling_pct(10.0, 2.0)
    assert got == pytest.approx((1.0 / 0.95 - 1.0) * 100.0, rel=1e-9)


def test_a_kernel_that_is_everything_inherits_its_own_speedup():
    assert amdahl_e2e_ceiling_pct(100.0, 1.5) == pytest.approx(50.0, rel=1e-9)


def test_no_speedup_is_no_gain():
    assert amdahl_e2e_ceiling_pct(30.0, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_ceiling_rises_with_share_and_with_speedup():
    assert amdahl_e2e_ceiling_pct(20.0, 1.5) > amdahl_e2e_ceiling_pct(10.0, 1.5)
    assert amdahl_e2e_ceiling_pct(10.0, 2.0) > amdahl_e2e_ceiling_pct(10.0, 1.5)


@pytest.mark.parametrize("pct,spd", [
    (0.0, 1.5), (-5.0, 1.5), (101.0, 1.5),
    (10.0, 0.0), (10.0, -1.0),
    (None, 1.5), (10.0, None), ("x", 1.5), (10.0, "x"),
])
def test_unusable_inputs_give_no_ceiling(pct, spd):
    assert amdahl_e2e_ceiling_pct(pct, spd) is None


# ── the gate ──────────────────────────────────────────────────────────────────

def test_the_documented_dead_zone_is_caught():
    """The minimum-share, minimum-speedup kernel cannot clear a 1% bar.

    A kernel at the 10% dispatch floor that just clears the 1.10x micro bar
    reaches 1/(0.9 + 0.1/1.1) = 0.92% end to end. Hyperloom would run a full
    serving benchmark to find that out.
    """
    assert amdahl_e2e_ceiling_pct(10.0, 1.10) < 1.0


def test_gate_declines_when_the_margin_rescues_it():
    """At 10%/1.10x the raw ceiling fails, but the gate inflates the share first."""
    raw = amdahl_e2e_ceiling_pct(10.0, 1.10)
    inflated = amdahl_e2e_ceiling_pct(10.0 * AMDAHL_GPU_PCT_MARGIN, 1.10)
    assert raw < 1.0 < inflated
    assert integrate_cannot_pass(10.0, 1.10, 1.0) is False


def test_gate_fires_only_when_even_the_inflated_share_fails():
    # 2% of GPU time at 1.10x: even at 3% the ceiling stays far under 1%.
    assert integrate_cannot_pass(2.0, 1.10, 1.0) is True


def test_a_big_kernel_always_runs():
    assert integrate_cannot_pass(40.0, 1.5, 1.0) is False


def test_a_large_speedup_on_a_small_kernel_still_runs_when_it_could_pass():
    # 5% at 3x -> inflated to 7.5%, ceiling 5.4%, comfortably over the bar.
    assert integrate_cannot_pass(5.0, 3.0, 1.0) is False


@pytest.mark.parametrize("pct,spd", [(0.0, 1.5), (None, 1.5), (10.0, None), (10.0, 0.0)])
def test_gate_never_fires_on_inputs_it_cannot_read(pct, spd):
    """Unknown inputs must mean 'run the benchmark', never 'skip it'."""
    assert integrate_cannot_pass(pct, spd, 1.0) is False


def test_gate_never_fires_without_a_threshold():
    assert integrate_cannot_pass(1.0, 1.01, 0.0) is False
    assert integrate_cannot_pass(1.0, 1.01, -1.0) is False


def test_a_higher_bar_prunes_more():
    """Raising the threshold can only turn a run off, never on."""
    for thresh in (1.0, 2.0, 5.0):
        prev = integrate_cannot_pass(8.0, 1.2, thresh)
        assert prev in (True, False)
    assert integrate_cannot_pass(8.0, 1.2, 5.0) is True
    assert integrate_cannot_pass(8.0, 1.2, 1.0) is False


def test_gate_is_monotone_in_speedup():
    """A faster kernel must never be pruned when a slower one was kept."""
    assert integrate_cannot_pass(6.0, 1.05, 1.0) is True
    assert integrate_cannot_pass(6.0, 1.60, 1.0) is False


# ── the switch ────────────────────────────────────────────────────────────────

def test_gate_is_on_by_default(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_KERNEL_AMDAHL_GATE", raising=False)
    assert _amdahl_gate_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF"])
def test_gate_can_be_turned_off(monkeypatch, val):
    monkeypatch.setenv("HYPERLOOM_KERNEL_AMDAHL_GATE", val)
    assert _amdahl_gate_enabled() is False
