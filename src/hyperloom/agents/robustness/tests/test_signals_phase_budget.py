# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :func:`evaluate_phase_budget_signals`."""

from __future__ import annotations

from hyperloom.agents.robustness.role.prompt_inputs import (
    PhaseBudgetRow,
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.signals.phase_budget import (
    PhaseBudgetConfig,
    evaluate_phase_budget_signals,
)


def _ctx(
    phase: str = "EXPLORE",
    rows: list[PhaseBudgetRow] | None = None,
    *,
    closing_phase: bool = False,
    stop_reason: str = "",
) -> ReactorContext:
    snap = SharedStateSnapshot(closing_phase=closing_phase, stop_reason=stop_reason)
    return ReactorContext(
        phase=phase,
        phase_budget=rows or [],
        shared_state=snap,
    )


def _row(phase: str = "EXPLORE", *, elapsed: int = 500, cap: int = 600, used: float = 83.3) -> PhaseBudgetRow:
    return PhaseBudgetRow(phase=phase, elapsed_sec=elapsed, cap_sec=cap, used_pct=used)


def test_fires_when_above_threshold():
    ctx = _ctx(rows=[_row(used=95.0)])
    syms = evaluate_phase_budget_signals(ctx)
    assert len(syms) == 1
    s = syms[0]
    assert s.name == "phase_budget_nearly_exhausted"
    assert s.severity.value == "medium"
    assert "EXPLORE" in s.summary
    assert s.evidence["used_pct"] == 95.0


def test_silent_below_threshold():
    ctx = _ctx(rows=[_row(used=80.0)])
    assert evaluate_phase_budget_signals(ctx) == []


def test_fires_at_exact_threshold():
    ctx = _ctx(rows=[_row(used=90.0)])
    syms = evaluate_phase_budget_signals(ctx, config=PhaseBudgetConfig(warn_used_pct=90.0))
    assert len(syms) == 1


def test_silent_below_custom_threshold():
    ctx = _ctx(rows=[_row(used=89.9)])
    assert evaluate_phase_budget_signals(ctx, config=PhaseBudgetConfig(warn_used_pct=90.0)) == []


def test_silent_for_unlimited_cap():
    row = PhaseBudgetRow(phase="EXPLORE", elapsed_sec=9999, cap_sec=-1, used_pct=0.0)
    ctx = _ctx(rows=[row])
    assert evaluate_phase_budget_signals(ctx) == []


def test_silent_when_closing_phase():
    ctx = _ctx(rows=[_row(used=99.0)], closing_phase=True)
    assert evaluate_phase_budget_signals(ctx) == []


def test_silent_when_stop_reason_set():
    ctx = _ctx(rows=[_row(used=99.0)], stop_reason="sweep_done")
    assert evaluate_phase_budget_signals(ctx) == []


def test_silent_when_no_phase():
    ctx = _ctx(phase="", rows=[_row(used=99.0)])
    assert evaluate_phase_budget_signals(ctx) == []


def test_only_matches_current_phase():
    rows = [
        PhaseBudgetRow(phase="PRELUDE", elapsed_sec=100, cap_sec=200, used_pct=95.0),
        PhaseBudgetRow(phase="EXPLORE", elapsed_sec=50, cap_sec=600, used_pct=8.3),
    ]
    ctx = _ctx(phase="EXPLORE", rows=rows)
    assert evaluate_phase_budget_signals(ctx) == []


def test_evidence_fields_populated():
    ctx = _ctx(rows=[_row(elapsed=540, cap=600, used=90.0)])
    syms = evaluate_phase_budget_signals(ctx)
    ev = syms[0].evidence
    assert ev["elapsed_sec"] == 540
    assert ev["cap_sec"] == 600
    assert ev["phase"] == "EXPLORE"
