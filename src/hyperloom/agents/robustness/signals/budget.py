# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Wall-clock budget signals."""

from __future__ import annotations

from dataclasses import dataclass

from ..role.prompt_inputs import ReactorContext, SharedStateSnapshot
from .symptom import Symptom, SymptomSeverity


@dataclass
class BudgetConfig:
    """Tunables for :func:`evaluate_budget_signals`."""

    # Budget used past this with no validated gain → cheaper actions take over from kernel_opt.
    strategy_drift_pct: float = 0.5
    warn_pct: float = 0.70
    imminent_pct: float = 0.85
    # Absolute-time backstop; 30 min matches the SKILL playbook "If <30 min remain, prefer report".
    deadline_warning_minutes: float = 30.0
    # Wall-time emergency cut, mirrors the Coordinator's "< 5 min remaining".
    deadline_hard_cutoff_minutes: float = 5.0
    # Below this budget every signal is suppressed (sub-30-min smoke test).
    min_budget_minutes: float = 30.0
    # Validated gain (%) above which the percentage ladder is suppressed; absolute-time signals are only downgraded
    # HIGH→MEDIUM, not silenced.
    productive_gain_pct: float = 0.5


def evaluate_budget_signals(
    ctx: ReactorContext,
    *,
    config: BudgetConfig | None = None,
) -> list[Symptom]:
    """Evaluate the wall-clock budget ladder and absolute-time deadline rules."""
    cfg = config or BudgetConfig()
    snap: SharedStateSnapshot = ctx.shared_state
    budget = float(snap.budget_minutes or 0.0)
    elapsed = float(snap.elapsed_minutes or 0.0)
    remaining = float(snap.remaining_minutes or 0.0)
    if budget < cfg.min_budget_minutes:
        return []
    if elapsed <= 0.0:
        return []
    if snap.closing_phase:
        return []

    burn_pct = elapsed / budget
    validated = float(snap.cumulative_gain_validated or 0.0)

    # Absolute-time signals checked first; they bypass the ``productive_gain_pct`` gate.
    absolute_signals: list[Symptom] = []
    if remaining <= cfg.deadline_hard_cutoff_minutes:
        absolute_signals.append(_hard_cutoff_symptom(snap, remaining=remaining, cfg=cfg))
    elif remaining <= cfg.deadline_warning_minutes:
        absolute_signals.append(
            _deadline_warning_symptom(
                snap,
                remaining=remaining,
                validated=validated,
                cfg=cfg,
            )
        )

    # Percentage-based signals — gated by validated gain.
    percentage_signal: Symptom | None = None
    if validated < cfg.productive_gain_pct:
        if burn_pct >= cfg.imminent_pct:
            percentage_signal = _imminent_symptom(snap, burn_pct=burn_pct, cfg=cfg)
        elif burn_pct >= cfg.warn_pct:
            percentage_signal = _burn_no_gain_symptom(snap, burn_pct=burn_pct, cfg=cfg)
        elif burn_pct >= cfg.strategy_drift_pct:
            percentage_signal = _strategy_drift_symptom(snap, burn_pct=burn_pct, cfg=cfg)

    out: list[Symptom] = list(absolute_signals)
    if percentage_signal is not None:
        out.append(percentage_signal)
    return out


def _imminent_symptom(
    snap: SharedStateSnapshot,
    *,
    burn_pct: float,
    cfg: BudgetConfig,
) -> Symptom:
    """Build the HIGH ``deadline_imminent`` wind-down symptom."""
    return Symptom(
        name="deadline_imminent",
        severity=SymptomSeverity.HIGH,
        summary=(
            f"wall-clock budget {burn_pct * 100:.0f}% consumed "
            f"(elapsed={snap.elapsed_minutes:.1f}min / "
            f"budget={snap.budget_minutes:.0f}min) and "
            f"cumulative_gain_validated={snap.cumulative_gain_validated:.2f}% "
            f"— wind the session down before the deadline cuts it"
        ),
        evidence={
            "elapsed_minutes": snap.elapsed_minutes,
            "remaining_minutes": snap.remaining_minutes,
            "budget_minutes": snap.budget_minutes,
            "burn_pct": round(burn_pct, 3),
            "cumulative_gain_validated": snap.cumulative_gain_validated,
            "imminent_pct": cfg.imminent_pct,
            "productive_gain_pct": cfg.productive_gain_pct,
        },
        # Session-wide; empty subject collapses the dedup key across ticks.
        subject={},
        source="local",
        suggestion=("propose report to finalize at the last validated gain"),
    )


def _burn_no_gain_symptom(
    snap: SharedStateSnapshot,
    *,
    burn_pct: float,
    cfg: BudgetConfig,
) -> Symptom:
    """Build the MEDIUM ``budget_burn_no_gain`` mid-stage warning symptom."""
    return Symptom(
        name="budget_burn_no_gain",
        severity=SymptomSeverity.MEDIUM,
        summary=(
            f"wall-clock budget {burn_pct * 100:.0f}% consumed with "
            f"cumulative_gain_validated={snap.cumulative_gain_validated:.2f}% "
            f"— consider escalating strategy before the wind-down"
        ),
        evidence={
            "elapsed_minutes": snap.elapsed_minutes,
            "remaining_minutes": snap.remaining_minutes,
            "budget_minutes": snap.budget_minutes,
            "burn_pct": round(burn_pct, 3),
            "cumulative_gain_validated": snap.cumulative_gain_validated,
            "warn_pct": cfg.warn_pct,
        },
        subject={},
        source="local",
        suggestion=("escalate_strategy_change to wind down exploration and report"),
    )


def _strategy_drift_symptom(
    snap: SharedStateSnapshot,
    *,
    burn_pct: float,
    cfg: BudgetConfig,
) -> Symptom:
    """H2 early-warning symptom: budget half-burnt with nothing to ship."""
    return Symptom(
        name="budget_strategy_drift",
        severity=SymptomSeverity.MEDIUM,
        summary=(
            f"wall-clock budget {burn_pct * 100:.0f}% consumed with "
            f"cumulative_gain_validated={snap.cumulative_gain_validated:.2f}% "
            f"— early strategy hint: cheap actions are not paying off"
        ),
        evidence={
            "elapsed_minutes": snap.elapsed_minutes,
            "remaining_minutes": snap.remaining_minutes,
            "budget_minutes": snap.budget_minutes,
            "burn_pct": round(burn_pct, 3),
            "cumulative_gain_validated": snap.cumulative_gain_validated,
            "strategy_drift_pct": cfg.strategy_drift_pct,
            "warn_pct": cfg.warn_pct,
        },
        subject={},
        source="local",
        suggestion=("alert orchestration: consider shrinking scope to params/sweep before crossing the warn threshold"),
    )


def _deadline_warning_symptom(
    snap: SharedStateSnapshot,
    *,
    remaining: float,
    validated: float,
    cfg: BudgetConfig,
) -> Symptom:
    """H1 absolute-time warning symptom (deadline approaching)."""
    if validated < cfg.productive_gain_pct:
        severity = SymptomSeverity.HIGH
        tail = "validated_gain still 0; wind the session down now"
    else:
        severity = SymptomSeverity.MEDIUM
        tail = (
            "validated_gain present; stop proposing new long-running "
            "explore rounds so the existing wins survive the deadline"
        )
    return Symptom(
        name="deadline_warning",
        severity=severity,
        summary=(f"only {remaining:.1f}min remain (<= {cfg.deadline_warning_minutes:.0f}min); {tail}"),
        evidence={
            "elapsed_minutes": snap.elapsed_minutes,
            "remaining_minutes": remaining,
            "budget_minutes": snap.budget_minutes,
            "deadline_warning_minutes": cfg.deadline_warning_minutes,
            "cumulative_gain_validated": snap.cumulative_gain_validated,
            "productive_gain_pct": cfg.productive_gain_pct,
        },
        subject={},
        source="local",
        suggestion=(
            "propose report to finalize at the last validated gain"
            if severity is SymptomSeverity.HIGH
            else "escalate_strategy_change: wind down; do not propose kernel_opt or new explore"
        ),
    )


def _hard_cutoff_symptom(
    snap: SharedStateSnapshot,
    *,
    remaining: float,
    cfg: BudgetConfig,
) -> Symptom:
    """H1 absolute-time emergency-cutoff symptom (deadline imminent)."""
    return Symptom(
        name="deadline_hard_cutoff",
        severity=SymptomSeverity.HIGH,
        summary=(
            f"hard deadline cutoff: only {remaining:.1f}min remain "
            f"(<= {cfg.deadline_hard_cutoff_minutes:.0f}min); finalize now"
        ),
        evidence={
            "elapsed_minutes": snap.elapsed_minutes,
            "remaining_minutes": remaining,
            "budget_minutes": snap.budget_minutes,
            "deadline_hard_cutoff_minutes": cfg.deadline_hard_cutoff_minutes,
            "cumulative_gain_validated": snap.cumulative_gain_validated,
        },
        subject={},
        source="local",
        suggestion=("propose report immediately; any new task started now will be cut by the deadline supervisor"),
    )


__all__ = [
    "BudgetConfig",
    "evaluate_budget_signals",
]
