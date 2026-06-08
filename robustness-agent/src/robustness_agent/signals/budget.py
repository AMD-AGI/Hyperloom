# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Wall-clock budget signals.

Two complementary axes of coverage live here:

**Percentage-based (K2, 2026-05-18):**

* ``budget_strategy_drift`` *(new 2026-05-18)* — early-stage warning at
  ``burn_pct >= strategy_drift_pct`` (default 0.5) with no validated
  gain. Suggests Orchestration shrink scope from ``kernel_opt`` to
  faster actions before crossing the more aggressive thresholds below.
* ``budget_burn_no_gain`` — mid-stage warning at
  ``burn_pct >= warn_pct`` (default 0.70) with no validated gain.
* ``deadline_imminent`` — wind-down trigger at
  ``burn_pct >= imminent_pct`` (default 0.85) with no validated gain.
  Emits ``delegate(report)``.

**Absolute-time-based (H1 / H2 spec, 2026-05-18):**

* ``deadline_warning`` *(new 2026-05-18)* — fires when
  ``remaining_minutes <= deadline_warning_minutes`` (default 30) AND
  the session is not yet in ``closing_phase``. Severity depends on
  validated gain: HIGH when no gain has been validated yet, MEDIUM
  when there is shippable gain (so we still nudge towards finalize
  but do not auto-``delegate(report)``).
* ``deadline_hard_cutoff`` *(new 2026-05-18)* — fires when
  ``remaining_minutes <= deadline_hard_cutoff_minutes`` (default 5).
  Always HIGH, always emits ``delegate(report)``: by this point any
  new explore proposal will likely be cut by the deadline supervisor.
  This mirrors the ``WARNING: < 5 min remaining`` line the Coordinator
  already emits in the orchestration prompt — Robustness now gets the
  same trigger so it can ACT on it.

The check is intentionally minimal. We rely on the Coordinator-rendered
``=== Time budget ===`` block parsed into
:class:`SharedStateSnapshot.elapsed_minutes` /
:attr:`SharedStateSnapshot.remaining_minutes` /
:attr:`SharedStateSnapshot.budget_minutes`. When the prompt does not
carry these fields (legacy host or run without a wall-clock deadline)
all signals stay silent.

``cumulative_gain_validated`` is the canonical "validated gain" measure;
``cumulative_gain`` (un-validated) is intentionally NOT used because the
core scenario we are trying to rescue is: the session shows un-validated
gains that keep regressing in ``validate_stack`` — exactly the
``cumulative_gain > 0 but cumulative_gain_validated == 0`` failure mode
observed in the 2026-05 GPU-leak post-mortem.

The two axes intentionally overlap. For a 6h budget, ``imminent_pct``
fires at 54 min remaining while ``deadline_warning`` fires at 30 min;
the latter is the tighter, time-anchored fallback that catches the
case where the early percentage-based gate was silenced by healthy
gain. For a 24h budget the ordering reverses (85% = 3.6h remaining,
well before 30 min); ``deadline_warning`` then arrives first.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..role.prompt_inputs import ReactorContext, SharedStateSnapshot
from .symptom import Symptom, SymptomSeverity


@dataclass
class BudgetConfig:
    """Tunables for :func:`evaluate_budget_signals`.

    Three percentage-based thresholds form an escalating ladder
    (``strategy_drift_pct < warn_pct < imminent_pct``) plus two
    absolute-time thresholds (``deadline_warning_minutes >=
    deadline_hard_cutoff_minutes``) that catch sessions whose budget
    is so long that percentage gates fire too late. ``min_budget_minutes``
    suppresses every signal on sub-30-min smoke tests where nothing
    Robustness can do helps.
    """

    # Above ``strategy_drift_pct`` budget used AND no validated gain → a
    # medium-severity early hint that the explore strategy is not
    # finding wins. Cheaper actions (``params`` / ``sweep``) should
    # take over from ``kernel_opt`` before we reach the harder gates.
    strategy_drift_pct: float = 0.5
    # Above ``warn_pct`` budget used → medium severity alert.
    warn_pct: float = 0.70
    # Above ``imminent_pct`` budget used AND no validated gain → high
    # severity escalate + ``delegate(report)``.
    imminent_pct: float = 0.85
    # ``deadline_warning_minutes`` is the absolute-time backstop that
    # catches long sessions where the percentage gates are still well
    # below ``imminent_pct``. Default 30 min matches the SKILL.md
    # operator playbook: "If <30 min remain, prefer report".
    deadline_warning_minutes: float = 30.0
    # ``deadline_hard_cutoff_minutes`` is the wall-time emergency cut.
    # Mirrors the Coordinator's orchestration-prompt
    # ``WARNING: < 5 min remaining`` line so Robustness can act on the
    # same trigger that already exists on the Orchestration side.
    deadline_hard_cutoff_minutes: float = 5.0
    # Sessions shorter than this never trigger the signal; nothing the
    # robustness role can do helps a sub-30-min smoke test.
    min_budget_minutes: float = 30.0
    # Validated gain (%) above which we consider the run "productive"
    # and suppress the wind-down for the percentage-based ladder. The
    # absolute-time signals are NOT silenced by gain because a 30-min
    # deadline still matters even if the run has shippable progress —
    # we just downgrade their severity from HIGH to MEDIUM.
    productive_gain_pct: float = 0.5


def evaluate_budget_signals(
    ctx: ReactorContext,
    *,
    config: BudgetConfig | None = None,
) -> list[Symptom]:
    """Evaluate the wall-clock budget ladder and absolute-time deadline rules.

    Computes burn percentage and remaining time from the shared-state snapshot
    and emits the appropriate percentage-based and time-anchored symptoms.
    Stays silent on sub-``min_budget_minutes`` sessions and during the closing
    phase.

    Args:
        ctx (ReactorContext): Reactor context providing the shared-state
            snapshot.
        config (BudgetConfig | None): Tunables; defaults to :class:`BudgetConfig`
            when ``None``.

    Returns:
        list[Symptom]: Any budget/deadline symptoms for this tick, possibly
            empty.
    """
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
        # Already in wind-down; no need to nag.
        return []

    burn_pct = elapsed / budget
    validated = float(snap.cumulative_gain_validated or 0.0)

    # ------------------------------------------------------------------
    # Absolute-time signals — checked FIRST because they bypass the
    # ``productive_gain_pct`` gate (gain healthy or not, the wall is
    # coming). The dedup logic in :mod:`signals.classifier` collapses
    # any overlap with the percentage-based signals below.
    # ------------------------------------------------------------------
    absolute_signals: list[Symptom] = []
    if remaining <= cfg.deadline_hard_cutoff_minutes:
        absolute_signals.append(
            _hard_cutoff_symptom(snap, remaining=remaining, cfg=cfg)
        )
    elif remaining <= cfg.deadline_warning_minutes:
        absolute_signals.append(
            _deadline_warning_symptom(
                snap,
                remaining=remaining,
                validated=validated,
                cfg=cfg,
            )
        )

    # ------------------------------------------------------------------
    # Percentage-based signals — gated by validated gain so they go
    # quiet once the run has something shippable.
    # ------------------------------------------------------------------
    percentage_signal: Symptom | None = None
    if validated < cfg.productive_gain_pct:
        if burn_pct >= cfg.imminent_pct:
            percentage_signal = _imminent_symptom(
                snap, burn_pct=burn_pct, cfg=cfg
            )
        elif burn_pct >= cfg.warn_pct:
            percentage_signal = _burn_no_gain_symptom(
                snap, burn_pct=burn_pct, cfg=cfg
            )
        elif burn_pct >= cfg.strategy_drift_pct:
            percentage_signal = _strategy_drift_symptom(
                snap, burn_pct=burn_pct, cfg=cfg
            )

    out: list[Symptom] = list(absolute_signals)
    if percentage_signal is not None:
        out.append(percentage_signal)
    return out


def _imminent_symptom(
    snap: SharedStateSnapshot, *, burn_pct: float, cfg: BudgetConfig,
) -> Symptom:
    """Build the HIGH ``deadline_imminent`` wind-down symptom.

    Args:
        snap (SharedStateSnapshot): Current shared-state snapshot.
        burn_pct (float): Fraction of the budget already consumed.
        cfg (BudgetConfig): Budget tunables.

    Returns:
        Symptom: A HIGH-severity symptom recommending ``delegate(report)``.
    """
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
        # Session-wide signal; subject empty so dedup key collapses
        # cleanly across ticks (the ladder cooldown keeps it from
        # flooding the inbox).
        subject={},
        source="local",
        suggestion=(
            "delegate(report) to finalize at the last validated gain"
        ),
    )


def _burn_no_gain_symptom(
    snap: SharedStateSnapshot, *, burn_pct: float, cfg: BudgetConfig,
) -> Symptom:
    """Build the MEDIUM ``budget_burn_no_gain`` mid-stage warning symptom.

    Args:
        snap (SharedStateSnapshot): Current shared-state snapshot.
        burn_pct (float): Fraction of the budget already consumed.
        cfg (BudgetConfig): Budget tunables.

    Returns:
        Symptom: A MEDIUM-severity symptom nudging a strategy change.
    """
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
        suggestion=(
            "escalate_strategy_change to switch from exploration to "
            "validate_stack / report"
        ),
    )


def _strategy_drift_symptom(
    snap: SharedStateSnapshot, *, burn_pct: float, cfg: BudgetConfig,
) -> Symptom:
    """H2 early warning: 50% burnt and the run still has nothing to ship.

    Intentionally low-severity (MEDIUM) — Robustness only diagnoses
    here; the ActionLadder emits an ``alert(medium)`` so Orchestration
    can see the early-strategy hint without losing decision authority.

    Args:
        snap (SharedStateSnapshot): Current shared-state snapshot.
        burn_pct (float): Fraction of the budget already consumed.
        cfg (BudgetConfig): Budget tunables.

    Returns:
        Symptom: A MEDIUM-severity ``budget_strategy_drift`` symptom.
    """
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
        suggestion=(
            "alert orchestration: consider shrinking scope to "
            "params/sweep before crossing the warn threshold"
        ),
    )


def _deadline_warning_symptom(
    snap: SharedStateSnapshot,
    *,
    remaining: float,
    validated: float,
    cfg: BudgetConfig,
) -> Symptom:
    """H1 absolute-time warning: less than 30 min remain.

    Severity depends on whether the run has validated gain. With gain
    we only nudge (MEDIUM); without gain we treat the time crunch as
    HIGH because the run is now both empty AND about to be cut.

    Args:
        snap (SharedStateSnapshot): Current shared-state snapshot.
        remaining (float): Minutes left before the deadline.
        validated (float): Validated cumulative gain percentage.
        cfg (BudgetConfig): Budget tunables.

    Returns:
        Symptom: A ``deadline_warning`` symptom, HIGH when there is no validated
            gain and MEDIUM otherwise.
    """
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
        summary=(
            f"only {remaining:.1f}min remain (<= "
            f"{cfg.deadline_warning_minutes:.0f}min); {tail}"
        ),
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
            "delegate(report) to finalize at the last validated gain"
            if severity is SymptomSeverity.HIGH else
            "escalate_strategy_change: do not propose kernel_opt or "
            "new explore; validate_stack and report only"
        ),
    )


def _hard_cutoff_symptom(
    snap: SharedStateSnapshot, *, remaining: float, cfg: BudgetConfig,
) -> Symptom:
    """H1 absolute-time emergency cut: <= 5 min remain, no exceptions.

    Always HIGH, always emits ``delegate(report)`` in the ladder.
    Mirrors the Coordinator orchestration prompt's
    ``WARNING: < 5 min remaining`` warning, just on the Robustness side.

    Args:
        snap (SharedStateSnapshot): Current shared-state snapshot.
        remaining (float): Minutes left before the deadline.
        cfg (BudgetConfig): Budget tunables.

    Returns:
        Symptom: A HIGH-severity ``deadline_hard_cutoff`` symptom.
    """
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
        suggestion=(
            "delegate(report) immediately; any new task started now "
            "will be cut by the deadline supervisor"
        ),
    )


__all__ = [
    "BudgetConfig",
    "evaluate_budget_signals",
]
