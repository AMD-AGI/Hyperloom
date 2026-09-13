# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase-budget overrun signal."""

from __future__ import annotations

from dataclasses import dataclass

from ..role.prompt_inputs import ReactorContext
from .symptom import Symptom, SymptomSeverity


@dataclass
class PhaseBudgetConfig:
    """Tunables for :func:`evaluate_phase_budget_signals`."""

    warn_used_pct: float = 90.0


def evaluate_phase_budget_signals(
    ctx: ReactorContext,
    *,
    config: PhaseBudgetConfig | None = None,
) -> list[Symptom]:
    """Emit a symptom when the current phase has nearly exhausted its budget."""
    cfg = config or PhaseBudgetConfig()
    snap = ctx.shared_state
    if snap.closing_phase or snap.stop_reason:
        return []
    current_phase = (ctx.phase or "").strip().upper()
    if not current_phase or not ctx.phase_budget:
        return []
    for row in ctx.phase_budget:
        if row.phase != current_phase:
            continue
        if row.cap_sec < 0:
            return []
        if row.used_pct < cfg.warn_used_pct:
            return []
        return [
            Symptom(
                name="phase_budget_nearly_exhausted",
                severity=SymptomSeverity.MEDIUM,
                summary=(
                    f"{current_phase} phase budget {row.used_pct:.0f}% consumed "
                    f"({row.elapsed_sec}s of {row.cap_sec}s cap)"
                ),
                evidence={
                    "phase": current_phase,
                    "elapsed_sec": row.elapsed_sec,
                    "cap_sec": row.cap_sec,
                    "used_pct": row.used_pct,
                },
                source="shared_state",
                suggestion="wind down this phase or escalate a phase change",
            )
        ]
    return []
