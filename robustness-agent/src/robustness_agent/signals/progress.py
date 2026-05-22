"""Progress-stagnation detectors (B2 / B3).

Two stateful rules live here:

* ``gain_plateau`` — the validated gain has not moved by more than
  ``epsilon_pct`` across the last ``window_ticks`` ticks despite
  Orchestration still proposing actions. Search space is likely
  exhausted; we suggest Orchestration switch to ``report``.

* ``no_levers_found`` — after ``min_observation_minutes`` of elapsed
  wall-clock time the ``optimization_stack`` is still empty AND
  validated gain is still 0. This is the
  ``remain_issue.md`` #8 ("1h+ session with no attribution trace")
  failure mode: the run wasted time without finding anything worth
  keeping, and the operator gets no diagnostic.

Both detectors short-circuit when the session is in ``closing_phase``
(already winding down) or already has a ``stop_reason`` set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext, SharedStateSnapshot
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



@dataclass
class ProgressConfig:
    """Tunables for :class:`ProgressDetector`.

    ``gain_window_ticks`` is the size of the rolling gain history; we
    only fire ``gain_plateau`` once the buffer is full so an early-tick
    flat-line doesn't trip the rule. ``gain_epsilon_pct`` is the
    absolute %-point window inside which we still call the gain
    "unchanged" (matches the upstream ``min_keep_gain_pct`` convention).
    ``no_levers_min_minutes`` is the elapsed-time floor: below it we
    refuse to call the session "empty" because cold-start alone could
    push past 30-40 min on a 671B FP8 model.
    """

    gain_window_ticks: int = 6
    gain_epsilon_pct: float = 0.5
    no_levers_min_minutes: float = 45.0
    no_levers_min_ticks: int = 8
    productive_gain_pct: float = 0.5


class ProgressDetector:
    """Stateful per-tick rule for ``gain_plateau`` + ``no_levers_found``.

    The reactor constructs one detector instance and calls
    :meth:`evaluate` each tick. We keep a per-tick rolling history of
    ``cumulative_gain_validated`` so the rule is robust against missing
    snapshots on a degraded tick (an empty history short-circuits).
    """

    def __init__(
        self,
        config: ProgressConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._config = config or ProgressConfig()
        self._state_view = state_view
        # Disk-backed rolling history — B2 ``gain_plateau`` requires
        # ``gain_window_ticks`` samples (default 6) and B3
        # ``no_levers_found`` cross-references ``last_tick``. Without
        # persistence neither rule can ever fire under M1 subprocess
        # transport.
        loaded = state_view.load() if state_view is not None else {}
        raw_history = loaded.get("gain_history") or []
        if isinstance(raw_history, list):
            self._gain_history: list[float] = [
                float(v) for v in raw_history
                if isinstance(v, (int, float))
            ]
        else:
            self._gain_history = []
        # Cap on load to keep memory bounded across restarts.
        if len(self._gain_history) > 32:
            self._gain_history = self._gain_history[-32:]
        try:
            self._last_tick: int = int(loaded.get("last_tick", -1))
        except (TypeError, ValueError):
            self._last_tick = -1

    def _persist(self) -> None:
        if self._state_view is None:
            return
        self._state_view.save({
            "gain_history": list(self._gain_history),
            "last_tick": self._last_tick,
        })

    @property
    def gain_history(self) -> list[float]:
        """Visible for tests; production code should not rely on this."""
        return list(self._gain_history)

    def evaluate(
        self, ctx: ReactorContext, data: SourceData,
    ) -> list[Symptom]:
        snap = ctx.shared_state
        if snap.closing_phase or snap.stop_reason:
            # Run is already winding down; we don't pile more signals on.
            return []
        # Append once per Coordinator tick (the same tick can drive the
        # robustness backend multiple times during testing but we still
        # only want one history slot per tick).
        if snap.tick > self._last_tick:
            self._gain_history.append(float(snap.cumulative_gain_validated or 0.0))
            self._last_tick = int(snap.tick)
            if len(self._gain_history) > max(self._config.gain_window_ticks, 32):
                self._gain_history = self._gain_history[-32:]
            self._persist()

        out: list[Symptom] = []
        sym = self._gain_plateau_symptom(snap)
        if sym is not None:
            out.append(sym)
        sym = self._no_levers_symptom(snap)
        if sym is not None:
            out.append(sym)
        return out

    def _gain_plateau_symptom(self, snap: SharedStateSnapshot) -> Symptom | None:
        cfg = self._config
        if len(self._gain_history) < cfg.gain_window_ticks:
            return None
        window = self._gain_history[-cfg.gain_window_ticks:]
        delta = max(window) - min(window)
        if delta > cfg.gain_epsilon_pct:
            return None
        # Plateau is uninteresting if the run has already produced a
        # validated gain that's large enough to ship — we want the
        # ``no_levers_found`` rule to handle the empty-stack case
        # separately.
        if window[-1] >= cfg.productive_gain_pct:
            severity = SymptomSeverity.MEDIUM
            hint_tail = (
                "search space is exhausted; consider proposing `report` "
                "to lock in the validated gain"
            )
        else:
            severity = SymptomSeverity.HIGH
            hint_tail = (
                "validated_gain still 0 after a full window; propose "
                "`report` and end the run"
            )
        return Symptom(
            name="gain_plateau",
            severity=severity,
            summary=(
                f"cumulative_gain_validated flat at "
                f"{window[-1]:.2f}% across {cfg.gain_window_ticks} ticks "
                f"(delta={delta:.2f}%, epsilon={cfg.gain_epsilon_pct:.2f}%)"
            ),
            evidence={
                "window_ticks": cfg.gain_window_ticks,
                "history": [round(v, 3) for v in window],
                "delta_pct": round(delta, 3),
                "epsilon_pct": cfg.gain_epsilon_pct,
                "current_tick": snap.tick,
            },
            subject={},
            source="local",
            suggestion=f"escalate_strategy_change: {hint_tail}",
        )

    def _no_levers_symptom(self, snap: SharedStateSnapshot) -> Symptom | None:
        cfg = self._config
        if snap.optimization_stack_size > 0:
            return None
        if snap.cumulative_gain_validated >= cfg.productive_gain_pct:
            # Validated gain exists even though stack count is 0 — odd
            # but not our problem; let it be.
            return None
        if snap.elapsed_minutes < cfg.no_levers_min_minutes:
            return None
        if snap.tick < cfg.no_levers_min_ticks:
            return None
        return Symptom(
            name="no_levers_found",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"session has run {snap.elapsed_minutes:.0f}min over "
                f"{snap.tick} ticks but optimization_stack is empty and "
                f"validated_gain=0; no lever has been found"
            ),
            evidence={
                "elapsed_minutes": snap.elapsed_minutes,
                "tick": snap.tick,
                "optimization_stack_size": 0,
                "cumulative_gain_validated": snap.cumulative_gain_validated,
                "min_observation_minutes": cfg.no_levers_min_minutes,
                "min_observation_ticks": cfg.no_levers_min_ticks,
            },
            subject={},
            source="local",
            suggestion=(
                "delegate(report) to finalize and surface the attempted "
                "candidates; keep `decision_trace.json` for the operator"
            ),
        )


def evaluate_progress_signals(
    detector: ProgressDetector,
    ctx: ReactorContext,
    data: SourceData,
) -> list[Symptom]:
    """Module-level helper mirroring the other signal rule entry points."""
    return detector.evaluate(ctx, data)


__all__ = [
    "ProgressConfig",
    "ProgressDetector",
    "evaluate_progress_signals",
]
