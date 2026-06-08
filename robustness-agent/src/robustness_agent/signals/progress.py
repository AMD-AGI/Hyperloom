# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Progress-stagnation detectors (B2 / B3).

Two stateful rules: ``gain_plateau`` (validated gain flat within
``epsilon_pct`` across ``window_ticks`` while still proposing actions →
suggest ``report``) and ``no_levers_found`` (after
``min_observation_minutes`` the optimization_stack is empty and validated
gain still 0). Both short-circuit in ``closing_phase`` or once a
``stop_reason`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..role.prompt_inputs import ReactorContext, SharedStateSnapshot
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



@dataclass
class ProgressConfig:
    """Tunables for :class:`ProgressDetector`.

    ``gain_plateau`` only fires once the ``gain_window_ticks`` buffer is
    full; ``gain_epsilon_pct`` is the unchanged-gain window.
    ``no_levers_min_minutes`` (default 45) is the elapsed-time floor so
    cold-start alone doesn't look "empty"; multi-node large-model runs
    should override it higher (host passes 60.0 when ``nodes >= 2``).
    """

    gain_window_ticks: int = 6
    gain_epsilon_pct: float = 0.5
    no_levers_min_minutes: float = 45.0
    no_levers_min_ticks: int = 8
    productive_gain_pct: float = 0.5


class ProgressDetector:
    """Stateful per-tick rule for ``gain_plateau`` + ``no_levers_found``.

    Keeps a per-tick rolling history of ``cumulative_gain_validated``;
    an empty history short-circuits on a degraded tick.
    """

    def __init__(
        self,
        config: ProgressConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._config = config or ProgressConfig()
        self._state_view = state_view
        # Disk-backed rolling history; without it neither rule can fire
        # under the subprocess-per-tick transport.
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
            return []
        # Append at most one history slot per Coordinator tick.
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
        # Plateau requires "explored but stalled"; an empty stack is the
        # zero-by-construction case that ``no_levers_found`` owns instead.
        if snap.optimization_stack_size == 0:
            return None
        window = self._gain_history[-cfg.gain_window_ticks:]
        delta = max(window) - min(window)
        if delta > cfg.gain_epsilon_pct:
            return None
        if window[-1] >= cfg.productive_gain_pct:
            hint_tail = (
                "search space looks exhausted; consider proposing "
                "`report` to lock in the validated gain"
            )
        else:
            hint_tail = (
                "validated_gain still 0 after a full window; consider "
                "`escalate_strategy_change(skip_to_close)` or `report` "
                "to wind the run down"
            )
        return Symptom(
            name="gain_plateau",
            severity=SymptomSeverity.MEDIUM,
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
            suggestion=hint_tail,
        )

    def _no_levers_symptom(self, snap: SharedStateSnapshot) -> Symptom | None:
        cfg = self._config
        if snap.optimization_stack_size > 0:
            return None
        if snap.cumulative_gain_validated >= cfg.productive_gain_pct:
            # Validated gain with stack count 0 — odd but not our problem.
            return None
        # Don't claim "no lever" while kernel_opt is in flight or a KEEP
        # is waiting to integrate (avoids burning a pending win).
        if snap.kernel_opt_attempts_count > 0:
            return None
        if snap.has_keep_pending_integrate:
            return None
        # Defer until an explore family has actually run; otherwise
        # stack==0 + gain==0 are by-construction during cold start, not diagnostic.
        if not snap.explore_started:
            return None
        if snap.elapsed_minutes < cfg.no_levers_min_minutes:
            return None
        if snap.tick < cfg.no_levers_min_ticks:
            return None
        return Symptom(
            name="no_levers_found",
            severity=SymptomSeverity.MEDIUM,
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
                "kernel_opt_attempts_count": snap.kernel_opt_attempts_count,
                "has_keep_pending_integrate": snap.has_keep_pending_integrate,
                "min_observation_minutes": cfg.no_levers_min_minutes,
                "min_observation_ticks": cfg.no_levers_min_ticks,
            },
            subject={},
            source="local",
            suggestion=(
                "consider `escalate_strategy_change(skip_to_close)` or "
                "`report` to surface the attempted candidates; "
                "`decision_trace.json` already records what was tried"
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
