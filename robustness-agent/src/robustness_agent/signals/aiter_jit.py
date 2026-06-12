# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Aiter JIT-cache regression detector (A7).

Cross-tick view of aiter's ``jit/`` ``.so`` cache, warning before the next
cold-start hits the 3600s ``hipcc`` timeout. Catches cache-regressed-mid-session
and build-dir-stuck; tracks prior ``so_count`` itself since LocalProbe is stateless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



@dataclass
class AiterJitConfig:
    """Tunables for :class:`AiterJitDetector`."""

    # Below this count the cache is "cold".
    cold_so_count: int = 20
    # current/previous ``so_count`` ratio below which (when also cold) → regressed, fire HIGH.
    regression_ratio: float = 0.8
    # Build-dir stale: count > stale_build_threshold AND unchanged for stale_build_persist_ticks ticks.
    stale_build_threshold: int = 1
    stale_build_persist_ticks: int = 5


class AiterJitDetector:
    """Stateful per-tick rule for aiter JIT cache health."""

    def __init__(
        self,
        config: AiterJitConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the detector and restore cross-tick JIT counters.

        Args:
            config (AiterJitConfig | None): Tunables; defaults to
                :class:`AiterJitConfig` when ``None``.
            state_view (DetectorStateView | None): Disk-backed state view used to
                load/persist ``last_so_count``, ``last_build_count`` and the
                stale-build streak.
        """
        self._config = config or AiterJitConfig()
        self._state_view = state_view
        # Disk-backed cross-tick state; required for the regression check under subprocess-per-tick transport.
        loaded = state_view.load() if state_view is not None else {}
        last_so = loaded.get("last_so_count")
        self._last_so_count: int | None = (
            int(last_so) if isinstance(last_so, (int, float)) else None
        )
        try:
            self._last_build_count: int = max(
                0, int(loaded.get("last_build_count", 0))
            )
        except (TypeError, ValueError):
            self._last_build_count = 0
        try:
            self._stale_build_streak: int = max(
                0, int(loaded.get("stale_build_streak", 0))
            )
        except (TypeError, ValueError):
            self._stale_build_streak = 0

    def _persist(self) -> None:
        """Write the cross-tick JIT counters to the state view, if any."""
        if self._state_view is None:
            return
        self._state_view.save({
            "last_so_count": self._last_so_count,
            "last_build_count": self._last_build_count,
            "stale_build_streak": self._stale_build_streak,
        })

    def evaluate(
        self, ctx: ReactorContext, data: SourceData,
    ) -> list[Symptom]:
        """Evaluate the JIT-cache regression and stuck-build rules for this tick.

        Updates the cross-tick counters and emits symptoms when the cache
        regresses below the cold threshold or the build dir stays stuck.

        Args:
            ctx (ReactorContext): Reactor context for the current tick.
            data (SourceData): Collected source data including
                ``local_aiter_jit``.

        Returns:
            list[Symptom]: Any ``aiter_jit_regressed`` / ``aiter_jit_build_stuck``
                symptoms for this tick, possibly empty.
        """
        info = data.local_aiter_jit
        if not isinstance(info, dict) or not info:
            # No JIT data this tick — keep counters; don't accuse on missing telemetry.
            return []
        so_count = info.get("so_count")
        build_count = info.get("build_count") or 0
        if not isinstance(so_count, int):
            return []
        symptoms: list[Symptom] = []

        # Stale build-dir tracker.
        if (
            isinstance(build_count, int)
            and build_count > self._config.stale_build_threshold
            and build_count == self._last_build_count
        ):
            self._stale_build_streak += 1
        else:
            self._stale_build_streak = 0
        if self._stale_build_streak >= self._config.stale_build_persist_ticks:
            symptoms.append(self._build_stuck_symptom(info))

        # Cache regression: needs a prior baseline AND the new value crossing both relative-drop and absolute-cold thresholds.
        prev = self._last_so_count
        if (
            prev is not None
            and prev > self._config.cold_so_count
            and so_count <= self._config.cold_so_count
            and so_count < prev * self._config.regression_ratio
        ):
            symptoms.append(self._regression_symptom(info, prev=prev))

        self._last_so_count = so_count
        self._last_build_count = int(build_count)
        self._persist()
        return symptoms

    def _regression_symptom(
        self, info: dict[str, Any], *, prev: int,
    ) -> Symptom:
        """Build the ``aiter_jit_regressed`` symptom for a cache that went cold.

        Args:
            info (dict[str, Any]): Current aiter JIT probe sample.
            prev (int): The previous tick's ``so_count`` used as the baseline.

        Returns:
            Symptom: A HIGH-severity symptom warning of an impending cold-start.
        """
        cfg = self._config
        return Symptom(
            name="aiter_jit_regressed",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"aiter jit so_count dropped {prev}→{info['so_count']} "
                f"(<= cold threshold {cfg.cold_so_count}); next baseline "
                f"will spend up to 60 min in hipcc"
            ),
            evidence={
                "previous_so_count": prev,
                "current_so_count": info["so_count"],
                "cold_so_count_threshold": cfg.cold_so_count,
                "regression_ratio": cfg.regression_ratio,
                "jit_dir": info.get("jit_dir"),
            },
            subject={},
            source="local",
            suggestion=(
                "skip the next baseline OR extend "
                "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC; do not "
                "just relaunch (consider escalate_strategy_change)"
            ),
        )

    def _build_stuck_symptom(self, info: dict[str, Any]) -> Symptom:
        """Build the ``aiter_jit_build_stuck`` symptom for a stalled build dir.

        Args:
            info (dict[str, Any]): Current aiter JIT probe sample.

        Returns:
            Symptom: A MEDIUM-severity symptom indicating a likely crashed
                mid-build ``hipcc`` invocation.
        """
        cfg = self._config
        return Symptom(
            name="aiter_jit_build_stuck",
            severity=SymptomSeverity.MEDIUM,
            summary=(
                f"aiter jit build_count={info.get('build_count', 0)} "
                f"unchanged for {self._stale_build_streak} consecutive "
                f"ticks; a prior hipcc invocation likely crashed mid-build"
            ),
            evidence={
                "build_count": info.get("build_count", 0),
                "stale_ticks": self._stale_build_streak,
                "stale_build_threshold": cfg.stale_build_threshold,
                "jit_dir": info.get("jit_dir"),
            },
            subject={},
            source="local",
            suggestion=(
                "observe; if it persists, suggest cleaning "
                "<jit_dir>/build/ manually between runs"
            ),
        )


def evaluate_aiter_jit_signals(
    detector: AiterJitDetector,
    ctx: ReactorContext,
    data: SourceData,
) -> list[Symptom]:
    """Module-level helper mirroring the other signal rule entry points.

    Args:
        detector (AiterJitDetector): The stateful detector owned by the caller.
        ctx (ReactorContext): Reactor context for the current tick.
        data (SourceData): Collected source data.

    Returns:
        list[Symptom]: The detector's symptoms for this tick, possibly empty.
    """
    return detector.evaluate(ctx, data)


__all__ = [
    "AiterJitConfig",
    "AiterJitDetector",
    "evaluate_aiter_jit_signals",
]
