# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Aiter JIT-cache regression detector (A7).

Hyperloom's baseline cold-start cost is dominated by aiter's JIT
compiler producing per-shape ``.so`` artefacts. ``BaselineExecutor``
already logs ``baseline_executor: COLD_START`` vs ``WARM`` based on
the count of ``.so`` files under aiter's ``jit/`` cache root, but it
only acts on the value at the moment the next baseline starts. We add
a cross-tick view here so the robustness reactor can shout BEFORE the
next cold-start ticks the run over the 3600s ``hipcc`` timeout.

Two failure modes the detector catches:

1. **Cache regressed mid-session** — operator (or a stale ``install.sh``
   re-run) emptied ``jit/``; the next baseline / validate_stack will
   silently fall into cold-path.
2. **Build dir stuck** — ``jit/build/`` keeps a non-zero count of
   in-flight artefacts for many consecutive ticks; usually means a
   prior ``hipcc`` invocation crashed mid-build and left stale staging
   files that block re-attempts.

The detector tracks the previous tick's ``so_count`` in its own
counter; the LocalProbe is stateless so we cannot move this into the
sub-probe itself.
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

    # Below this count the detector calls the cache "cold". Hyperloom
    # SKILL pins the warm/cold split at 20.
    cold_so_count: int = 20
    # Relative drop ratio (current / previous) below which we treat the
    # cache as regressed. 0.8 means: if ``so_count`` falls below 80% of
    # the previous tick value AND it is now cold, fire HIGH.
    regression_ratio: float = 0.8
    # Build-dir staling: count > stale_build_threshold AND unchanged for
    # ``stale_build_persist_ticks`` consecutive ticks.
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
        self._config = config or AiterJitConfig()
        self._state_view = state_view
        # Disk-backed cross-tick state: ``last_so_count`` is the
        # baseline used for the regression comparison; ``last_build_count``
        # and ``stale_build_streak`` track the build-dir staling rule.
        # Without persistence the regression check never has a prior
        # value to compare against under the subprocess-per-tick transport.
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
        info = data.local_aiter_jit
        if not isinstance(info, dict) or not info:
            # No JIT data this tick — keep last counters intact, do not
            # accuse the run of regressing on missing telemetry.
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

        # Cache regression check — only fires when we have a previous
        # baseline AND the new value crossed both the relative-drop and
        # absolute-cold thresholds.
        prev = self._last_so_count
        if (
            prev is not None
            and prev > self._config.cold_so_count
            and so_count <= self._config.cold_so_count
            and so_count < prev * self._config.regression_ratio
        ):
            symptoms.append(self._regression_symptom(info, prev=prev))

        # Always update the counters last so the next tick has fresh
        # references even if no symptom fired this tick.
        self._last_so_count = so_count
        self._last_build_count = int(build_count)
        self._persist()
        return symptoms

    def _regression_symptom(
        self, info: dict[str, Any], *, prev: int,
    ) -> Symptom:
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
                "escalate_strategy_change: skip the next baseline OR "
                "extend INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC; do "
                "not just relaunch"
            ),
        )

    def _build_stuck_symptom(self, info: dict[str, Any]) -> Symptom:
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
    """Module-level helper mirroring the other signal rule entry points."""
    return detector.evaluate(ctx, data)


__all__ = [
    "AiterJitConfig",
    "AiterJitDetector",
    "evaluate_aiter_jit_signals",
]
