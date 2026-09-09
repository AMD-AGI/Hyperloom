# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Objective abstraction — the goal driving early-stop and the Orchestration prompt header.

Four implementations: TargetGain, TargetTput, TargetBaseline,
TimeOnly. `build_objective(env)` takes at most one TARGET_* var.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.io import safe_mtime
from hyperloom.common.jsonio import read_json as _read_json

from .shared_state import resolve_grading_anchor_tput

if TYPE_CHECKING:  # pragma: no cover
    from .shared_state import SharedState


log = logging.getLogger(__name__)


class ObjectiveError(ValueError):
    """Raised by `build_objective` on bad/conflicting inputs."""


@dataclass
class Objective(ABC):
    """Goal optimized against (pure functions of SharedState)."""

    @abstractmethod
    def kind(self) -> str:
        """Return the short tag identifying this objective type.

        Returns:
            str: Stable kind identifier (e.g. ``"gain_pct"`` or ``"time_only"``).
        """

    @abstractmethod
    def progress(self, state: "SharedState") -> float:
        """Compute fractional progress toward the goal.

        Args:
            state (SharedState): Current shared optimization state to evaluate.

        Returns:
            float: Progress in the range 0.0 → 1.0, where 1.0 means the goal is hit.
        """

    @abstractmethod
    def reached(self, state: "SharedState") -> bool:
        """Report whether the goal has been met.

        Args:
            state (SharedState): Current shared optimization state to evaluate.

        Returns:
            bool: ``True`` if the objective is satisfied, otherwise ``False``.
        """

    @abstractmethod
    def describe(self) -> str:
        """Return a one-line summary of the objective for prompt injection.

        Returns:
            str: Human-readable description of the configured target.
        """

    @abstractmethod
    def gap_pct(self, state: "SharedState") -> float:
        """Compute the percent improvement still required to reach the goal.

        Args:
            state (SharedState): Current shared optimization state to evaluate.

        Returns:
            float: Remaining distance in percent; 0.0 once the goal is met.
        """


@dataclass
class _RatioObjective(Objective):
    """Objective scored as ``current / target`` (clamped to [0, 1]); reached when ``current >= target``.

    Subclasses supply the ``_current`` / ``_target`` hooks; ``progress`` and
    ``reached`` are shared. Targets are validated positive by each subclass.
    """

    @abstractmethod
    def _current(self, state: "SharedState") -> float:
        """Return the live metric compared against the target."""

    @abstractmethod
    def _target(self) -> float:
        """Return the positive target the metric is compared against."""

    def progress(self, state: "SharedState") -> float:
        """Fraction of the target reached, clamped to [0, 1] (0.0 when either side is non-positive)."""
        cur = self._current(state)
        target = self._target()
        if cur <= 0 or target <= 0:
            return 0.0
        return min(1.0, cur / target)

    def reached(self, state: "SharedState") -> bool:
        """Report whether the live metric meets or exceeds the target."""
        return self._current(state) >= self._target()

    def gap_pct(self, state: "SharedState") -> float:
        """Return the shortfall as a percent of the live metric (0.0 before the first measurement)."""
        cur = self._current(state)
        if cur <= 0:
            return 0.0
        return max(0.0, (self._target() - cur) / cur * 100.0)


@dataclass
class TargetGainObjective(_RatioObjective):
    """Reach ``target_gain_pct`` % over baseline_tput (progress = cumulative_gain_validated / target, capped at 1.0)."""

    target_gain_pct: float

    def __post_init__(self) -> None:
        """Validate the configured target after dataclass initialization.

        Raises:
            ObjectiveError: If ``target_gain_pct`` is not strictly positive.
        """
        if self.target_gain_pct <= 0:
            raise ObjectiveError(f"TargetGainObjective: target_gain_pct must be > 0, got {self.target_gain_pct}")

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"gain_pct"``.
        """
        return "gain_pct"

    def _current(self, state: "SharedState") -> float:
        """Return the cumulative validated gain percentage."""
        return state.cumulative_gain_validated

    def _target(self) -> float:
        """Return the configured gain-percent target."""
        return self.target_gain_pct

    def gap_pct(self, state: "SharedState") -> float:
        """Return the gain percentage points still missing (both sides are already percentages)."""
        return max(0.0, self._target() - self._current(state))

    def describe(self) -> str:
        """Return a one-line summary of the configured gain target.

        Returns:
            str: Description of the form ``"target_gain_pct=<value>"``.
        """
        return f"target_gain_pct={self.target_gain_pct}"


@dataclass
class TargetRooflineObjective(_RatioObjective):
    """Reach ``target_within_pct`` % of the modelled roofline ceiling.

    The live metric reads 0 until a roofline has been measured, so the
    objective cannot be met on a session that never profiled one.
    """

    target_within_pct: float

    def __post_init__(self) -> None:
        """Validate the configured target after dataclass initialization.

        Raises:
            ObjectiveError: If ``target_within_pct`` is outside ``(0, 100]``.
        """
        if not 0 < self.target_within_pct <= 100:
            raise ObjectiveError(
                f"TargetRooflineObjective: target_within_pct must be in (0, 100], got {self.target_within_pct}"
            )

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"roofline_pct"``.
        """
        return "roofline_pct"

    def _current(self, state: "SharedState") -> float:
        """Return the latest measured share of the roofline ceiling."""
        return float(state.current_within_roofline_pct() or 0.0)

    def _target(self) -> float:
        """Return the configured roofline-percentage target."""
        return self.target_within_pct

    def gap_pct(self, state: "SharedState") -> float:
        """Return the roofline percentage points still missing (both sides are already percentages)."""
        return max(0.0, self._target() - self._current(state))

    def describe(self) -> str:
        """Return a one-line summary of the configured roofline target.

        Returns:
            str: Description of the form ``"target_within_roofline_pct=<value>"``.
        """
        return f"target_within_roofline_pct={self.target_within_pct}"


@dataclass
class AnyObjective(Objective):
    """Met when any member is met.

    ``gap_pct`` is the first member's. Members count percentage points of
    different quantities, so anything that picks between them per call changes
    the axis of a value ``explore`` renders as one throughput gap.
    """

    objectives: list[Objective]

    def kind(self) -> str:
        """Return the members' kinds joined by ``+``."""
        return "+".join(o.kind() for o in self.objectives)

    def progress(self, state: "SharedState") -> float:
        """Return the highest member progress."""
        return max(o.progress(state) for o in self.objectives)

    def reached(self, state: "SharedState") -> bool:
        """Report whether any member is satisfied."""
        return any(o.reached(state) for o in self.objectives)

    def gap_pct(self, state: "SharedState") -> float:
        """Return the first member's remaining distance, so the axis is stable."""
        return self.objectives[0].gap_pct(state)

    def describe(self) -> str:
        """Return the members' descriptions joined by ``or``."""
        return " or ".join(o.describe() for o in self.objectives)


@dataclass
class TargetTputObjective(_RatioObjective):
    """Reach an absolute throughput number (progress against best-so-far tput, not baseline).

    The unit is framework-dependent: tok/s for serving frameworks, img/s for
    scriptable xDiT (surfaced elsewhere as the equivalent e2el_mean_ms). Scope
    is whole-server total, not per-GPU; the ``per_gpu`` field name is a
    misnomer kept for compatibility. Callers wanting a per-GPU target convert
    on the way in.
    """

    target_tput_per_gpu: float

    def __post_init__(self) -> None:
        """Validate the configured target after dataclass initialization.

        Raises:
            ObjectiveError: If ``target_tput_per_gpu`` is not strictly positive.
        """
        if self.target_tput_per_gpu <= 0:
            raise ObjectiveError(
                f"TargetTputObjective: target_tput_per_gpu must be > 0, got {self.target_tput_per_gpu}"
            )

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"tput"``.
        """
        return "tput"

    def _current(self, state: "SharedState") -> float:
        """Resolve current whole-server throughput (best-so-far, else baseline)."""
        return resolve_grading_anchor_tput(state)

    def _target(self) -> float:
        """Return the configured whole-server throughput target."""
        return self.target_tput_per_gpu

    def describe(self) -> str:
        """Return a one-line summary of the configured throughput target.

        Returns:
            str: Description of the form ``"target_tput_per_gpu=<value>"``.
        """
        return f"target_tput_per_gpu={self.target_tput_per_gpu}"


@dataclass
class TargetBaselineObjective(_RatioObjective):
    """Match (or beat) the throughput recorded in another session's baseline (reads ``output_throughput``)."""

    baseline_dir: str
    _ref_tput: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Load the reference throughput from the newest report in the baseline directory.

        Prefers a measured round. A budget-dropped measure round leaves the
        warmup as the only report, which is a usable reference and is reported
        as such rather than refused.

        Raises:
            ObjectiveError: If the directory is missing, no report is found, or the
                report's ``output_throughput`` is missing or not strictly positive.
        """
        path = Path(self.baseline_dir)
        if not path.exists():
            raise ObjectiveError(f"TargetBaselineObjective: baseline_dir not found: {path}")
        reports = list(path.rglob("benchmark_report.json"))
        measured = [p for p in reports if "warmup_round" not in p.parts]
        if reports and not measured:
            log.warning("TargetBaselineObjective: reference throughput comes from a warmup round under %s", path)
        candidates = sorted(measured or reports, key=safe_mtime)
        if not candidates:
            raise ObjectiveError(f"TargetBaselineObjective: no benchmark_report.json under {path}")
        ref = _read_json(candidates[-1], default={}, require_dict=True)
        tput = (ref.get("throughput") or {}).get("output_throughput")
        if not isinstance(tput, (int, float)) or tput <= 0:
            raise ObjectiveError(f"TargetBaselineObjective: invalid output_throughput in {candidates[-1]}")
        self._ref_tput = float(tput)

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"baseline"``.
        """
        return "baseline"

    def _current(self, state: "SharedState") -> float:
        """Resolve current throughput (best-so-far, else baseline)."""
        return resolve_grading_anchor_tput(state)

    def _target(self) -> float:
        """Return the reference throughput loaded from the baseline session."""
        return self._ref_tput

    def describe(self) -> str:
        """Return a one-line summary of the configured baseline target.

        Returns:
            str: Description including the baseline directory and reference throughput.
        """
        return f"target_baseline_dir={self.baseline_dir} (ref_tput={self._ref_tput:.1f})"


@dataclass
class TimeOnlyObjective(Objective):
    """No target — just spend the budget. Never "reached".

    Used when no ``TARGET_*`` env var is supplied; optimization runs until the
    Coordinator's wall-clock budget (``MAX_HOURS``) is exhausted.
    """

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"time_only"``.
        """
        return "time_only"

    def progress(self, state: "SharedState") -> float:
        """Report progress, which is always zero since there is no target.

        Args:
            state (SharedState): Current shared optimization state (unused).

        Returns:
            float: Always 0.0.
        """
        return 0.0

    def reached(self, state: "SharedState") -> bool:
        """Report whether the goal is met, which is never for this objective.

        Args:
            state (SharedState): Current shared optimization state (unused).

        Returns:
            bool: Always ``False``.
        """
        return False

    def describe(self) -> str:
        """Return a one-line summary indicating no target is configured.

        Returns:
            str: Always ``"time_only (no target)"``.
        """
        return "time_only (no target)"

    def gap_pct(self, state: "SharedState") -> float:
        """Report the distance to the goal, which is always zero since there is no target.

        Args:
            state (SharedState): Current shared optimization state (unused).

        Returns:
            float: Always 0.0.
        """
        return 0.0


def build_objective(env: dict[str, Any]) -> Objective:
    """Factory: requires MAX_HOURS; at most one throughput target, plus an optional roofline one.

    ``TARGET_GAIN_PCT`` / ``TARGET_TPUT_PER_GPU`` / ``TARGET_DIR`` measure the
    same axis three ways and stay mutually exclusive.
    ``TARGET_WITHIN_ROOFLINE_PCT`` measures a different one, so it composes with
    whichever of those is set: either being met ends the run.

    Args:
        env: Environment mapping; must contain ``MAX_HOURS``, may contain at
            most one of ``TARGET_GAIN_PCT``, ``TARGET_TPUT_PER_GPU`` or
            ``TARGET_DIR``, and may contain ``TARGET_WITHIN_ROOFLINE_PCT``.

    Returns:
        The objective matching the supplied targets, an :class:`AnyObjective`
        when both sides are named, or a ``TimeOnlyObjective`` when none is.

    Raises:
        ObjectiveError: If ``MAX_HOURS`` is missing, non-numeric, or
            non-positive, or if more than one throughput target is supplied.
    """
    if "MAX_HOURS" not in env:
        raise ObjectiveError("build_objective: MAX_HOURS is required")
    try:
        max_hours = float(env["MAX_HOURS"])
    except (TypeError, ValueError) as exc:
        raise ObjectiveError(f"build_objective: MAX_HOURS not a float: {exc}") from exc
    if max_hours <= 0:
        raise ObjectiveError(f"build_objective: MAX_HOURS must be > 0, got {max_hours}")

    targets = [k for k in ("TARGET_GAIN_PCT", "TARGET_TPUT_PER_GPU", "TARGET_DIR") if env.get(k) not in (None, "")]
    if len(targets) > 1:
        raise ObjectiveError(f"build_objective: at most one TARGET_* allowed, got {targets}")

    primary: Objective | None = None
    if env.get("TARGET_GAIN_PCT") not in (None, ""):
        primary = TargetGainObjective(float(env["TARGET_GAIN_PCT"]))
    elif env.get("TARGET_TPUT_PER_GPU") not in (None, ""):
        primary = TargetTputObjective(float(env["TARGET_TPUT_PER_GPU"]))
    elif env.get("TARGET_DIR") not in (None, ""):
        primary = TargetBaselineObjective(str(env["TARGET_DIR"]))
    if env.get("TARGET_WITHIN_ROOFLINE_PCT") in (None, ""):
        return primary or TimeOnlyObjective()
    roofline = TargetRooflineObjective(float(env["TARGET_WITHIN_ROOFLINE_PCT"]))
    return AnyObjective([primary, roofline]) if primary is not None else roofline


__all__ = [
    "AnyObjective",
    "Objective",
    "ObjectiveError",
    "TargetBaselineObjective",
    "TargetGainObjective",
    "TargetRooflineObjective",
    "TargetTputObjective",
    "TimeOnlyObjective",
    "build_objective",
]
