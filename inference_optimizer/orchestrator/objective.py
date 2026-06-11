# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Objective abstraction — the goal driving early-stop, pressure scoring, and the Orchestration prompt.

Four implementations (DESIGN §11.2): TargetGain, TargetTput, TargetBaseline,
TimeOnly. `build_objective(env)` (§11.3) takes at most one TARGET_* var.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .shared_state import SharedState


class ObjectiveError(ValueError):
    """Raised by `build_objective` on bad/conflicting inputs."""


# ---------------------------------------------------------------------------
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
    def remaining_gap(self, state: "SharedState") -> float:
        """Compute how far we still need to move to reach the goal.

        Args:
            state (SharedState): Current shared optimization state to evaluate.

        Returns:
            float: Remaining distance to the target; units depend on the objective kind.
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
    def pressure_input(self, state: "SharedState") -> float:
        """Feed to scheduler.pressure() (§12): 0.0 = relaxed, 1.0 = max urgency."""

    @abstractmethod
    def describe(self) -> str:
        """Return a one-line summary of the objective for prompt injection.

        Returns:
            str: Human-readable description of the configured target.
        """


# ---------------------------------------------------------------------------
@dataclass
class TargetGainObjective(Objective):
    """Reach ``target_gain_pct`` % over baseline_tput (progress = cumulative_gain / target, capped at 1.0)."""

    target_gain_pct: float

    def __post_init__(self) -> None:
        """Validate the configured target after dataclass initialization.

        Raises:
            ObjectiveError: If ``target_gain_pct`` is not strictly positive.
        """
        if self.target_gain_pct <= 0:
            raise ObjectiveError(
                f"TargetGainObjective: target_gain_pct must be > 0, "
                f"got {self.target_gain_pct}"
            )

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"gain_pct"``.
        """
        return "gain_pct"

    def progress(self, state: "SharedState") -> float:
        """Compute progress as cumulative gain divided by the target, clamped to [0, 1].

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Fraction of the target gain achieved, in the range 0.0 → 1.0.
        """
        return min(1.0, max(0.0, state.cumulative_gain / self.target_gain_pct))

    def remaining_gap(self, state: "SharedState") -> float:
        """Compute the remaining percentage gain needed to hit the target.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Non-negative percentage points still required to reach the target.
        """
        return max(0.0, self.target_gain_pct - state.cumulative_gain)

    def reached(self, state: "SharedState") -> bool:
        """Report whether the cumulative gain has met or exceeded the target.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            bool: ``True`` once ``cumulative_gain >= target_gain_pct``.
        """
        return state.cumulative_gain >= self.target_gain_pct

    def pressure_input(self, state: "SharedState") -> float:
        """Compute normalized progress toward the gain target.

        Args:
            state: Current shared state.

        Returns:
            Progress in ``[0.0, 1.0]``; stays ``0.0`` until the baseline
            throughput has been measured.
        """
        # Stays 0 until baseline finishes.
        if state.baseline_tput <= 0:
            return 0.0
        return min(1.0, state.cumulative_gain / self.target_gain_pct)

    def describe(self) -> str:
        """Return a one-line summary of the configured gain target.

        Returns:
            str: Description of the form ``"target_gain_pct=<value>"``.
        """
        return f"target_gain_pct={self.target_gain_pct}"


@dataclass
class TargetTputObjective(Objective):
    """Reach an absolute tok/s/GPU number (progress against best-so-far tput, not baseline)."""

    target_tput_per_gpu: float

    def __post_init__(self) -> None:
        """Validate the configured target after dataclass initialization.

        Raises:
            ObjectiveError: If ``target_tput_per_gpu`` is not strictly positive.
        """
        if self.target_tput_per_gpu <= 0:
            raise ObjectiveError(
                f"TargetTputObjective: target_tput_per_gpu must be > 0, "
                f"got {self.target_tput_per_gpu}"
            )

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"tput"``.
        """
        return "tput"

    def _current_tput(self, state: "SharedState") -> float:
        """Resolve the current throughput, preferring the best-so-far result.

        Reads ``state.current_best['tput']`` when it is a positive number,
        otherwise falls back to the baseline throughput.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Current throughput in tok/s/GPU, or 0.0 if none is available.
        """
        cb = state.current_best or {}
        v = cb.get("tput") if isinstance(cb, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        return float(state.baseline_tput or 0.0)

    def progress(self, state: "SharedState") -> float:
        """Compute progress as current throughput divided by the target, capped at 1.0.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Fraction of the target throughput achieved, in the range 0.0 → 1.0.
        """
        cur = self._current_tput(state)
        if cur <= 0:
            return 0.0
        return min(1.0, cur / self.target_tput_per_gpu)

    def remaining_gap(self, state: "SharedState") -> float:
        """Compute the remaining throughput needed to hit the target.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Non-negative tok/s/GPU still required to reach the target.
        """
        return max(0.0, self.target_tput_per_gpu - self._current_tput(state))

    def reached(self, state: "SharedState") -> bool:
        """Report whether the current throughput has met or exceeded the target.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            bool: ``True`` once current throughput ``>= target_tput_per_gpu``.
        """
        return self._current_tput(state) >= self.target_tput_per_gpu

    def pressure_input(self, state: "SharedState") -> float:
        """Compute urgency, equal to current progress toward the target.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Urgency in the range 0.0 → 1.0.
        """
        return self.progress(state)

    def describe(self) -> str:
        """Return a one-line summary of the configured throughput target.

        Returns:
            str: Description of the form ``"target_tput_per_gpu=<value>"``.
        """
        return f"target_tput_per_gpu={self.target_tput_per_gpu}"


@dataclass
class TargetBaselineObjective(Objective):
    """Match (or beat) the throughput recorded in another session's baseline (reads ``output_throughput``)."""

    baseline_dir: str
    _ref_tput: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Load the reference throughput from the baseline directory.

        Recursively searches ``baseline_dir`` for ``benchmark_report.json`` files,
        reads the most recent one (by sorted path), and extracts
        ``throughput.output_throughput`` into ``_ref_tput``.

        Raises:
            ObjectiveError: If the directory is missing, no report is found, or the
                report's ``output_throughput`` is missing or not strictly positive.
        """
        path = Path(self.baseline_dir)
        if not path.exists():
            raise ObjectiveError(
                f"TargetBaselineObjective: baseline_dir not found: {path}"
            )
        candidates = sorted(path.rglob("benchmark_report.json"))
        if not candidates:
            raise ObjectiveError(
                f"TargetBaselineObjective: no benchmark_report.json under {path}"
            )
        with candidates[-1].open(encoding="utf-8") as f:
            ref = json.load(f)
        tput = (ref.get("throughput") or {}).get("output_throughput")
        if not isinstance(tput, (int, float)) or tput <= 0:
            raise ObjectiveError(
                f"TargetBaselineObjective: invalid output_throughput in {candidates[-1]}"
            )
        self._ref_tput = float(tput)

    def kind(self) -> str:
        """Return the objective kind tag.

        Returns:
            str: Always ``"baseline"``.
        """
        return "baseline"

    def _cur(self, state: "SharedState") -> float:
        """Resolve the current throughput, preferring the best-so-far result.

        Reads ``state.current_best['tput']`` when it is a positive number,
        otherwise falls back to the baseline throughput.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Current throughput in tok/s, or 0.0 if none is available.
        """
        cb = state.current_best or {}
        v = cb.get("tput") if isinstance(cb, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        return float(state.baseline_tput or 0.0)

    def progress(self, state: "SharedState") -> float:
        """Compute progress as current throughput divided by the reference, capped at 1.0.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Fraction of the reference throughput achieved, in the range 0.0 → 1.0.
        """
        cur = self._cur(state)
        if self._ref_tput <= 0:
            return 0.0
        return min(1.0, cur / self._ref_tput)

    def remaining_gap(self, state: "SharedState") -> float:
        """Compute the remaining throughput needed to match the reference baseline.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Non-negative throughput still required to reach the reference.
        """
        return max(0.0, self._ref_tput - self._cur(state))

    def reached(self, state: "SharedState") -> bool:
        """Report whether the current throughput matches or beats the reference.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            bool: ``True`` once current throughput ``>= _ref_tput``.
        """
        return self._cur(state) >= self._ref_tput

    def pressure_input(self, state: "SharedState") -> float:
        """Compute urgency, equal to current progress toward the reference.

        Args:
            state (SharedState): Current shared optimization state.

        Returns:
            float: Urgency in the range 0.0 → 1.0.
        """
        return self.progress(state)

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

    def remaining_gap(self, state: "SharedState") -> float:
        """Report the remaining gap, which is unbounded since there is no target.

        Args:
            state (SharedState): Current shared optimization state (unused).

        Returns:
            float: Always positive infinity.
        """
        return float("inf")

    def reached(self, state: "SharedState") -> bool:
        """Report whether the goal is met, which is never for this objective.

        Args:
            state (SharedState): Current shared optimization state (unused).

        Returns:
            bool: Always ``False``.
        """
        return False

    def pressure_input(self, state: "SharedState") -> float:
        """Report urgency, which is always relaxed since there is no target.

        Args:
            state (SharedState): Current shared optimization state (unused).

        Returns:
            float: Always 0.0.
        """
        return 0.0

    def describe(self) -> str:
        """Return a one-line summary indicating no target is configured.

        Returns:
            str: Always ``"time_only (no target)"``.
        """
        return "time_only (no target)"


# ---------------------------------------------------------------------------
def build_objective(env: dict[str, Any]) -> Objective:
    """Factory (DESIGN §11.3): requires MAX_HOURS; at most one of TARGET_GAIN_PCT / TARGET_TPUT_PER_GPU / TARGET_DIR (none → TimeOnly)."""
    if "MAX_HOURS" not in env:
        raise ObjectiveError("build_objective: MAX_HOURS is required")
    try:
        max_hours = float(env["MAX_HOURS"])
    except (TypeError, ValueError) as exc:
        raise ObjectiveError(f"build_objective: MAX_HOURS not a float: {exc}") from exc
    if max_hours <= 0:
        raise ObjectiveError(f"build_objective: MAX_HOURS must be > 0, got {max_hours}")

    targets = [k for k in ("TARGET_GAIN_PCT", "TARGET_TPUT_PER_GPU", "TARGET_DIR")
               if env.get(k) not in (None, "")]
    if len(targets) > 1:
        raise ObjectiveError(
            f"build_objective: at most one TARGET_* allowed, got {targets}"
        )

    if "TARGET_GAIN_PCT" in env and env["TARGET_GAIN_PCT"] not in (None, ""):
        return TargetGainObjective(float(env["TARGET_GAIN_PCT"]))
    if "TARGET_TPUT_PER_GPU" in env and env["TARGET_TPUT_PER_GPU"] not in (None, ""):
        return TargetTputObjective(float(env["TARGET_TPUT_PER_GPU"]))
    if "TARGET_DIR" in env and env["TARGET_DIR"] not in (None, ""):
        return TargetBaselineObjective(str(env["TARGET_DIR"]))
    return TimeOnlyObjective()


__all__ = [
    "Objective",
    "ObjectiveError",
    "TargetBaselineObjective",
    "TargetGainObjective",
    "TargetTputObjective",
    "TimeOnlyObjective",
    "build_objective",
]
