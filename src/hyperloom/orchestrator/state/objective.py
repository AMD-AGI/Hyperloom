# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Objective abstraction — the goal driving early-stop, pressure scoring, and the Orchestration prompt.

Four implementations: TargetGain, TargetTput, TargetBaseline,
TimeOnly. `build_objective(env)` takes at most one TARGET_* var.
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


def _resolve_current_tput(state: "SharedState") -> float:
    """Resolve current throughput, preferring ``current_best['tput']`` over baseline (0.0 if none)."""
    cb = state.current_best or {}
    v = cb.get("tput") if isinstance(cb, dict) else None
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return float(state.baseline_tput or 0.0)


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


@dataclass
class TargetGainObjective(_RatioObjective):
    """Reach ``target_gain_pct`` % over baseline_tput (progress = cumulative_gain / target, capped at 1.0)."""

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
        return state.cumulative_gain

    def _target(self) -> float:
        """Return the configured gain-percent target."""
        return self.target_gain_pct

    def describe(self) -> str:
        """Return a one-line summary of the configured gain target.

        Returns:
            str: Description of the form ``"target_gain_pct=<value>"``.
        """
        return f"target_gain_pct={self.target_gain_pct}"


@dataclass
class TargetTputObjective(_RatioObjective):
    """Reach an absolute per-GPU throughput number (progress against best-so-far tput, not baseline).

    The unit is framework-dependent: tok/s/GPU for serving frameworks, img/s
    for scriptable xDiT (surfaced elsewhere as the equivalent e2el_mean_ms).
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
        """Resolve current throughput (best-so-far, else baseline)."""
        return _resolve_current_tput(state)

    def _target(self) -> float:
        """Return the configured per-GPU throughput target."""
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
            raise ObjectiveError(f"TargetBaselineObjective: baseline_dir not found: {path}")
        candidates = sorted(path.rglob("benchmark_report.json"))
        if not candidates:
            raise ObjectiveError(f"TargetBaselineObjective: no benchmark_report.json under {path}")
        with candidates[-1].open(encoding="utf-8") as f:
            ref = json.load(f)
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
        return _resolve_current_tput(state)

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


def build_objective(env: dict[str, Any]) -> Objective:
    """Factory: requires MAX_HOURS; at most one of TARGET_GAIN_PCT / TARGET_TPUT_PER_GPU / TARGET_DIR (none → TimeOnly).

    Args:
        env: Environment mapping; must contain ``MAX_HOURS`` and may contain at
            most one of ``TARGET_GAIN_PCT``, ``TARGET_TPUT_PER_GPU``, or
            ``TARGET_DIR``.

    Returns:
        The objective matching the supplied target, or a ``TimeOnlyObjective``
        when no target is given.

    Raises:
        ObjectiveError: If ``MAX_HOURS`` is missing, non-numeric, or
            non-positive, or if more than one ``TARGET_*`` key is supplied.
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
