# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Objective abstraction — the goal driving early-stop and the Orchestration prompt header."""

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
        """Return the short tag identifying this objective type."""

    @abstractmethod
    def progress(self, state: "SharedState") -> float:
        """Compute fractional progress toward the goal."""

    @abstractmethod
    def reached(self, state: "SharedState") -> bool:
        """Report whether the goal has been met."""

    @abstractmethod
    def describe(self) -> str:
        """Return a one-line summary of the objective for prompt injection."""

    @abstractmethod
    def gap_pct(self, state: "SharedState") -> float:
        """Compute the percent improvement still required to reach the goal."""


@dataclass
class _RatioObjective(Objective):
    """Objective scored as ``current / target`` (clamped to [0, 1]); reached when ``current >= target``."""

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
        """Validate the configured target after dataclass initialization."""
        if self.target_gain_pct <= 0:
            raise ObjectiveError(f"TargetGainObjective: target_gain_pct must be > 0, got {self.target_gain_pct}")

    def kind(self) -> str:
        """Return the objective kind tag."""
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
        """Return a one-line summary of the configured gain target."""
        return f"target_gain_pct={self.target_gain_pct}"


@dataclass
class TargetRooflineObjective(_RatioObjective):
    """Reach ``target_within_pct`` % of the modelled roofline ceiling."""

    target_within_pct: float

    def __post_init__(self) -> None:
        """Validate the configured target after dataclass initialization."""
        if not 0 < self.target_within_pct <= 100:
            raise ObjectiveError(
                f"TargetRooflineObjective: target_within_pct must be in (0, 100], got {self.target_within_pct}"
            )

    def kind(self) -> str:
        """Return the objective kind tag."""
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
        """Return a one-line summary of the configured roofline target."""
        return f"target_within_roofline_pct={self.target_within_pct}"


@dataclass
class AnyObjective(Objective):
    """Met when any member is met."""

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
    """Reach an absolute throughput number (progress against best-so-far tput, not baseline)."""

    target_tput_per_gpu: float

    def __post_init__(self) -> None:
        """Validate the configured target after dataclass initialization."""
        if self.target_tput_per_gpu <= 0:
            raise ObjectiveError(
                f"TargetTputObjective: target_tput_per_gpu must be > 0, got {self.target_tput_per_gpu}"
            )

    def kind(self) -> str:
        """Return the objective kind tag."""
        return "tput"

    def _current(self, state: "SharedState") -> float:
        """Resolve current whole-server throughput (best-so-far, else baseline)."""
        return resolve_grading_anchor_tput(state)

    def _target(self) -> float:
        """Return the configured whole-server throughput target."""
        return self.target_tput_per_gpu

    def describe(self) -> str:
        """Return a one-line summary of the configured throughput target."""
        return f"target_tput_per_gpu={self.target_tput_per_gpu}"


@dataclass
class TargetBaselineObjective(_RatioObjective):
    """Match (or beat) the throughput recorded in another session's baseline (reads ``output_throughput``)."""

    baseline_dir: str
    _ref_tput: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Load the reference throughput from the newest report in the baseline directory."""
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
        """Return the objective kind tag."""
        return "baseline"

    def _current(self, state: "SharedState") -> float:
        """Resolve current throughput (best-so-far, else baseline)."""
        return resolve_grading_anchor_tput(state)

    def _target(self) -> float:
        """Return the reference throughput loaded from the baseline session."""
        return self._ref_tput

    def describe(self) -> str:
        """Return a one-line summary of the configured baseline target."""
        return f"target_baseline_dir={self.baseline_dir} (ref_tput={self._ref_tput:.1f})"


@dataclass
class TimeOnlyObjective(Objective):
    """No target — just spend the budget. Never \"reached\"."""

    def kind(self) -> str:
        """Return the objective kind tag."""
        return "time_only"

    def progress(self, state: "SharedState") -> float:
        """Report progress, which is always zero since there is no target."""
        return 0.0

    def reached(self, state: "SharedState") -> bool:
        """Report whether the goal is met, which is never for this objective."""
        return False

    def describe(self) -> str:
        """Return a one-line summary indicating no target is configured."""
        return "time_only (no target)"

    def gap_pct(self, state: "SharedState") -> float:
        """Report the distance to the goal, which is always zero since there is no target."""
        return 0.0


def build_objective(env: dict[str, Any]) -> Objective:
    """Factory: requires MAX_HOURS; at most one throughput target, plus an optional roofline one."""
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
