"""Objective abstraction — DESIGN §8.

Four flavours:

    TargetGainObjective(target_gain_pct)
    TargetTputObjective(target_tput)
    TargetBaselineObjective(target_dir)
    TimeOnlyObjective()

The Conductor / scheduler / early-stop / prompt builder all call the same
methods on the abstract base. Adding a new flavour is one new subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


# ---------------------------------------------------------------------------
# Protocol describing what the Objective needs from SharedState. Keeping the
# coupling loose means the unit tests can pass any duck-typed namespace.
# ---------------------------------------------------------------------------
class StateView(Protocol):
    cumulative_gain: float
    baseline_tput: float | None
    current_tput: float | None
    elapsed_minutes: float
    max_minutes: float


# ---------------------------------------------------------------------------
class Objective(ABC):
    @abstractmethod
    def kind(self) -> str: ...
    @abstractmethod
    def progress(self, state: StateView) -> float: ...
    @abstractmethod
    def remaining_gap(self, state: StateView) -> float: ...
    @abstractmethod
    def reached(self, state: StateView) -> bool: ...
    @abstractmethod
    def pressure_input(self, state: StateView) -> float: ...
    @abstractmethod
    def describe(self) -> str: ...


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetGainObjective(Objective):
    target_gain_pct: float

    def kind(self) -> str:
        return "gain_pct"

    def progress(self, state: StateView) -> float:
        if self.target_gain_pct <= 0:
            return 1.0
        return min(1.0, state.cumulative_gain / self.target_gain_pct)

    def remaining_gap(self, state: StateView) -> float:
        return max(0.0, self.target_gain_pct - state.cumulative_gain)

    def reached(self, state: StateView) -> bool:
        return state.cumulative_gain >= self.target_gain_pct

    def pressure_input(self, state: StateView) -> float:
        return max(0.0, 1.0 - self.progress(state))

    def describe(self) -> str:
        return f"target: +{self.target_gain_pct}% gain over baseline"


@dataclass(frozen=True)
class TargetTputObjective(Objective):
    target_tput: float

    def kind(self) -> str:
        return "tput"

    def progress(self, state: StateView) -> float:
        if state.baseline_tput is None or state.current_tput is None:
            return 0.0
        gained = state.current_tput - state.baseline_tput
        needed = self.target_tput - state.baseline_tput
        if needed <= 0:
            return 1.0
        return max(0.0, min(1.0, gained / needed))

    def remaining_gap(self, state: StateView) -> float:
        if state.current_tput is None:
            return self.target_tput
        return max(0.0, self.target_tput - state.current_tput)

    def reached(self, state: StateView) -> bool:
        return (
            state.current_tput is not None
            and state.current_tput >= self.target_tput
        )

    def pressure_input(self, state: StateView) -> float:
        return max(0.0, 1.0 - self.progress(state))

    def describe(self) -> str:
        return f"target: {self.target_tput} tok/s/GPU absolute"


@dataclass(frozen=True)
class TargetBaselineObjective(TargetTputObjective):
    """Like TargetTput but the number is parsed from a baseline directory.

    The dir parser is intentionally a stub — production code will plug in
    the existing sprint baseline parser. The shape of this class is what
    matters at this layer.
    """
    target_dir: str = ""

    @classmethod
    def from_dir(cls, target_dir: str, parsed_tput: float) -> "TargetBaselineObjective":
        return cls(target_tput=parsed_tput, target_dir=target_dir)

    def kind(self) -> str:
        return "baseline"

    def describe(self) -> str:
        return (
            f"target: match baseline dir={self.target_dir!r} "
            f"({self.target_tput} tok/s/GPU)"
        )


@dataclass(frozen=True)
class TimeOnlyObjective(Objective):
    def kind(self) -> str:
        return "time_only"

    def progress(self, state: StateView) -> float:
        if state.max_minutes <= 0:
            return 0.0
        return min(1.0, state.elapsed_minutes / state.max_minutes)

    def remaining_gap(self, state: StateView) -> float:
        return float("inf")

    def reached(self, state: StateView) -> bool:
        return False  # only time can stop us

    def pressure_input(self, state: StateView) -> float:
        return 0.0

    def describe(self) -> str:
        return "time-only mode (no effect target)"


# ---------------------------------------------------------------------------
def build_objective(env: Mapping[str, str]) -> Objective:
    """Factory — DESIGN §8.4."""
    if "MODEL_PATH" not in env:
        raise ValueError("MODEL_PATH is required")
    if "MAX_HOURS" not in env:
        raise ValueError("MAX_HOURS is required")
    try:
        max_hours = float(env["MAX_HOURS"])
    except (TypeError, ValueError) as exc:
        raise ValueError("MAX_HOURS must be numeric") from exc
    if max_hours <= 0:
        raise ValueError("MAX_HOURS must be > 0")

    target_keys = ("TARGET_GAIN_PCT", "TARGET_TPUT_PER_GPU", "TARGET_DIR")
    present = [k for k in target_keys if env.get(k) not in (None, "")]
    if len(present) > 1:
        raise ValueError(
            f"At most one of {target_keys!r} may be set; got {present!r}"
        )

    if env.get("TARGET_GAIN_PCT"):
        return TargetGainObjective(float(env["TARGET_GAIN_PCT"]))
    if env.get("TARGET_TPUT_PER_GPU"):
        return TargetTputObjective(float(env["TARGET_TPUT_PER_GPU"]))
    if env.get("TARGET_DIR"):
        # Real parser comes later. For now a stub baseline of 0 means the
        # caller is expected to override before scheduling decisions.
        return TargetBaselineObjective.from_dir(env["TARGET_DIR"], parsed_tput=0.0)
    return TimeOnlyObjective()
