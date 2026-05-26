"""Objective abstraction

The Objective is the goal that drives early-stop, pressure scoring, and
the Orchestration prompt. The Coordinator's long-run loop checks
`objective.reached(state)` after each tick to decide whether to stop.

Four concrete implementations (DESIGN §11.2):

* :class:`TargetGainObjective`     — reach `target_gain_pct` % over baseline
* :class:`TargetTputObjective`     — reach `target_tput_per_gpu` absolute tok/s
* :class:`TargetBaselineObjective` — match a referenced external baseline_dir
* :class:`TimeOnlyObjective`       — only `MAX_HOURS` matters (never "reached")

`build_objective(env)` mirrors §11.3 — at most one TARGET_* env var; if
none, falls back to TimeOnly.
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
    """Goal that the Coordinator + Orchestration optimize against.

    All implementations are pure functions of the SharedState — they don't
    touch the bus, the DB, or the LLMs. PolicyGate / Coordinator read their
    output to make stop / pressure / prompt decisions.
    """

    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    def progress(self, state: "SharedState") -> float:
        """0.0 → 1.0, where 1.0 means goal hit."""

    @abstractmethod
    def remaining_gap(self, state: "SharedState") -> float:
        """How far we still need to move (units depend on kind)."""

    @abstractmethod
    def reached(self, state: "SharedState") -> bool: ...

    @abstractmethod
    def pressure_input(self, state: "SharedState") -> float:
        """Feed to scheduler.pressure(): 0.0 = relaxed, 1.0 = max urgency.

        Used by §12 Budget-Aware Scheduler. P2 doesn't run the full
        scheduler yet but the Coordinator still surfaces this value in the
        Orchestration prompt so the LLM can self-pace.
        """

    @abstractmethod
    def describe(self) -> str:
        """One-line summary for prompt injection."""


# ---------------------------------------------------------------------------
@dataclass
class TargetGainObjective(Objective):
    """Reach ``target_gain_pct`` % over baseline_tput.

    Progress = cumulative_gain / target_gain_pct, capped at 1.0.
    """

    target_gain_pct: float

    def __post_init__(self) -> None:
        if self.target_gain_pct <= 0:
            raise ObjectiveError(
                f"TargetGainObjective: target_gain_pct must be > 0, "
                f"got {self.target_gain_pct}"
            )

    def kind(self) -> str:
        return "gain_pct"

    def progress(self, state: "SharedState") -> float:
        return min(1.0, max(0.0, state.cumulative_gain / self.target_gain_pct))

    def remaining_gap(self, state: "SharedState") -> float:
        return max(0.0, self.target_gain_pct - state.cumulative_gain)

    def reached(self, state: "SharedState") -> bool:
        return state.cumulative_gain >= self.target_gain_pct

    def pressure_input(self, state: "SharedState") -> float:
        # Linear ramp 0..1 as we approach target; remains 0 until we have
        # any gain at all (avoids panic before baseline finishes).
        if state.baseline_tput <= 0:
            return 0.0
        return min(1.0, state.cumulative_gain / self.target_gain_pct)

    def describe(self) -> str:
        return f"target_gain_pct={self.target_gain_pct}"


@dataclass
class TargetTputObjective(Objective):
    """Reach an absolute tok/s/GPU number.

    Progress is computed against the **best-so-far** tput, not baseline.
    """

    target_tput_per_gpu: float

    def __post_init__(self) -> None:
        if self.target_tput_per_gpu <= 0:
            raise ObjectiveError(
                f"TargetTputObjective: target_tput_per_gpu must be > 0, "
                f"got {self.target_tput_per_gpu}"
            )

    def kind(self) -> str:
        return "tput"

    def _current_tput(self, state: "SharedState") -> float:
        cb = state.current_best or {}
        v = cb.get("tput") if isinstance(cb, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        return float(state.baseline_tput or 0.0)

    def progress(self, state: "SharedState") -> float:
        cur = self._current_tput(state)
        if cur <= 0:
            return 0.0
        return min(1.0, cur / self.target_tput_per_gpu)

    def remaining_gap(self, state: "SharedState") -> float:
        return max(0.0, self.target_tput_per_gpu - self._current_tput(state))

    def reached(self, state: "SharedState") -> bool:
        return self._current_tput(state) >= self.target_tput_per_gpu

    def pressure_input(self, state: "SharedState") -> float:
        return self.progress(state)

    def describe(self) -> str:
        return f"target_tput_per_gpu={self.target_tput_per_gpu}"


@dataclass
class TargetBaselineObjective(Objective):
    """Match (or beat) the throughput recorded in another session's baseline.

    The reference baseline file is a JSON of the same schema BaselineExecutor
    writes to ``benchmark_report.json``. We only read ``output_throughput``.
    """

    baseline_dir: str
    _ref_tput: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
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
        return "baseline"

    def _cur(self, state: "SharedState") -> float:
        cb = state.current_best or {}
        v = cb.get("tput") if isinstance(cb, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        return float(state.baseline_tput or 0.0)

    def progress(self, state: "SharedState") -> float:
        cur = self._cur(state)
        if self._ref_tput <= 0:
            return 0.0
        return min(1.0, cur / self._ref_tput)

    def remaining_gap(self, state: "SharedState") -> float:
        return max(0.0, self._ref_tput - self._cur(state))

    def reached(self, state: "SharedState") -> bool:
        return self._cur(state) >= self._ref_tput

    def pressure_input(self, state: "SharedState") -> float:
        return self.progress(state)

    def describe(self) -> str:
        return f"target_baseline_dir={self.baseline_dir} (ref_tput={self._ref_tput:.1f})"


@dataclass
class TimeOnlyObjective(Objective):
    """No target — just spend the budget. Never "reached"."""

    def kind(self) -> str:
        return "time_only"

    def progress(self, state: "SharedState") -> float:
        return 0.0

    def remaining_gap(self, state: "SharedState") -> float:
        return float("inf")

    def reached(self, state: "SharedState") -> bool:
        return False

    def pressure_input(self, state: "SharedState") -> float:
        return 0.0

    def describe(self) -> str:
        return "time_only (no target)"


# ---------------------------------------------------------------------------
def build_objective(env: dict[str, Any]) -> Objective:
    """Factory mirroring DESIGN §11.3.

    Required: MAX_HOURS (validated as positive float; the Coordinator uses
    it for the wall-clock stop, not us, but we still validate).
    Optional: at most ONE of TARGET_GAIN_PCT / TARGET_TPUT_PER_GPU /
    TARGET_DIR. None → TimeOnlyObjective.
    """
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
