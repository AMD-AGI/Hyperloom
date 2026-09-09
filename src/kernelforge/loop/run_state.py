# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""File-backed run state + event log for the long-horizon forge-loop."""

from __future__ import annotations

import collections
import fcntl
import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from kernelforge.loop.search_policy import (
    OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
    SEARCH_MODE_EXPLOIT,
    SEARCH_MODES,
)
from kernelforge.durable_io import atomic_write_text

log = logging.getLogger(__name__)

SCHEMA_VERSION = 19

# How many trailing events the store keeps in memory to serve ``recent_events`` without re-reading ``events.jsonl``
# each iteration (see LoopStateStore).
_RECENT_CACHE = 64

# How many trailing ``iteration_result`` events the store keeps in memory to serve ``recent_results``.
_RECENT_RESULT_CACHE = 32

# Phase labels.
PHASE_EXPLORE = "explore"
PHASE_EXPLOIT = "exploit_best_lineage"
PHASE_STALLED = "stalled_explore"

SESSION_RUNNING = "running"
SESSION_PAUSED = "paused"
SESSION_COMPLETED = "completed"
SESSION_INTERRUPTED = "interrupted"
_TERMINAL_SESSION_STATUSES = {
    SESSION_PAUSED,
    SESSION_COMPLETED,
    SESSION_INTERRUPTED,
}

ORCHESTRATION_CIRCUIT_CLOSED = "closed"
ORCHESTRATION_CIRCUIT_OPEN = "open"
ORCHESTRATION_CIRCUIT_HALF_OPEN = "half_open"
ORCHESTRATION_CIRCUIT_STATES = frozenset(
    {
        ORCHESTRATION_CIRCUIT_CLOSED,
        ORCHESTRATION_CIRCUIT_OPEN,
        ORCHESTRATION_CIRCUIT_HALF_OPEN,
    }
)


@dataclass
class BestRecord:
    """The current best kept iteration (the loop's KEEP anchor)."""

    iteration: int = 0
    # Raw aggregate diagnostic for the selected candidate.
    wall_ms: float | None = None
    mean_case_speedup: float | None = None
    commit_hash: str = ""
    plan: str = ""
    source: str = ""


@dataclass
class StallState:
    """No-improvement streaks and Supervisor attempt/intervention anchors."""

    no_improvement_iters: int = 0
    unresolved_stall_iters: int = 0
    last_supervisor_iter: int = 0
    last_supervisor_attempt_iter: int = 0


@dataclass
class CumulativeCounters:
    """Reporting totals accumulated across all campaign sessions."""

    iterations: int = 0
    kept: int = 0
    reverted: int = 0
    api_errors: int = 0
    orchestration_errors: int = 0


@dataclass
class AnalysisRefreshState:
    """Durable anchor and attempt state for Analysis refresh decisions."""

    evidence_commit: str = ""
    evidence_mean_case_speedup: float | None = None
    evidence_status: str = ""
    last_attempt_commit: str = ""
    last_attempt_status: str = ""
    last_attempt_iteration: int = -1


@dataclass
class CriticRuling:
    """The last Plan Critic verdict, and where the review that made it lives."""

    verdict: str = ""
    review_path: str = ""


# How many recent rounds the cost history keeps.
ROUND_COST_WINDOW = 5


@dataclass
class RoundCost:
    """What one round cost, split at the point its plans were published."""

    iteration: int = 0
    lanes: int = 1
    planning_sec: float = 0.0
    total_sec: float = 0.0
    measurement_sec: float = 0.0


@dataclass
class RoundCostState:
    """Observed round costs: campaign totals, plus a bounded recent window."""

    rounds: int = 0
    planning_total_sec: float = 0.0
    total_sec: float = 0.0
    campaign_sec: float = 0.0
    recent: list[RoundCost] = field(default_factory=list)

    def planning_share_pct(self) -> float | None:
        """Planning as a percentage of the campaign wall-clock it was spent in."""
        if self.campaign_sec <= 0:
            return None
        return 100.0 * self.planning_total_sec / self.campaign_sec


def _validate_round_costs(costs: "RoundCostState") -> None:
    """Reject a cost history the admission estimate could not be built on."""
    durations: list[tuple[str, object]] = [
        ("round_costs.planning_total_sec", costs.planning_total_sec),
        ("round_costs.total_sec", costs.total_sec),
        ("round_costs.campaign_sec", costs.campaign_sec),
    ]
    for index, entry in enumerate(costs.recent):
        durations.append((f"round_costs.recent[{index}].planning_sec", entry.planning_sec))
        durations.append((f"round_costs.recent[{index}].total_sec", entry.total_sec))
        durations.append(
            (
                f"round_costs.recent[{index}].measurement_sec",
                entry.measurement_sec,
            )
        )
    for label, value in durations:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"run state {label} must be a number")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"run state {label} must be a non-negative finite duration")
    if costs.rounds < 0:
        raise ValueError("run state round_costs.rounds must not be negative")
    if float(costs.campaign_sec) < float(costs.planning_total_sec):
        raise ValueError("run state round_costs.campaign_sec must cover round_costs.planning_total_sec")


@dataclass
class RunState:
    """Small, resumable control checkpoint for one forge-loop campaign."""

    schema_version: int = SCHEMA_VERSION
    campaign_id: str = ""
    session_index: int = 0
    session_status: str = ""
    last_experiment_id: str = ""
    kernel_path: str = ""
    task_fingerprint: str = ""
    git_branch: str = ""
    head_commit: str = ""
    iteration: int = 0
    next_iteration: int = 1
    cumulative: CumulativeCounters = field(default_factory=CumulativeCounters)
    orchestration_error_streak: int = 0
    orchestration_circuit_state: str = ORCHESTRATION_CIRCUIT_CLOSED
    intervention_count: int = 0
    phase: str = PHASE_EXPLORE
    search_mode: str = SEARCH_MODE_EXPLOIT
    search_reason_codes: list[str] = field(default_factory=list)
    search_objective: str = OBJECTIVE_IMMEDIATE_CANONICAL_GAIN
    search_mode_residence_remaining: int = 0
    diversification_cycle_completed: bool = False
    baseline_wall_ms: float | None = None
    pristine_baseline_wall_ms: float | None = None
    # Per-scored-case baseline wall times (case_id -> ms), captured once on the pristine kernel.
    baseline_case_times: dict = field(default_factory=dict)
    # Scoring state that decides keep/revert.
    best_case_times: dict = field(default_factory=dict)
    unscored_cases: list[str] = field(default_factory=list)
    best: BestRecord = field(default_factory=BestRecord)
    stall: StallState = field(default_factory=StallState)
    analysis: AnalysisRefreshState = field(default_factory=AnalysisRefreshState)
    last_critic: CriticRuling = field(default_factory=CriticRuling)
    # What this campaign's own rounds have cost, which is what decides whether the remaining budget can pay for
    # another one.
    round_costs: RoundCostState = field(default_factory=RoundCostState)
    # Iterations worth re-reading in full (best + notable near-misses).
    pinned_iterations: list[int] = field(default_factory=list)
    termination_reason: str = ""

    def to_dict(self) -> dict:
        """Serialize the current durable state schema."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        """Rebuild a RunState from the exact current schema."""
        if not isinstance(d, dict):
            raise ValueError("run state must be a JSON object")
        version = d.get("schema_version")
        payload = dict(d)
        if version == 13:
            # v13 predates the durable Analysis refresh anchor.
            payload["analysis"] = asdict(AnalysisRefreshState())
            version = 14
        if version == 14:
            # v14 predates the durable Plan Critic ruling.
            payload["last_critic"] = asdict(CriticRuling())
            version = 15
        if version == 15:
            # v15 predates the round cost history.
            payload["round_costs"] = asdict(RoundCostState())
            version = 16
        if version == 16:
            # v16 recorded what a round spent planning but not what its canonical measurement cost, which was then
            # priced from the per-step timeout ceilings rather than from observation.
            costs = payload.get("round_costs")
            if isinstance(costs, dict):
                for entry in costs.get("recent") or []:
                    if isinstance(entry, dict):
                        entry.setdefault("measurement_sec", 0.0)
            version = 17
        if version == 17:
            # v17 accumulated campaign-cumulative planning with no campaign wall-clock to divide it by, so the report
            # divided it by the CURRENT process's elapsed time -- the wrong span on any resumed campaign, and the
            # reason a 10-minute session against 45 minutes of cumulative planning published "450% of the run".
            costs = payload.get("round_costs")
            if isinstance(costs, dict):
                costs.setdefault(
                    "campaign_sec",
                    max(
                        float(costs.get("total_sec", 0.0) or 0.0),
                        float(costs.get("planning_total_sec", 0.0) or 0.0),
                    ),
                )
            version = 18
        if version == 18:
            # v18 read one counter for both the supervisor cooldown and the search-mode switch.
            stall = payload.get("stall")
            if isinstance(stall, dict):
                stall.setdefault(
                    "unresolved_stall_iters",
                    int(stall.get("no_improvement_iters", 0) or 0),
                )
            version = SCHEMA_VERSION
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported run state schema: expected v{SCHEMA_VERSION}, got {version!r}")
        payload["schema_version"] = SCHEMA_VERSION

        expected = set(cls.__dataclass_fields__)
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError("run state missing fields: " + ", ".join(sorted(missing)))
        if unknown:
            raise ValueError("run state has unknown fields: " + ", ".join(sorted(unknown)))

        def nested(
            value: object,
            model,
            label: str,
        ):
            if not isinstance(value, dict):
                raise ValueError(f"run state {label} must be an object")
            nested_expected = set(model.__dataclass_fields__)
            nested_missing = nested_expected - set(value)
            nested_unknown = set(value) - nested_expected
            if nested_missing:
                raise ValueError(f"run state {label} missing fields: " + ", ".join(sorted(nested_missing)))
            if nested_unknown:
                raise ValueError(f"run state {label} has unknown fields: " + ", ".join(sorted(nested_unknown)))
            return model(**value)

        payload["best"] = nested(payload["best"], BestRecord, "best")
        payload["stall"] = nested(payload["stall"], StallState, "stall")
        payload["analysis"] = nested(
            payload["analysis"],
            AnalysisRefreshState,
            "analysis",
        )
        payload["cumulative"] = nested(
            payload["cumulative"],
            CumulativeCounters,
            "cumulative",
        )
        payload["last_critic"] = nested(
            payload["last_critic"],
            CriticRuling,
            "last_critic",
        )
        round_costs = nested(
            payload["round_costs"],
            RoundCostState,
            "round_costs",
        )
        if not isinstance(round_costs.recent, list):
            raise ValueError("run state round_costs.recent must be a list")
        round_costs.recent = [
            nested(entry, RoundCost, f"round_costs.recent[{index}]") for index, entry in enumerate(round_costs.recent)
        ]
        payload["round_costs"] = round_costs
        state = cls(**payload)
        _validate_round_costs(state.round_costs)
        if state.search_mode not in SEARCH_MODES:
            raise ValueError(f"run state has unsupported search mode: {state.search_mode!r}")
        if state.orchestration_circuit_state not in ORCHESTRATION_CIRCUIT_STATES:
            raise ValueError(
                f"run state has unsupported orchestration circuit state: {state.orchestration_circuit_state!r}"
            )
        if state.next_iteration < 1:
            raise ValueError("run state next_iteration must be positive")
        return state


def start_session(
    state: "RunState",
    *,
    campaign_id: str = "",
    experiment_id: str = "",
) -> "RunState":
    """Start the next process-local session while preserving campaign identity."""
    requested_campaign_id = (campaign_id or "").strip()
    if state.campaign_id:
        if requested_campaign_id and requested_campaign_id != state.campaign_id:
            raise ValueError(f"campaign mismatch: expected {state.campaign_id}, got {requested_campaign_id}")
    else:
        state.campaign_id = requested_campaign_id or uuid.uuid4().hex

    if state.session_status == SESSION_COMPLETED:
        raise ValueError("completed campaign cannot start another session")

    state.session_index += 1
    state.session_status = SESSION_RUNNING
    state.last_experiment_id = (experiment_id or "").strip()
    state.next_iteration = max(1, state.next_iteration, state.iteration + 1)
    state.termination_reason = ""
    return state


def reconcile_stale_running_session(state: "RunState") -> bool:
    """Mark a prior process-local RUNNING session as interrupted."""
    if state.session_status != SESSION_RUNNING:
        return False
    finish_session(
        state,
        status=SESSION_INTERRUPTED,
        reason="stale_running_session_reconciled",
    )
    return True


def finish_session(
    state: "RunState",
    *,
    status: str,
    reason: str = "",
) -> "RunState":
    """Finish the active session as paused, completed, or interrupted."""
    if state.session_status != SESSION_RUNNING:
        raise ValueError("no running session to finish")
    if status not in _TERMINAL_SESSION_STATUSES:
        raise ValueError(f"invalid terminal session status: {status}")
    state.session_status = status
    state.termination_reason = (reason or "").strip()
    return state


def make_event(event_type: str, iteration: int, **fields: object) -> dict:
    """Build one factual event record (timestamp + type + iteration + fields)."""
    event: dict = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": str(event_type),
        "iter": int(iteration),
    }
    for key, value in fields.items():
        if value is not None:
            event[key] = value
    return event


# Stall streak at/above which the run is labelled stalled (mirrors the loop's default ``supervise_after``; the loop
# passes its own value in).
_DEFAULT_STALL_PHASE_THRESHOLD = 3

# Decisions that record an infrastructure failure rather than an attempt at the kernel.
INFRASTRUCTURE_DECISIONS = frozenset({"API_ERROR", "ORCHESTRATION_ERROR"})


def is_infrastructure_decision(decision: str) -> bool:
    """Whether this decision label reports infrastructure, not optimization."""
    return str(decision or "").strip().upper() in INFRASTRUCTURE_DECISIONS


# Decisions where the session ended without ever measuring the direction it was given: the infrastructure failures
# above plus an AGENT_ERROR, which ends the session on the same empty diff.
UNMEASURED_DECISIONS = INFRASTRUCTURE_DECISIONS | frozenset({"AGENT_ERROR"})


def measured_nothing(decision: str) -> bool:
    """Whether this decision label reports that nothing was measured at all."""
    return str(decision or "").strip().upper() in UNMEASURED_DECISIONS


def _phase_for(state: "RunState", stall_threshold: int) -> str:
    """Derive the coarse phase from the current best + unresolved stall."""
    if state.stall.unresolved_stall_iters >= max(1, stall_threshold):
        return PHASE_STALLED
    if state.best.iteration > 0 or state.best.commit_hash:
        return PHASE_EXPLOIT
    return PHASE_EXPLORE


def apply_iteration(
    state: "RunState",
    *,
    iteration: int,
    decision: str,
    kept: bool,
    wall_ms: float | None,
    mean_case_speedup: float | None = None,
    commit_hash: str,
    plan: str,
    baseline_wall_ms: float | None,
    best_wall_ms: float | None,
    best_mean_case_speedup: float | None = None,
    stall_threshold: int = _DEFAULT_STALL_PHASE_THRESHOLD,
    orchestration_error_threshold: int = 3,
    max_pinned: int = 8,
) -> "RunState":
    """Reduce one finished iteration's outcome into the run state (in place)."""
    if iteration < state.next_iteration:
        raise ValueError(
            f"iteration {iteration} would reuse completed iteration; next iteration is {state.next_iteration}"
        )
    infrastructure = is_infrastructure_decision(decision)
    state.iteration = iteration
    state.next_iteration = iteration + 1
    state.cumulative.iterations += 1
    if kept:
        state.cumulative.kept += 1
    elif infrastructure:
        if str(decision or "").strip().upper() == "API_ERROR":
            state.cumulative.api_errors += 1
        else:
            state.cumulative.orchestration_errors += 1
    else:
        state.cumulative.reverted += 1
    if baseline_wall_ms is not None:
        state.baseline_wall_ms = baseline_wall_ms

    if kept:
        state.best = BestRecord(
            iteration=iteration,
            wall_ms=wall_ms if wall_ms is not None else best_wall_ms,
            mean_case_speedup=(mean_case_speedup if mean_case_speedup is not None else best_mean_case_speedup),
            commit_hash=commit_hash or state.best.commit_hash,
            plan=(plan or "").strip()[:120],
            source="iteration",
        )
        state.stall.no_improvement_iters = 0
        state.stall.unresolved_stall_iters = 0
        pin_iteration(state, iteration, max_pinned=max_pinned)
    elif not infrastructure:
        state.stall.no_improvement_iters += 1
        state.stall.unresolved_stall_iters += 1
    if str(decision or "").strip().upper() == "ORCHESTRATION_ERROR":
        state.orchestration_error_streak += 1
        if (
            state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_HALF_OPEN
            or state.orchestration_error_streak >= max(1, orchestration_error_threshold)
        ):
            state.orchestration_circuit_state = ORCHESTRATION_CIRCUIT_OPEN
    else:
        state.orchestration_error_streak = 0
        state.orchestration_circuit_state = ORCHESTRATION_CIRCUIT_CLOSED

    # Safety sync of the loop's authoritative mean case speedup, but ONLY once a real KEEP exists.
    if (
        best_mean_case_speedup is not None
        and (state.best.iteration > 0 or bool(state.best.commit_hash))
        and (state.best.mean_case_speedup is None or best_mean_case_speedup >= state.best.mean_case_speedup)
    ):
        state.best.mean_case_speedup = best_mean_case_speedup
        state.best.wall_ms = best_wall_ms

    state.phase = _phase_for(state, stall_threshold)
    return state


def apply_round_cost(
    state: "RunState",
    *,
    iteration: int,
    lanes: int,
    planning_sec: float,
    total_sec: float,
    campaign_sec: float,
    measurement_sec: float = 0.0,
    window: int = ROUND_COST_WINDOW,
) -> "RunState":
    """Record what one finished round cost (in place)."""
    planning = max(0.0, float(planning_sec))
    total = max(planning, float(total_sec))
    measurement = max(0.0, float(measurement_sec))
    if planning <= 0:
        raise ValueError("a round that did not plan has no cost to record")
    costs = state.round_costs
    costs.rounds += 1
    costs.planning_total_sec += planning
    costs.total_sec += total
    costs.campaign_sec = max(
        costs.campaign_sec,
        float(campaign_sec),
        costs.planning_total_sec,
    )
    costs.recent.append(
        RoundCost(
            iteration=iteration,
            lanes=max(1, int(lanes)),
            planning_sec=planning,
            total_sec=total,
            measurement_sec=measurement,
        )
    )
    costs.recent = costs.recent[-max(1, window) :]
    return state


def begin_orchestration_probe(state: "RunState") -> "RunState":
    """Move an explicitly resumed open circuit into half-open state."""
    if state.orchestration_circuit_state != ORCHESTRATION_CIRCUIT_OPEN:
        raise ValueError("only an open orchestration circuit can enter half-open")
    state.orchestration_circuit_state = ORCHESTRATION_CIRCUIT_HALF_OPEN
    return state


def complete_orchestration_probe(state: "RunState") -> "RunState":
    """Close the circuit after one successful orchestration call."""
    if state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN:
        raise ValueError("an open orchestration circuit cannot complete a probe")
    state.orchestration_error_streak = 0
    state.orchestration_circuit_state = ORCHESTRATION_CIRCUIT_CLOSED
    return state


def apply_supervisor_attempt(
    state: "RunState",
    *,
    iteration: int,
) -> "RunState":
    """Persist the cooldown anchor for every actual Supervisor call."""
    state.stall.last_supervisor_attempt_iter = iteration
    return state


def apply_supervisor_intervention(
    state: "RunState",
    *,
    iteration: int,
    stall_threshold: int = _DEFAULT_STALL_PHASE_THRESHOLD,
) -> "RunState":
    """Reset the durable supervisor cooldown after an intervention."""
    state.stall.no_improvement_iters = 0
    state.stall.last_supervisor_iter = iteration
    state.stall.last_supervisor_attempt_iter = iteration
    state.intervention_count += 1
    state.phase = _phase_for(state, stall_threshold)
    return state


def should_resume(state: "RunState", head_commit: str) -> bool:
    """Whether a loaded state is a safe resume point for the current HEAD."""
    recorded = (state.best.commit_hash or "").strip()
    head = (head_commit or "").strip()
    return bool(recorded and state.best.mean_case_speedup is not None and head and head == recorded)


# How many iterations the retrieval map can hold at once.
MAX_PINNED_ITERATIONS = 8


def pin_iteration(
    state: "RunState",
    iteration: int,
    *,
    max_pinned: int = MAX_PINNED_ITERATIONS,
) -> None:
    """Mark an iteration worth re-reading in full (deduped, capped, in place)."""
    if iteration in state.pinned_iterations:
        return
    state.pinned_iterations.append(iteration)
    if len(state.pinned_iterations) <= max_pinned:
        return
    recent = state.pinned_iterations[-max_pinned:]
    best = state.best.iteration
    if best in state.pinned_iterations and best not in recent:
        recent = [best, *recent[1:]]
    state.pinned_iterations = recent


class WorkspaceLockError(RuntimeError):
    """Raised when another process already owns a campaign workspace."""


class WorkspaceLock:
    """Non-blocking process lock for one ``forge_experiments`` root."""

    def __init__(self, path: Path):
        self.path = path
        self._file: TextIO | None = None

    def acquire(self) -> "WorkspaceLock":
        if self._file is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.path, "a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            lock_file.close()
            raise WorkspaceLockError(f"workspace is already in use: {self.path.parent.parent}") from e
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        self._file = lock_file
        return self

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "WorkspaceLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


class LoopStateStore:
    """Best-effort file store for ``run_state.json`` + ``events.jsonl``."""

    def __init__(self, workspace_dir: str):
        self.root = Path(workspace_dir) / "forge_experiments"
        self.state_path = self.root / "run_state.json"
        self.events_path = self.root / "events.jsonl"
        self.lock_path = self.root / "workspace.lock"
        self.degraded = False
        self.persistence_errors: list[str] = []
        # Bounded in-memory tails of recent events so ``recent_events`` and ``recent_results`` (called once per
        # iteration for the prompt header and the search policy) are O(1) and never re-parse the whole, ever-growing
        # ``events.jsonl``.
        self._recent: collections.deque[dict] = collections.deque(maxlen=_RECENT_CACHE)
        self._recent_results: collections.deque[dict] = collections.deque(maxlen=_RECENT_RESULT_CACHE)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001 - best-effort
            self._mark_degraded("create root", e)
        self._prime_recent()

    def _mark_degraded(self, operation: str, error: Exception) -> None:
        """Record a bounded, externally visible persistence failure."""
        self.degraded = True
        message = f"{operation}: {error}"
        self.persistence_errors.append(message)
        self.persistence_errors = self.persistence_errors[-10:]
        log.debug("run_state: %s", message)

    def workspace_lock(self) -> WorkspaceLock:
        """Return a fail-closed, non-blocking lock for this workspace."""
        return WorkspaceLock(self.lock_path)

    def _prime_recent(self) -> None:
        """Seed the in-memory recent-event caches from disk (once, at init)."""
        try:
            events = self.read_events()
            for event in events[-_RECENT_CACHE:]:
                self._recent.append(event)
            # Scanned in full rather than from the tail above: an outcome older than the last ``_RECENT_CACHE`` events
            # is still inside the outcome window, and the deque keeps only what fits.
            for event in events:
                if event.get("type") == "iteration_result":
                    self._recent_results.append(event)
        except Exception as e:  # noqa: BLE001 - best-effort
            self._mark_degraded("prime recent cache", e)

    def load(self) -> RunState:
        """Load the current persisted state, or a fresh state when absent."""
        if not self.state_path.exists():
            return RunState()
        try:
            payload = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid run state checkpoint: {self.state_path}") from error
        return RunState.from_dict(payload)

    def save(self, state: RunState) -> None:
        """Atomically overwrite ``run_state.json``."""
        try:
            atomic_write_text(
                self.state_path,
                json.dumps(state.to_dict(), indent=2, sort_keys=True),
            )
        except Exception as e:  # noqa: BLE001 - best-effort
            self._mark_degraded(f"save {self.state_path}", e)

    def append_event(self, event: dict) -> None:
        """Append one factual event as a JSON line and to the recent caches."""
        # Update the in-memory tails first so the prompt view reflects this event even if the disk append fails (both
        # are best-effort).
        self._recent.append(event)
        if event.get("type") == "iteration_result":
            self._recent_results.append(event)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.events_path, "a") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:  # noqa: BLE001 - best-effort
            self._mark_degraded(f"append {self.events_path}", e)

    def read_events(self) -> list[dict]:
        """All events in order, skipping malformed lines (best-effort)."""
        out: list[dict] = []
        try:
            if not self.events_path.exists():
                return out
            for line in self.events_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception as e:  # noqa: BLE001 - skip a bad line
                    log.debug("run_state: skipping malformed event line: %s", e)
                    continue
                event_type = event.get("type") if isinstance(event, dict) else None
                iteration = event.get("iter") if isinstance(event, dict) else None
                if (
                    isinstance(event, dict)
                    and isinstance(event_type, str)
                    and event_type
                    and isinstance(iteration, int)
                    and not isinstance(iteration, bool)
                    and iteration >= 0
                ):
                    out.append(event)
                else:
                    log.debug(
                        "run_state: skipping invalid event record: %r",
                        event,
                    )
        except Exception as e:  # noqa: BLE001 - best-effort
            self._mark_degraded(f"read {self.events_path}", e)
        return out

    def recent_events(self, n: int) -> list[dict]:
        """The last ``n`` events (oldest first), served from the in-memory cache."""
        if n <= 0:
            return []
        return list(self._recent)[-n:]

    def recent_results(self, n: int) -> list[dict]:
        """The last ``n`` ``iteration_result`` events (oldest first), from cache."""
        if n <= 0:
            return []
        bound = self._recent_results.maxlen
        if n > bound:
            raise ValueError(f"recent_results({n}) exceeds the cached outcome bound {bound}")
        return list(self._recent_results)[-n:]
