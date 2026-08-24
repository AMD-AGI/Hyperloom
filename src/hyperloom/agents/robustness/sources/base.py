# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source protocol + DegradeRouter.

DegradeRouter wraps a single source with a backoff state machine.  After
``fail_threshold`` consecutive failures the source is marked DEGRADED and
skipped for ``recheck_interval_s`` seconds; successful ticks restore it to
HEALTHY.  While DEGRADED, ``collect`` returns an empty ``SourceData`` with
``local_processes_known=False`` so downstream signals do not interpret "no
process data" as evidence that nothing is running.

State machine::

    HEALTHY  --(fail_streak >= fail_threshold)-->  DEGRADED
    DEGRADED --(success after recheck_interval_s)--> HEALTHY
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


log = logging.getLogger(__name__)


class HealthState(str, Enum):
    """Routing state of the source inside the DegradeRouter.

    Attributes:
        HEALTHY (str): Source is being consulted normally.
        DEGRADED (str): Source failed enough times to be skipped; it is
            reprobed periodically.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"


class SourceUnavailable(RuntimeError):
    """Raised by a :class:`Source` when its backing service is not reachable.

    DegradeRouter treats it as a countable failure; other exceptions
    propagate so genuine bugs are not masked.
    """


@dataclass
class SourceData:
    """Per-tick snapshot the reactor consumes.

    Every field defaults to an empty container so downstream signals
    treat "no data" uniformly.
    """

    local_gpu: dict[str, Any] = field(default_factory=dict)
    local_processes: list[dict[str, Any]] = field(default_factory=list)
    # ``False`` when the process probe could not answer (``ps`` missing, timed
    # out, disabled). An empty ``local_processes`` then means "we do not know
    # what is running", not "nothing is running", and a consumer must not read
    # the absence of a process as evidence.
    local_processes_known: bool = True
    local_disk: dict[str, Any] = field(default_factory=dict)
    local_log_tail: list[str] = field(default_factory=list)
    local_log_errors: list[dict[str, Any]] = field(default_factory=list)
    local_server_health: list[dict[str, Any]] = field(default_factory=list)
    # LocalProbe extras: local_ray ``{healthy, reason, stderr, returncode}``;
    # local_fd ``{pid, used, limit, used_pct}``; local_aiter_jit ``{jit_dir, so_count, build_count}``.
    local_ray: dict[str, Any] = field(default_factory=dict)
    local_fd: dict[str, Any] = field(default_factory=dict)
    local_aiter_jit: dict[str, Any] = field(default_factory=dict)
    # Decision-audit: ``recent_integrate``, ``ci_metrics`` ({} if absent), ``oob_attempts``.
    local_decision_audit: dict[str, Any] = field(default_factory=dict)
    # Preflight inputs (signals/preflight.py): ``local_manifest`` raw manifest.json;
    # ``local_kernel_breakdown`` ``{tier_pcts, total_kernels, total_gpu_pct, mtime}``.
    local_manifest: dict[str, Any] = field(default_factory=dict)
    local_kernel_breakdown: dict[str, Any] = field(default_factory=dict)
    # Critic health: ``recent_judges`` + ``workdir_count`` (subdirs under critic-workdir/).
    local_critic_health: dict[str, Any] = field(default_factory=dict)
    # State-integrity slots: ``state_json``, ``wal`` {wal_bytes, db_bytes, db_path},
    # ``agents`` {<role>: {inbox_bytes, outbox_bytes}}, ``coordinator``
    # {recorded_pid, alive, pid_file}.
    local_state_integrity: dict[str, Any] = field(default_factory=dict)
    # External-deps: ``gateway`` (OPENAI_BASE_URL/models), ``mounts`` (stat latency for
    # TRACELENS_ROOT / TRACELENS_INTERNAL_ROOT / INFERENCEX_PATH), ``tracelens_cli``.
    local_external_deps: dict[str, Any] = field(default_factory=dict)
    # Coordinator bus events: ``{id, agent, topic, payload, ts}``, oldest first.
    # ``id`` is the ``events.seq`` key detectors dedupe inbox items against.
    coordinator_events: list[dict[str, Any]] = field(default_factory=list)
    # In-flight work: ``{running, by_agent: {agent: {last_progress_unix, task,
    # oldest_progress_unix, oldest_task}}}``.
    # A composite task reports a heartbeat per internal unit, so this answers
    # "is *this agent's* dispatched work still moving" for an agent that is
    # legitimately quiet while it waits on one.
    local_task_progress: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    """Async source contract the DegradeRouter consumes."""

    name: str

    async def fetch(self, ctx: Any) -> SourceData:
        """Return a snapshot or raise :class:`SourceUnavailable`.

        Args:
            ctx (Any): The per-tick reactor context (clock, shared
                state, session id, etc.).

        Returns:
            SourceData: The snapshot collected for this tick.

        Raises:
            SourceUnavailable: When the backing service is unreachable.
        """


@dataclass
class _SourceState:
    """Mutable per-source bookkeeping tracked by the DegradeRouter.

    Attributes:
        name (str): The source's name (used in transition logs).
        state (HealthState): Current routing state of the source.
        fail_streak (int): Consecutive failure count; reset on success.
        last_recheck (float): Clock value at the last fetch attempt,
            used to space out reprobes while DEGRADED.
    """

    name: str
    state: HealthState = HealthState.HEALTHY
    fail_streak: int = 0
    last_recheck: float = 0.0


class DegradeRouter:
    """Single-source router with a backoff state machine.

    Consults the source each tick while HEALTHY; after ``fail_threshold``
    consecutive failures transitions to DEGRADED and returns empty
    ``SourceData`` (with ``local_processes_known=False``) until
    ``recheck_interval_s`` seconds elapse, at which point one reprobe is
    attempted.  A successful reprobe restores HEALTHY; a failed one keeps
    the source DEGRADED and resets the recheck timer.
    """

    def __init__(
        self,
        primary: Source,
        *,
        fail_threshold: int = 3,
        recheck_interval_s: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialise the router.

        Args:
            primary (Source): The sole source consulted each tick.
            fail_threshold (int): Consecutive primary failures required
                to mark it DEGRADED; clamped to at least 1.
            recheck_interval_s (float): Seconds between primary reprobes
                once DEGRADED; clamped to at least 0.
            clock (Callable[[], float] | None): Optional time source;
                defaults to :func:`time.monotonic`.
        """
        self._primary = primary
        self._fail_threshold = max(1, int(fail_threshold))
        self._recheck_interval_s = max(0.0, float(recheck_interval_s))
        self._clock = clock or time.monotonic
        self._state = _SourceState(name=primary.name)

    async def collect(self, ctx: Any) -> SourceData:
        """Fetch one tick of source data with backoff on repeated failure.

        Returns an empty ``SourceData(local_processes_known=False)`` while
        the primary is DEGRADED so downstream probe-derived rules stay
        quiet rather than misfiring.

        Args:
            ctx (Any): The per-tick reactor context passed to the source's
                ``fetch``.

        Returns:
            SourceData: The snapshot from the primary, or an empty
            snapshot while DEGRADED.
        """
        if not self._should_try_primary():
            return SourceData(local_processes_known=False)

        try:
            data = await self._primary.fetch(ctx)
        except SourceUnavailable as exc:
            self._record_failure(str(exc))
            return SourceData(local_processes_known=False)
        except Exception:
            self._record_failure("unexpected_exception")
            log.exception("source %s raised unexpectedly", self._primary.name)
            return SourceData(local_processes_known=False)
        else:
            self._record_success()
            return data

    # -- state machine helpers ------------------------------------------

    def _should_try_primary(self) -> bool:
        """Decide whether the primary should be attempted this tick."""
        if self._state.state is HealthState.HEALTHY:
            return True
        now = self._clock()
        if (now - self._state.last_recheck) >= self._recheck_interval_s:
            self._state.last_recheck = now
            return True
        return False

    def _record_success(self) -> None:
        """Mark the source healthy after a successful fetch."""
        if self._state.state is not HealthState.HEALTHY:
            self._maybe_log_transition(HealthState.HEALTHY, "recovered")
            self._state.state = HealthState.HEALTHY
        self._state.fail_streak = 0
        self._state.last_recheck = self._clock()

    def _record_failure(self, reason: str) -> None:
        """Record a failed fetch and degrade the source past threshold."""
        self._state.fail_streak += 1
        self._state.last_recheck = self._clock()
        if self._state.state is HealthState.HEALTHY and self._state.fail_streak >= self._fail_threshold:
            self._maybe_log_transition(HealthState.DEGRADED, reason)
            self._state.state = HealthState.DEGRADED

    def _maybe_log_transition(self, target: HealthState, reason: str) -> None:
        """Emit a single WARN log for a state transition."""
        if self._state.state is target:
            return
        log.warning(
            "source %s state %s -> %s (reason=%s, streak=%d)",
            self._state.name,
            self._state.state.value,
            target.value,
            reason,
            self._state.fail_streak,
        )
