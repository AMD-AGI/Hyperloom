# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Source protocol + DegradeRouter.

DegradeRouter routes to a primary source and falls back to a secondary
on repeated failure. State transitions emit one WARN log; in-state
retries are silent. State machine::

    HEALTHY  --(fail_streak >= fail_threshold)-->  DEGRADED
    DEGRADED --(success after recheck_interval_s)--> HEALTHY
    DEGRADED --(consecutive failures still over budget)--> FAILED  (M2+)

M1 uses only HEALTHY/DEGRADED: DEGRADED means "use the fallback this
tick"; the next tick re-probes after ``recheck_interval_s``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


log = logging.getLogger(__name__)


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class SourceUnavailable(RuntimeError):
    """Raised by a :class:`Source` when its backing service is not reachable.

    DegradeRouter treats it as a countable failure; other exceptions
    propagate so genuine bugs are not masked.
    """


@dataclass
class SourceData:
    """Per-tick snapshot the reactor consumes.

    Every field defaults to an empty container so downstream signals
    treat "no data" uniformly. ``sources_used`` records which source
    produced each tick; ``degraded_reason`` is set on fallback.
    """

    session_pods: list[dict[str, Any]] = field(default_factory=list)
    session_metrics: dict[str, Any] = field(default_factory=dict)
    session_events: list[dict[str, Any]] = field(default_factory=list)
    session_summary: dict[str, Any] = field(default_factory=dict)
    cluster_faults: list[dict[str, Any]] = field(default_factory=list)
    local_gpu: dict[str, Any] = field(default_factory=dict)
    local_processes: list[dict[str, Any]] = field(default_factory=list)
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
    # Critic health: ``recent_judges`` + ``workdir_count`` (subdirs under critic-workdir/, E4).
    local_critic_health: dict[str, Any] = field(default_factory=dict)
    # State-integrity slots: ``state_json``, ``wal`` {wal_bytes, db_bytes, db_path},
    # ``leases`` (pid liveness), ``agents`` {<role>: {inbox_bytes, outbox_bytes}},
    # ``coordinator`` {recorded_pid, alive, pid_file}.
    local_state_integrity: dict[str, Any] = field(default_factory=dict)
    # External-deps: ``gateway`` (OPENAI_BASE_URL/models), ``mounts`` (stat latency for
    # TRACELENS_ROOT / TRACELENS_INTERNAL_ROOT / INFERENCEX_PATH / OOB_SRC), ``tracelens_cli``.
    local_external_deps: dict[str, Any] = field(default_factory=dict)
    coordinator_events: list[dict[str, Any]] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    degraded_reason: str | None = None

    def merge_from(self, other: "SourceData") -> None:
        """Merge another snapshot in, preserving non-empty existing fields."""
        for slot in (
            "session_pods",
            "session_events",
            "cluster_faults",
            "local_processes",
            "local_log_tail",
            "local_log_errors",
            "local_server_health",
            "coordinator_events",
        ):
            if not getattr(self, slot):
                setattr(self, slot, list(getattr(other, slot)))
        for slot in (
            "session_metrics",
            "session_summary",
            "local_gpu",
            "local_disk",
            "local_ray",
            "local_fd",
            "local_aiter_jit",
            "local_decision_audit",
            "local_manifest",
            "local_kernel_breakdown",
            "local_critic_health",
            "local_state_integrity",
            "local_external_deps",
        ):
            if not getattr(self, slot):
                setattr(self, slot, dict(getattr(other, slot)))
        if other.degraded_reason and not self.degraded_reason:
            self.degraded_reason = other.degraded_reason
        for name in other.sources_used:
            if name not in self.sources_used:
                self.sources_used.append(name)


@runtime_checkable
class Source(Protocol):
    """Async source contract the DegradeRouter consumes."""

    name: str

    async def fetch(self, ctx: Any) -> SourceData:
        """Return a snapshot or raise :class:`SourceUnavailable`."""
        ...


@dataclass
class _SourceState:
    name: str
    state: HealthState = HealthState.HEALTHY
    fail_streak: int = 0
    last_recheck: float = 0.0


class DegradeRouter:
    """Coordinator-tick routing across [primary, fallback] sources.

    Parameters
    ----------
    primary, fallback:
        :class:`Source` instances. The router consults the primary
        first; on enough consecutive failures it switches to the
        fallback for subsequent ticks and reprobes the primary every
        ``recheck_interval_s`` seconds.
    fail_threshold:
        Consecutive primary failures required to mark it DEGRADED.
    recheck_interval_s:
        Time between primary reprobes once it is DEGRADED.
    clock:
        Optional ``Callable[[], float]`` providing the current time;
        defaults to :func:`time.monotonic`. Tests inject a controllable
        clock to assert recheck timing without sleeping.
    """

    def __init__(
        self,
        primary: Source,
        fallback: Source,
        *,
        fail_threshold: int = 3,
        recheck_interval_s: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fail_threshold = max(1, int(fail_threshold))
        self._recheck_interval_s = max(0.0, float(recheck_interval_s))
        self._clock = clock or time.monotonic
        self._states: dict[str, _SourceState] = {
            primary.name: _SourceState(name=primary.name),
            fallback.name: _SourceState(name=fallback.name),
        }

    @property
    def primary_state(self) -> HealthState:
        return self._states[self._primary.name].state

    async def collect(self, ctx: Any) -> SourceData:
        """Fetch one tick of source data, with degrade routing."""
        primary_state = self._states[self._primary.name]
        if self._should_try_primary(primary_state):
            try:
                data = await self._primary.fetch(ctx)
            except SourceUnavailable as exc:
                self._record_failure(primary_state, str(exc))
            except Exception:
                # Count unexpected errors as failures so we eventually degrade.
                self._record_failure(primary_state, "unexpected_exception")
                log.exception("primary source %s raised unexpectedly", self._primary.name)
            else:
                self._record_success(primary_state)
                if self._primary.name not in data.sources_used:
                    data.sources_used = [*data.sources_used, self._primary.name]
                return data

        fallback_data = await self._fetch_fallback(ctx)
        return fallback_data

    async def _fetch_fallback(self, ctx: Any) -> SourceData:
        fallback_state = self._states[self._fallback.name]
        try:
            data = await self._fallback.fetch(ctx)
        except SourceUnavailable as exc:
            self._record_failure(fallback_state, str(exc))
            self._maybe_log_transition(
                fallback_state,
                HealthState.FAILED,
                f"fallback unavailable: {exc}",
            )
            fallback_state.state = HealthState.FAILED
            return SourceData(
                degraded_reason=f"both sources unavailable: primary+{self._fallback.name}",
                sources_used=[],
            )
        except Exception:
            log.exception("fallback source %s raised unexpectedly", self._fallback.name)
            self._record_failure(fallback_state, "fallback_exception")
            return SourceData(
                degraded_reason="fallback raised unexpected exception",
                sources_used=[],
            )
        else:
            self._record_success(fallback_state)
            if self._fallback.name not in data.sources_used:
                data.sources_used = [*data.sources_used, self._fallback.name]
            primary_state = self._states[self._primary.name]
            if primary_state.state is HealthState.DEGRADED and not data.degraded_reason:
                data.degraded_reason = (
                    f"primary {self._primary.name} degraded; using {self._fallback.name}"
                )
            return data

    # -- state machine helpers ------------------------------------------

    def _should_try_primary(self, state: _SourceState) -> bool:
        if state.state is HealthState.HEALTHY:
            return True
        now = self._clock()
        if (now - state.last_recheck) >= self._recheck_interval_s:
            state.last_recheck = now
            return True
        return False

    def _record_success(self, state: _SourceState) -> None:
        if state.state is not HealthState.HEALTHY:
            self._maybe_log_transition(state, HealthState.HEALTHY, "recovered")
            state.state = HealthState.HEALTHY
        state.fail_streak = 0
        state.last_recheck = self._clock()

    def _record_failure(self, state: _SourceState, reason: str) -> None:
        state.fail_streak += 1
        state.last_recheck = self._clock()
        if (
            state.state is HealthState.HEALTHY
            and state.fail_streak >= self._fail_threshold
        ):
            self._maybe_log_transition(state, HealthState.DEGRADED, reason)
            state.state = HealthState.DEGRADED

    def _maybe_log_transition(
        self,
        state: _SourceState,
        target: HealthState,
        reason: str,
    ) -> None:
        if state.state is target:
            return
        log.warning(
            "source %s state %s -> %s (reason=%s, streak=%d)",
            state.name,
            state.state.value,
            target.value,
            reason,
            state.fail_streak,
        )


# ---------------------------------------------------------------------------
# Helpers used by source implementations
# ---------------------------------------------------------------------------

async def call_with_timeout(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    timeout_s: float,
    label: str,
) -> Any:
    """Run ``coro_factory()`` under :func:`asyncio.wait_for`.

    Translates :class:`asyncio.TimeoutError` to :class:`SourceUnavailable`
    so DegradeRouter treats it as a counted failure. Other exceptions
    propagate.
    """
    try:
        return await asyncio.wait_for(coro_factory(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise SourceUnavailable(f"{label}: timeout after {timeout_s}s") from exc
