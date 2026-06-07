# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Source protocol + DegradeRouter.

The reactor pulls a single :class:`SourceData` snapshot per tick. The
DegradeRouter routes to a primary source (typically
``robustness-server``) and falls back to a secondary (typically
``local_probe``) when the primary fails repeatedly. State transitions
emit one WARN log; in-state retries are silent so we do not flood the
log stream.

State machine
~~~~~~~~~~~~~

::

    HEALTHY  --(fail_streak >= fail_threshold)-->  DEGRADED
    DEGRADED --(success after recheck_interval_s)--> HEALTHY
    DEGRADED --(consecutive failures still over budget)--> FAILED  (M2+)

For M1 we use only the HEALTHY/DEGRADED two-state form: DEGRADED simply
means "use the fallback for this tick"; the next tick re-probes after
``recheck_interval_s`` has elapsed. FAILED is reserved for cases where
the fallback itself is unhealthy and the reactor should report a
degraded heartbeat (M2 wires this up).
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

    DegradeRouter catches this exception type and treats it as a
    countable failure. Other exceptions propagate so genuine bugs are
    not masked.
    """


@dataclass
class SourceData:
    """Per-tick snapshot the reactor consumes.

    Every field defaults to an empty container so downstream signals can
    treat "no data" uniformly. ``sources_used`` records which source
    produced each tick's data; ``degraded_reason`` is set when the
    primary was skipped or failed and the fallback served the request.
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
    # 2026-05-18 LocalProbe extras (A5 / A6 / A7).
    # ``local_ray`` carries ``{"healthy": bool, "reason": str, "stderr": str,
    # "returncode": int}`` from :func:`local_probe._probe_ray_head`.
    # ``local_fd`` carries ``{"pid": int, "used": int, "limit": int,
    # "used_pct": float}`` from :func:`local_probe._sample_fd_usage`.
    # ``local_aiter_jit`` carries ``{"jit_dir": str, "so_count": int,
    # "build_count": int}`` from :func:`local_probe._sample_aiter_jit`;
    # the cross-tick regression detector reads it.
    local_ray: dict[str, Any] = field(default_factory=dict)
    local_fd: dict[str, Any] = field(default_factory=dict)
    local_aiter_jit: dict[str, Any] = field(default_factory=dict)
    # G-section decision-audit probe (2026-05-18). Carries persisted
    # decision artefacts:
    #   ``recent_integrate`` — last N integrate result.json entries
    #     (decision/gain_pct/patch_path/patch_size_bytes/base_tput/new_tput/
    #      kernel_id/dispatched_count/ts).
    #   ``ci_metrics`` — most recent ci_metrics{,_final}.json content as-is
    #     (or ``{}`` when absent — the file is produced by an external
    #     report_back system, not Hyperloom proper).
    #   ``oob_attempts`` — tail of optimization_attempts.jsonl entries
    #     (kernel_id / backend / report_text / microbench_speedup).
    local_decision_audit: dict[str, Any] = field(default_factory=dict)
    # C-section preflight inputs (2026-05-19). The two slots are read by
    # ``signals/preflight.py`` to detect impossible model-GPU configs,
    # low Amdahl ceilings for kernel optimization, and cold-start vs
    # remaining-budget mismatches.
    #
    # ``local_manifest`` is the raw ``manifest.json`` dict written by
    # ``inference_optimizer/manifest.py`` at session boot — empty when
    # absent (resume from broken session, or non-Hyperloom host).
    #
    # ``local_kernel_breakdown`` aggregates
    # ``<session>/profiles/kernel_breakdown.json`` into the projection
    # ``{tier_pcts: {T1_TRITON, T2_AITER_CK, ...}, total_kernels,
    # total_gpu_pct, mtime}``. Empty when the profile action has not
    # produced a breakdown yet.
    local_manifest: dict[str, Any] = field(default_factory=dict)
    local_kernel_breakdown: dict[str, Any] = field(default_factory=dict)
    # E-section critic health (2026-05-19). Carries:
    #   ``recent_judges`` — last N ``critic-workdir/<turn>/judge_bundle.json``
    #     entries projected to ``{turn_dir, kb_read_skipped_reason,
    #     proposal_count, mtime}``.
    #   ``workdir_count`` — total ``<turn>/`` subdirs under ``critic-workdir/``
    #     (used by E4 ``critic_prune_stuck``).
    local_critic_health: dict[str, Any] = field(default_factory=dict)
    # I-section state-integrity probe (2026-05-19). Aggregates the
    # five C-side state slots into one payload so a single LocalProbe
    # tick scans them all without re-doing IO per signal:
    #   ``state_json`` — ``{valid, size_bytes, mtime, error}`` for state.json.
    #   ``wal``        — ``{wal_bytes, db_bytes, db_path}`` for coordinator.db-wal.
    #   ``leases``     — list of ``{task_id, holder_pid, alive, ts}`` rows
    #                    (cross-referenced against ``os.kill(pid, 0)``).
    #   ``agents``     — ``{<role>: {inbox_bytes, outbox_bytes}}``.
    #   ``coordinator``— ``{recorded_pid, alive, pid_file}`` (from
    #                    ``optimizer_runs/run_*.pid``).
    local_state_integrity: dict[str, Any] = field(default_factory=dict)
    # J-section external-deps probe (2026-05-19). Carries:
    #   ``gateway``    — ``{url, reachable, status_code, error}`` for
    #                    ``OPENAI_BASE_URL/models``.
    #   ``mounts``     — list of ``{path, latency_ms, ok, error}`` for
    #                    ``$TRACELENS_ROOT`` / ``$TRACELENS_INTERNAL_ROOT`` /
    #                    ``$INFERENCEX_PATH`` / ``$OOB_SRC``.
    #   ``tracelens_cli`` — ``{cli_names, found, any_present}`` — whether
    #                    each CLI name is found via ``shutil.which``.
    local_external_deps: dict[str, Any] = field(default_factory=dict)
    coordinator_events: list[dict[str, Any]] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    degraded_reason: str | None = None

    def merge_from(self, other: "SourceData") -> None:
        """Merge another snapshot in, preserving non-empty existing fields.

        Used by callers that want to enrich a primary result with extra
        fallback data; M1 does not exercise this path but keeps the
        helper available for M2 ``signals/*`` evolution.
        """
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
                # Treat unexpected errors as failures too, but re-raise
                # if the source is supposed to never raise: per Source
                # contract, unexpected exceptions go through but still
                # count as failures so we eventually degrade.
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
