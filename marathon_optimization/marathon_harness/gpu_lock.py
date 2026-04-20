"""gpu_lock.py — GPU resource coordination between orchestrator and kernel manager.

Problem: the marathon runs 3 GPU-hungry agents concurrently:
  - Orchestrator merge-ops (kill server → rebuild → restart → benchmark)
  - Kernel Manager local tests (compile → correctness → micro-bench)
  - rocprof profiling

Without coordination, these stomp on each other. CK/HIP kernel compilation
needs the GPU device, the server holds all GPUs, and benchmarks need the
server running.

Solution: a shared asyncio lock with:
  - Typed phases so agents communicate intent
  - Heartbeat-aware watchdog that respects active holders
  - Fairness tracking so no agent starves
  - Wait-with-timeout so KM blocks briefly instead of always deferring

Heartbeat contract:
  While the lock is held, an auto-heartbeat coroutine pings every
  HEARTBEAT_INTERVAL_S seconds.  If the event loop is alive (i.e. the
  holder's awaits are progressing), the heartbeat stays fresh.  Other
  agents and the watchdog use ``holder_alive`` to distinguish "slow but
  working" from "dead / hung".  Force-release only fires when the
  heartbeat goes stale — never while the holder is doing useful work.

Usage:
    gpu = GpuLock()
    async with gpu.acquire("server", "orchestrator"):
        await run_benchmark()
    async with gpu.acquire("compile", "kernel-manager"):
        await rebuild_aiter()
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)

PHASE_PRIORITY = {
    "compile": 0,
    "rebuild": 1,
    "server-start": 2,
    "benchmark": 3,
    "micro-bench": 4,
    "profile": 5,
    "local-test": 6,
    "idle": 99,
}

# Soft max hold time per phase.  The watchdog only force-releases when
# the holder's heartbeat is ALSO stale (see HEARTBEAT_STALE_S).
MAX_HOLD_S: dict[str, float] = {
    "compile": 300,
    "rebuild": 900,
    "server-start": 900,
    "benchmark": 600,
    "micro-bench": 600,
    "profile": 300,
    "local-test": 1800,
}
DEFAULT_MAX_HOLD_S = 600

# ── Heartbeat constants ──────────────────────────────────────────────
HEARTBEAT_INTERVAL_S = 30      # auto-heartbeat fires this often
HEARTBEAT_STALE_S = 120        # no heartbeat for this long → "stale"
HEARTBEAT_HARD_CAP_MULT = 3    # absolute max = MAX_HOLD_S * this

# After this many consecutive deferrals, the KM should block-wait
# instead of deferring again.
DEFER_PATIENCE = 3


@dataclass
class GpuState:
    """Snapshot of current GPU usage."""
    phase: str = "idle"
    holder: str = ""
    acquired_at: float = 0.0
    server_running: bool = False
    server_pid: int = 0

    @property
    def busy(self) -> bool:
        return self.phase != "idle"

    @property
    def held_seconds(self) -> float:
        return time.monotonic() - self.acquired_at if self.busy else 0.0


class GpuLock:
    """Shared GPU resource lock for multi-agent coordination.

    Key guarantees:
      1. Only one agent holds the GPU at a time.
      2. A max-hold watchdog prevents permanent lock-out from crashes.
      3. Fairness: after DEFER_PATIENCE consecutive deferrals, an agent
         switches from non-blocking check to blocking wait.
      4. Utilization history for dashboard/debugging.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state = GpuState()
        self._waiters: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._defer_counts: dict[str, int] = {}
        self._watchdog_task: asyncio.Task[None] | None = None
        self._stale = False
        self._stale_count = 0
        self._stale_logged_holder: str = ""
        self._holder_task: asyncio.Task[Any] | None = None
        self._recent_contention_ts: float = 0.0
        # Heartbeat state
        self._last_heartbeat: float = 0.0
        self._auto_heartbeat_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> GpuState:
        return self._state

    # ── Heartbeat API ─────────────────────────────────────────────────

    def heartbeat(self) -> None:
        """Signal that the lock holder is alive and doing useful work.

        Called automatically by the auto-heartbeat task, but holders may
        also call this explicitly for tighter liveness guarantees.
        """
        self._last_heartbeat = time.monotonic()

    @property
    def seconds_since_heartbeat(self) -> float:
        """Seconds since the last heartbeat.  Falls back to held_seconds
        when no heartbeat has been recorded (pre-acquire)."""
        if self._last_heartbeat <= 0:
            return self._state.held_seconds
        return time.monotonic() - self._last_heartbeat

    @property
    def holder_alive(self) -> bool:
        """True if the lock is held AND the holder has heartbeated recently."""
        return self._state.busy and self.seconds_since_heartbeat < HEARTBEAT_STALE_S

    async def _auto_heartbeat(self) -> None:
        """Background coroutine that heartbeats every HEARTBEAT_INTERVAL_S.

        Proves the event loop is alive.  If the holder blocks the event
        loop (non-async blocking call, segfault, etc.) this task stops
        firing, seconds_since_heartbeat grows, and the watchdog can
        safely force-release.
        """
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                self._last_heartbeat = time.monotonic()
        except asyncio.CancelledError:
            pass

    def start_watchdog(self) -> None:
        """Start the background watchdog that auto-releases abandoned locks."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        """Periodically check if the lock holder has exceeded max hold time.

        Three tiers:
          1. Under MAX_HOLD_S → no action, everything is fine.
          2. Over MAX_HOLD_S but heartbeat alive → warn, don't force.
             The holder is doing useful work; other agents should wait.
          3. Over MAX_HOLD_S and heartbeat stale → force-release.
             The holder is likely dead or hung.
          Hard cap: Over MAX_HOLD_S * HEARTBEAT_HARD_CAP_MULT →
             force-release regardless (prevents infinite hold).
        """
        while True:
            await asyncio.sleep(10)
            if not self._state.busy:
                self._stale_count = 0
                self._stale_logged_holder = ""
                continue

            max_s = MAX_HOLD_S.get(self._state.phase, DEFAULT_MAX_HOLD_S)
            held = self._state.held_seconds
            hb_age = self.seconds_since_heartbeat
            hb_alive = hb_age < HEARTBEAT_STALE_S
            hard_cap = max_s * HEARTBEAT_HARD_CAP_MULT
            holder_key = f"{self._state.holder}/{self._state.phase}"

            if held <= max_s:
                self._stale_count = 0
                continue

            if hb_alive and held < hard_cap:
                if holder_key != self._stale_logged_holder:
                    log.info(
                        "GPU WATCHDOG: %s over soft limit (%.0fs > %ds) but "
                        "heartbeat alive (%.0fs ago) — waiting cooperatively",
                        holder_key, held, max_s, hb_age,
                    )
                    self._stale_logged_holder = holder_key
                self._stale_count = 0
                continue

            # Heartbeat stale OR hard cap exceeded → force-release
            if holder_key != self._stale_logged_holder:
                reason = "heartbeat stale" if not hb_alive else "hard cap exceeded"
                log.error(
                    "GPU WATCHDOG: %s held lock for %.0fs (max %ds), %s "
                    "(heartbeat %.0fs ago) — force-releasing",
                    holder_key, held, max_s, reason, hb_age,
                )
                self._stale_logged_holder = holder_key
                self._stale_count = 0
            self._stale_count += 1
            self._force_release()
            if self._stale_count >= 6 and self._holder_task is not None:
                log.error(
                    "GPU WATCHDOG: %s stale for %d checks (%.0fs) — cancelling holder task",
                    holder_key, self._stale_count, held,
                )
                self._holder_task.cancel()
                self._holder_task = None
            elif self._stale_count % 6 == 0:
                log.warning("GPU WATCHDOG: %s still stale (%d checks, %.0fs)",
                            holder_key, self._stale_count, held)

    def _force_release(self) -> None:
        """Mark lock as stale so holder's finally block knows to skip its work.

        We do NOT release the asyncio.Lock from a different task — that would
        let two tasks into the critical section simultaneously (P0 bug).
        Instead we set a _stale flag; the holder's finally block checks it
        and does a quick exit. If the holder's task is truly dead (cancelled),
        asyncio's Lock implementation releases on task destruction.
        """
        if self._lock.locked():
            if not self._stale:
                self._history.append({
                    "holder": self._state.holder,
                    "phase": self._state.phase,
                    "duration_s": round(self._state.held_seconds, 1),
                    "timestamp": time.time(),
                    "force_released": True,
                })
            self._stale = True
            self._recent_contention_ts = time.monotonic()

    @asynccontextmanager
    async def acquire(
        self,
        phase: str,
        holder: str = "",
        timeout_s: float = 600,
    ) -> AsyncIterator[GpuState]:
        """Acquire the GPU for a given phase. Blocks until available or timeout."""
        if phase not in PHASE_PRIORITY:
            phase = "idle"

        waiter_info = {
            "phase": phase,
            "holder": holder,
            "requested_at": time.monotonic(),
        }
        self._waiters.append(waiter_info)

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError:
            if waiter_info in self._waiters:
                self._waiters.remove(waiter_info)
            log.error(
                "GPU lock timeout (%ds) for %s/%s — current holder: %s/%s (held %.0fs)",
                timeout_s, holder, phase,
                self._state.holder, self._state.phase,
                self._state.held_seconds,
            )
            # Only force-release from timeout handler if heartbeat is stale
            max_s = MAX_HOLD_S.get(self._state.phase, DEFAULT_MAX_HOLD_S)
            if self._state.held_seconds > max_s and not self.holder_alive:
                log.warning("Force-releasing stale lock from timeout handler (heartbeat stale)")
                self._force_release()
            elif self.holder_alive:
                log.info("Lock holder is alive (heartbeat %.0fs ago) — not force-releasing",
                         self.seconds_since_heartbeat)
            raise RuntimeError(
                f"GPU lock timeout: {holder}/{phase} waited {timeout_s}s, "
                f"held by {self._state.holder}/{self._state.phase}"
            )

        if waiter_info in self._waiters:
            self._waiters.remove(waiter_info)
        prev_phase = self._state.phase
        self._state.phase = phase
        self._state.holder = holder
        self._state.acquired_at = time.monotonic()

        # Reset defer count for this holder since they got the lock
        self._defer_counts[holder] = 0

        log.info("GPU acquired: %s/%s (was %s, waited %.1fs)",
                 holder, phase, prev_phase,
                 time.monotonic() - waiter_info["requested_at"])

        self._stale = False
        self._stale_count = 0
        self._stale_logged_holder = ""
        try:
            self._holder_task = asyncio.current_task()
        except RuntimeError:
            self._holder_task = None

        # Start auto-heartbeat to prove the event loop is alive
        self._last_heartbeat = time.monotonic()
        self._auto_heartbeat_task = asyncio.create_task(self._auto_heartbeat())

        try:
            yield self._state
        finally:
            # Stop auto-heartbeat
            if self._auto_heartbeat_task and not self._auto_heartbeat_task.done():
                self._auto_heartbeat_task.cancel()
                self._auto_heartbeat_task = None
            self._last_heartbeat = 0.0

            held_s = time.monotonic() - self._state.acquired_at
            was_stale = self._stale
            self._history.append({
                "holder": holder,
                "phase": phase,
                "duration_s": round(held_s, 1),
                "timestamp": time.time(),
                "force_released": was_stale,
            })
            if len(self._history) > 200:
                self._history = self._history[-200:]

            if was_stale:
                log.warning("GPU released (was stale): %s/%s (held %.1fs)", holder, phase, held_s)
            else:
                log.info("GPU released: %s/%s (held %.1fs)", holder, phase, held_s)
            self._state.phase = "idle"
            self._state.holder = ""
            self._state.acquired_at = 0.0
            self._stale = False
            self._lock.release()

    async def wait_or_defer(
        self,
        phase: str,
        holder: str,
        quick_timeout_s: float = 30,
        full_timeout_s: float = 300,
    ) -> bool:
        """Smart wait: defer quickly at first, block-wait after repeated deferrals.

        Returns True if the lock was acquired (caller must release via the
        context manager). Returns False if the caller should defer.

        This is the main entry point for the KM's "should I wait or skip?" logic.
        After DEFER_PATIENCE consecutive deferrals for this holder, switches
        from quick non-blocking check to a real blocking wait.
        """
        defers = self._defer_counts.get(holder, 0)

        if defers >= DEFER_PATIENCE:
            log.info("GPU: %s has deferred %d times, switching to blocking wait (up to %ds)",
                     holder, defers, full_timeout_s)
            try:
                await asyncio.wait_for(
                    self._wait_until_free(), timeout=full_timeout_s
                )
                self._defer_counts[holder] = 0
                return True
            except asyncio.TimeoutError:
                log.warning("GPU: %s blocking wait timed out after %ds", holder, full_timeout_s)
                self._defer_counts[holder] = 0
                return False

        # Quick check first
        if not self._state.busy:
            return True

        # GPU is busy — try a short wait
        if quick_timeout_s > 0:
            try:
                await asyncio.wait_for(
                    self._wait_until_free(), timeout=quick_timeout_s
                )
                return True
            except asyncio.TimeoutError:
                pass

        # Still busy — record deferral
        self._defer_counts[holder] = defers + 1
        log.info("GPU: %s deferred (%d/%d), holder=%s/%s held %.0fs",
                 holder, self._defer_counts[holder], DEFER_PATIENCE,
                 self._state.holder, self._state.phase,
                 self._state.held_seconds)
        return False

    async def _wait_until_free(self) -> None:
        """Poll until the GPU state is idle."""
        while self._state.busy:
            await asyncio.sleep(1)

    @property
    def is_stale(self) -> bool:
        """True if the watchdog marked the current hold as stale."""
        return self._stale

    def had_recent_contention(self, window_s: float = 300) -> bool:
        """True if the GPU had stale-lock contention within the last window_s seconds."""
        if self._recent_contention_ts <= 0:
            return False
        return (time.monotonic() - self._recent_contention_ts) < window_s

    def record_deferral(self, holder: str) -> int:
        """Manually record a deferral and return the current count."""
        self._defer_counts[holder] = self._defer_counts.get(holder, 0) + 1
        return self._defer_counts[holder]

    def reset_deferrals(self, holder: str) -> None:
        """Reset deferral count after a successful GPU operation."""
        self._defer_counts[holder] = 0

    def server_up(self, pid: int = 0) -> None:
        self._state.server_running = True
        self._state.server_pid = pid

    def server_down(self) -> None:
        self._state.server_running = False
        self._state.server_pid = 0

    @property
    def server_running(self) -> bool:
        return self._state.server_running

    def is_available_for(self, phase: str) -> bool:
        """Non-blocking check: can this phase run right now?"""
        return not self._state.busy

    def pending_waiters(self) -> list[dict[str, Any]]:
        return list(self._waiters)

    def utilization_summary(self) -> dict[str, Any]:
        current_info = {
            "phase": self._state.phase,
            "holder": self._state.holder,
            "held_s": round(self._state.held_seconds, 1),
            "heartbeat_age_s": round(self.seconds_since_heartbeat, 1),
            "holder_alive": self.holder_alive,
        }
        if not self._history:
            return {"total_held_s": 0, "phase_breakdown": {}, "entries": 0,
                    "force_releases": 0, "defer_counts": dict(self._defer_counts),
                    "current": current_info}

        total = sum(e["duration_s"] for e in self._history)
        by_phase: dict[str, float] = {}
        force_releases = 0
        for e in self._history:
            by_phase[e["phase"]] = by_phase.get(e["phase"], 0) + e["duration_s"]
            if e.get("force_released"):
                force_releases += 1

        return {
            "total_held_s": round(total, 1),
            "phase_breakdown": {k: round(v, 1) for k, v in sorted(by_phase.items())},
            "entries": len(self._history),
            "force_releases": force_releases,
            "defer_counts": dict(self._defer_counts),
            "current": current_info,
        }
