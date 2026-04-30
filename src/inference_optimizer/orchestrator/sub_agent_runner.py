"""SubAgentRunner — DESIGN §11.

Spawns a Claude / Codex sub-agent for one delegated :class:`Task` (kind=
``delegate``):

    queued
       │   SubAgentRunner.run() picks it up
       ▼
    running   ←── lanes acquired, prompt composed, backend.run started
       │
       ├── intents parsed → metrics extracted → succeeded
       ├── backend exception            → failed
       ├── policy denial                → safely_failed (no retry)
       └── lane contention timeout      → failed (retryable)

In v0.6 the runner is a *dry skeleton*: it reuses the Conductor's existing
backend instance instead of spawning a real OOB process. That is enough to
prove the lane / prompt / parse pipeline end-to-end. Real OOB spawning lands
with the MCP custom tool registration in F4 and a per-sub-agent backend
factory in Phase 7.

The Conductor side wires this in via a periodic dispatcher loop
(:meth:`Conductor._delegator`) that polls ``tasks`` for ``kind=delegate
state=queued`` rows.

References:
    - DESIGN §11   SubAgentRunner lifecycle
    - DESIGN §13.4 Task state machine
    - DESIGN §3.5.5 Lane / lease back-off
    - IMPLEMENTATION-CHECKLIST.md Phase 4 §4.9 — §4.21
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .action_executors import (
    EXECUTOR_REGISTRY,
    ExecutorContext,
    ExecutorEnvError,
    get_executor,
)
from .intent_parser import Intent, IntentType

if TYPE_CHECKING:  # pragma: no cover
    from .action_executors.base import ActionExecutor, ExecutorResult
    from .action_registry import ActionMetadata, ActionRegistry
    from .backends.base import Backend
    from .policy import PolicyGate
    from .resource_lock import Lease, ResourceLockManager
    from .task_registry import Task, TaskRegistry


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class TaskResult:
    """Return value of :meth:`SubAgentRunner.run`."""

    task_id: str
    status: str  # "succeeded" | "failed" | "safely_failed" | "needs_manual_review"
    metrics: dict[str, Any]
    artifacts: list[str]
    intents: list[Intent]
    notes: str = ""


class NeedsManualReviewError(RuntimeError):
    """Raised when the sub-agent ran but produced no actionable intent."""


# ---------------------------------------------------------------------------
class SubAgentRunner:
    """Owns sub-agent invocation lifecycle for one delegated task.

    Construction takes the production wiring (backend, policy, locks).
    Pass ``backend=None`` only in unit tests where an inert runner is fine.
    """

    DEFAULT_LEASE_TTL_SEC = 1800

    def __init__(
        self,
        *,
        backend: "Backend",
        policy: "PolicyGate",
        locks: "ResourceLockManager",
        action_registry: "ActionRegistry | None",
        tasks: "TaskRegistry | None" = None,
        workspace: Path | None = None,
        agent_name: str = "sub-agent",
        env: dict[str, str] | None = None,
        executor_registry: dict[str, "ActionExecutor"] | None = None,
        intent_sink: "Callable[[str, Intent], Awaitable[None]] | None" = None,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.locks = locks
        self.actions = action_registry
        self.tasks = tasks
        self.workspace = Path(workspace) if workspace else None
        self.agent_name = agent_name
        # ``env`` is forwarded to executors so they can read MODEL / TP /
        # CONC / OOB_API_KEY / etc. Falls back to ``os.environ`` when None
        # (kept None-safe for the dry-run / test paths).
        import os
        self.env: dict[str, str] = dict(env) if env is not None else dict(os.environ)
        # Test seam: pass a custom registry to mock executors. ``None``
        # means use the global EXECUTOR_REGISTRY (the production path).
        self._executor_registry_override = executor_registry
        # When set, executor-emitted intents flow through this callback
        # before becoming part of TaskResult. The Conductor wires this
        # to its own _gate_intent + _handle_intent pipeline so executor
        # intents end up on the events bus exactly like LLM intents.
        self._intent_sink = intent_sink

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    async def run(self, task: "Task") -> TaskResult:
        """Run one delegated task end-to-end.

        Order of operations:

            1. Resolve action metadata. Unknown action -> safely_failed.
            2. Acquire required lanes (back-off via ResourceLockManager).
            3. Compose prompt: action.md + task.params + IronRules header.
            4. backend.run(prompt) bounded by action.max_turns.
            5. Parse returned intents -> metrics.
            6. Release lanes (always, in `finally`).
            7. Optionally update TaskRegistry state (queued -> running ->
               succeeded/failed/safely_failed).
        """
        action_name = self._action_name_from_task(task)
        action = self._resolve_action(action_name)
        if action is None:
            note = f"unknown action {action_name!r} (no metadata)"
            log.info("sub-agent: %s", note)
            # State machine requires queued -> running -> failed; we move
            # through `running` so the failure is recorded with full history.
            await self._mark(task, "running", evidence={"note": "unknown_action_probe"})
            await self._mark(task, "failed", evidence={"note": note})
            return TaskResult(
                task_id=task.task_id,
                status="safely_failed",
                metrics={},
                artifacts=[],
                intents=[],
                notes=note,
            )

        lanes = list(action.requires_lanes) or list(task.requires_lanes)
        ttl_sec = action.lease_ttl_sec or task.lease_ttl_sec or self.DEFAULT_LEASE_TTL_SEC
        await self._mark(task, "running")

        lease: "Lease | None" = None
        try:
            if lanes:
                try:
                    lease = await self.locks.acquire(
                        lanes,
                        holder_id=f"{self.agent_name}-{uuid.uuid4().hex[:8]}",
                        task_id=task.task_id,
                        action=action.name,
                        ttl_sec=ttl_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.info(
                        "sub-agent: lane contention for %s lanes=%s: %s",
                        action.name, lanes, exc,
                    )
                    await self._mark(
                        task, "failed",
                        evidence={"reason": "lane_contention", "error": repr(exc)},
                    )
                    return TaskResult(
                        task_id=task.task_id,
                        status="failed",
                        metrics={},
                        artifacts=[],
                        intents=[],
                        notes=f"lane_contention lanes={lanes}",
                    )

            # Two paths: prefer the Python ``ActionExecutor`` (real
            # subprocess/GPU work). Fall back to the LLM-driven path
            # when no executor is registered or when the executor signals
            # missing env / artefacts via ``ExecutorEnvError``.
            executor = self._lookup_executor(action.name)
            if executor is not None:
                exec_result = await self._try_executor(
                    executor=executor, task=task, action=action,
                    lanes_held=lanes,
                )
                if exec_result is not None:
                    return exec_result
                # ExecutorEnvError → fall through to LLM path

            prompt = self._compose_prompt(task, action)
            allowed_tools = tuple(action.allowed_tools or ("emit_intent",))
            try:
                intents = await self.backend.run(
                    prompt,
                    agent_name=self.agent_name,
                    allowed_tools=allowed_tools,
                    max_turns=int(action.max_turns or 30),
                    extra={
                        "task_id": task.task_id,
                        "action": action.name,
                        "params": dict(task.params or {}),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "sub-agent: backend failed for %s: %s",
                    action.name, exc,
                )
                await self._mark(
                    task, "failed",
                    evidence={"reason": "backend_error", "error": repr(exc)},
                )
                return TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    metrics={},
                    artifacts=[],
                    intents=[],
                    notes=f"backend_error: {exc!r}",
                )

            metrics = self._extract_metrics(intents)
            artifacts = self._extract_artifacts(intents)
            note = "ok" if intents else "no intents emitted"
            status = "succeeded" if intents else "needs_manual_review"
            await self._mark(
                task,
                "succeeded" if status == "succeeded" else "needs_manual_review",
                evidence={"intent_count": len(intents), "metrics": metrics},
            )
            return TaskResult(
                task_id=task.task_id,
                status=status,
                metrics=metrics,
                artifacts=artifacts,
                intents=intents,
                notes=note,
            )
        finally:
            if lease is not None:
                try:
                    await self.locks.release(lease)
                except Exception:  # noqa: BLE001 — best-effort
                    log.exception("sub-agent: failed to release lanes for %s", task.task_id)

    # ------------------------------------------------------------------
    # Executor (Python ↔ shell bridge) integration
    # ------------------------------------------------------------------
    def _lookup_executor(self, action_name: str) -> "ActionExecutor | None":
        """Return a registered :class:`ActionExecutor` or ``None``.

        Honours :attr:`_executor_registry_override` so tests can inject
        a mock registry without touching the global one.
        """
        if self._executor_registry_override is not None:
            from .action_executors.base import _normalize
            return self._executor_registry_override.get(_normalize(action_name))
        return get_executor(action_name)

    async def _try_executor(
        self,
        *,
        executor: "ActionExecutor",
        task: "Task",
        action: "ActionMetadata",
        lanes_held: list[str],
    ) -> TaskResult | None:
        """Try the executor. Return a finalised :class:`TaskResult` on
        success/failure, or ``None`` when ``ExecutorEnvError`` signals
        the runner should fall back to the LLM path.

        Note: we keep the lease across both paths — the caller's
        ``finally`` releases it. Only one path actually shells out per
        task; callers don't need to worry about double-counting.
        """
        ctx = ExecutorContext(
            task=task,
            action_meta=action,
            lanes_held=list(lanes_held),
            session_dir=Path(self.workspace) if self.workspace else Path.cwd(),
            env=dict(self.env),
        )
        try:
            result = await executor.run(ctx)
        except ExecutorEnvError as exc:
            log.info(
                "sub-agent: executor %s opted out (env): %s — "
                "falling back to LLM path",
                action.name, exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001 — unexpected runtime crash
            log.exception(
                "sub-agent: executor %s crashed: %s",
                action.name, exc,
            )
            await self._mark(
                task, "failed",
                evidence={"reason": "executor_crash", "error": repr(exc)},
            )
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                metrics={},
                artifacts=[],
                intents=[],
                notes=f"executor_crash: {exc!r}",
            )

        # Persist the executor's intents to the bus before we return so
        # reactor cursors advance. We do this via the optional
        # ``intent_sink`` callback (wired by Conductor to its own
        # PolicyGate + _handle_intent pipeline), so executor-emitted
        # intents get exactly the same gating as LLM-emitted ones.
        if self._intent_sink is not None:
            for intent in result.intents:
                try:
                    await self._intent_sink(self.agent_name, intent)
                except Exception:  # noqa: BLE001 — best-effort
                    log.exception(
                        "sub-agent: intent_sink failed for %s",
                        intent.type.value,
                    )
        await self._mark(
            task, result.status if result.status in (
                "succeeded", "failed", "needs_manual_review"
            ) else "needs_manual_review",
            evidence={
                "via": "executor",
                "rc": result.rc,
                "intent_count": len(result.intents),
                "metrics": dict(result.metrics),
                "notes": result.notes,
            },
        )
        return TaskResult(
            task_id=task.task_id,
            status=result.status,
            metrics=dict(result.metrics),
            artifacts=list(result.artifacts),
            intents=list(result.intents),
            notes=result.notes,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _action_name_from_task(task: "Task") -> str:
        params = task.params or {}
        if isinstance(params, dict):
            name = params.get("action_name", "")
            return str(name) if name is not None else ""
        return ""

    def _resolve_action(self, action_name: str) -> "ActionMetadata | None":
        if not action_name:
            return None
        if self.actions is None:
            return None
        return self.actions.get(action_name)

    def _compose_prompt(
        self, task: "Task", action: "ActionMetadata"
    ) -> str:
        body = ""
        if self.actions is not None:
            body = self.actions.system_prompt_for(action.name) or ""
        params = dict((task.params or {}).get("params", {}) or {})
        return (
            f"# Sub-agent: {action.name}\n"
            f"## Task params\n{params!r}\n\n"
            f"## Action spec\n{body}\n"
            f"## Lanes held\n{', '.join(action.requires_lanes) or '(none)'}\n"
            f"## Allowed tools\n{', '.join(action.allowed_tools) or 'emit_intent'}\n"
        )

    @staticmethod
    def _extract_metrics(intents: list[Intent]) -> dict[str, Any]:
        """Pull useful numeric facts out of the sub-agent's intent stream.

        We currently honour:
            * ``update_state.changes.current_tput``   -> metrics["tput"]
            * ``propose_action.predicted_gain_pct``   -> metrics["predicted_gain_pct"]
        """
        out: dict[str, Any] = {}
        for it in intents:
            payload = it.payload or {}
            if it.type == IntentType.UPDATE_STATE:
                changes = payload.get("changes") or {}
                if "current_tput" in changes:
                    out["tput"] = changes["current_tput"]
            elif it.type == IntentType.PROPOSE_ACTION:
                gain = payload.get("predicted_gain_pct")
                if gain is not None:
                    out.setdefault("predicted_gain_pct", gain)
        return out

    @staticmethod
    def _extract_artifacts(intents: list[Intent]) -> list[str]:
        out: list[str] = []
        for it in intents:
            payload = it.payload or {}
            for key in ("artifact_path", "result_path", "log_path"):
                v = payload.get(key) if isinstance(payload, dict) else None
                if isinstance(v, str):
                    out.append(v)
        return out

    async def _mark(
        self,
        task: "Task",
        new_state: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if self.tasks is None:
            return
        try:
            await self.tasks.transition(task.task_id, new_state, evidence=evidence)
        except Exception:  # noqa: BLE001 — task may already be in terminal state
            log.exception(
                "sub-agent: failed to transition task=%s -> %s",
                task.task_id, new_state,
            )


# ---------------------------------------------------------------------------
async def dispatch_pending_delegates(
    runner: SubAgentRunner,
    *,
    db,
    poll_interval_s: float = 0.5,
    stop: asyncio.Event | None = None,
    on_task_done: "Callable[[Task, TaskResult], Awaitable[None] | None] | None" = None,
) -> int:
    """Background pump: drain queued delegate tasks via ``runner.run``.

    Returns the number of tasks dispatched (useful for tests). The dispatcher
    runs until ``stop`` is set OR no more queued delegates are visible *and*
    ``stop`` is None (one-shot mode used by tests).

    If ``on_task_done`` is provided it is invoked after each ``runner.run``
    completes (success or failure) with ``(task, task_result)``. This is the
    Conductor's hook for SharedState updates and follow-up scheduling.
    """
    import inspect

    dispatched = 0
    while True:
        rows = await db.fetchall(
            "SELECT * FROM tasks WHERE kind=? AND state=? ORDER BY created_at",
            ("delegate", "queued"),
        )
        if not rows:
            if stop is None:
                return dispatched
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)
                return dispatched
            except asyncio.TimeoutError:
                continue
        from .task_registry import Task  # local to avoid import cycle

        for row in rows:
            task = Task.from_row(row)
            result = await runner.run(task)
            if on_task_done is not None:
                try:
                    out = on_task_done(task, result)
                    if inspect.isawaitable(out):
                        await out
                except Exception:  # noqa: BLE001 — never let the hook break dispatch
                    log.exception("on_task_done hook failed for %s", task.task_id)
            dispatched += 1
        if stop is not None and stop.is_set():
            return dispatched


__all__ = [
    "NeedsManualReviewError",
    "SubAgentRunner",
    "TaskResult",
    "dispatch_pending_delegates",
]
