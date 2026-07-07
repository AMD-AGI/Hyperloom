# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import asyncio
import hashlib
import json
import os
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any
from inference_optimizer.protocol.intent import Intent, IntentType
from ..bus.message_bus import Message
from ..policy.gate import (
    PolicyDenied,
)
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)

from .coordinator import (
    _format_inbox_event,
)
import logging as _logging
log = _logging.getLogger(__name__)


class InlineActionsCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _inline_action_whitelist(self) -> frozenset[str]:
        """Derive the set of actions safe to run inline (A3): lane-light, registered executor, not in _INLINE_ACTION_DENY. PolicyGate remains the real security boundary.

        Returns:
            A frozenset of action names eligible for inline execution; empty
            when no action registry is loaded.
        """
        reg = getattr(self, "action_registry", None)
        if reg is None:
            return frozenset()
        executors = getattr(self.sub, "executor_registry", {}) or {}
        names_fn = getattr(reg, "names", None)
        try:
            names = list(names_fn()) if callable(names_fn) else []
        except Exception:  # noqa: BLE001 — defensive
            names = []
        allowed: set[str] = set()
        for name in names:
            if name in self._INLINE_ACTION_DENY:
                continue
            if name not in executors:
                continue
            lanes, _ttl = self._registry_lanes_ttl(name)
            if lanes:
                continue
            allowed.add(name)
        return frozenset(allowed)

    def _run_action_now_sync(
        self,
        action_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Bridge callable for the ``run_action_now`` context tool (A3): marshals the executor coroutine onto the Coordinator loop and blocks with a timeout.

        Args:
            action_name: Name of the action to run inline; must be
                inline-eligible per :meth:`_inline_action_whitelist`.
            params: Optional parameter mapping forwarded to the executor.

        Returns:
            A human-readable status string describing the inline run outcome,
            disablement, ineligibility, timeout, or error.
        """
        if not self._inline_fast_actions_enabled:
            return (
                "(run_action_now disabled: set "
                "INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS to a non-off value "
                "to enable; use emit_intent delegate for async execution)"
            )
        name = (action_name or "").strip()
        if not name:
            return "(run_action_now: action_name required)"
        whitelist = self._inline_action_whitelist()
        if name not in whitelist:
            return (
                f"(run_action_now: {name!r} is not inline-eligible — only "
                f"cheap, lane-light actions may run inline: "
                f"{sorted(whitelist)}. Use emit_intent delegate to run it "
                f"asynchronously.)"
            )
        loop = self._coordinator_loop
        if loop is None or loop.is_closed():
            return "(run_action_now unavailable: coordinator loop not running)"
        coro = self._run_action_now(name, dict(params or {}))
        # Cap inline wait under backend timeout so a slow action can't wedge the turn.
        try:
            timeout_s = float(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S",
                    "120",
                )
                or 120
            )
        except (TypeError, ValueError):
            timeout_s = 120.0
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as exc:
            return f"(run_action_now: could not schedule on coordinator loop: {exc!r})"
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            return (
                f"(run_action_now: {name!r} still running after "
                f"{timeout_s:.0f}s; it keeps running asynchronously — check "
                "get_recent_outcomes or the next-tick inbox for its "
                "delegated_result)"
            )
        except Exception as exc:  # noqa: BLE001 — never crash the turn
            log.exception("run_action_now: inline run of %r failed", name)
            return f"(run_action_now: {name!r} errored: {exc!r})"

    async def _run_action_now(
        self,
        action_name: str,
        params: dict[str, Any],
    ) -> str:
        """Coordinator-loop coroutine that runs a whitelisted action inline through PolicyGate + SubAgentRunner, publishing a delegated_result for audit/inbox parity.

        Args:
            action_name: Name of the action to execute.
            params: Parameter mapping forwarded to the task/executor.

        Returns:
            A status string: a policy/sequence denial message, an
            already-in-flight notice, or the rendered delegated_result line.
        """

        # PolicyGate parity: validate synthetic delegate intent so phase/role/paths/red-line gates apply.
        intent = Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": action_name, "params": dict(params or {})},
        )
        try:
            self.policy.validate_intent("orchestration", intent)
        except PolicyDenied as denied:
            await self._record_policy_denied("orchestration", intent, denied)
            return (
                f"(run_action_now: {action_name!r} denied by policy: "
                f"{getattr(denied, 'rule', '')!s} — "
                f"{str(getattr(denied, 'hint', denied))[:200]})"
            )
        seq_denied = self._sequence_denial_for_action(action_name)
        if seq_denied is not None:
            await self._record_policy_denied(
                "orchestration",
                intent,
                seq_denied,
                action_name=action_name,
            )
            return f"(run_action_now: {action_name!r} denied: {str(getattr(seq_denied, 'hint', seq_denied))[:200]})"
        lanes, ttl = self._registry_lanes_ttl(action_name)
        content_fp = hashlib.sha1(json.dumps(params or {}, sort_keys=True, default=str).encode()).hexdigest()[:10]
        key = f"inline:orchestration:{action_name}:t{int(self.shared_state.tick or 0)}:{content_fp}"
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=action_name,
            params=dict(params or {}),
            idempotency_key=key,
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing and task.state not in (
            "queued",
            "succeeded",
            "failed",
            "cancelled",
            "needs_manual_review",
        ):
            return (
                f"(run_action_now: an identical {action_name!r} task is "
                f"already {task.state!r}; wait for its delegated_result)"
            )
        result = await self.sub.run_task(task)
        result_payload = {
            "task_id": task.task_id,
            "kind": task.kind,
            "state": result.state,
            "result": result.result,
            "error": result.error,
        }
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "delegated_result",
                    {**result_payload, "inline": True},
                )
            )
        except Exception:  # noqa: BLE001 — audit best-effort
            log.exception(
                "run_action_now: failed to append delegated_result for %s",
                task.task_id,
            )
        rendered = _format_inbox_event(
            Message.new(
                "coordinator",
                "orchestration",
                "delegated_result",
                result_payload,
            )
        )
        return f"inline run complete: {rendered}"
