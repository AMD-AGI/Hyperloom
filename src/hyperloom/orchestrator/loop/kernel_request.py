# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolution of a ``kernel_agent`` request into the outcome :class:`IntentRouter` replies with.

A request is answered from shared state when the session already knows what it produces,
and otherwise by its programmatic handler, running under the catalogue lanes of the action
that owns the request kind.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, NamedTuple

from hyperloom.inference_optimizer.protocol.action_surfaces import REQUEST_KIND_TO_OWNED_ACTION

from ..bus.message_bus import Message
from ..phases.machine_state import KERNEL_HEARTBEAT_SEC

log = logging.getLogger(__name__)

# Path-like keys surfaced from a kernel handler payload/result so operators can see where a step's artifacts went.
_LIFECYCLE_PATH_KEYS: tuple[str, ...] = (
    "trace_input",
    "trace_dir",
    "candidates_path",
    "analysis_md_path",
    "kernel_candidates",
    "best_artifact_path",
    "patch_path",
    "target_file",
    "workspace",
    "workspace_path",
    "out_dir",
    "output_dir",
    "run_dir",
    "report_path",
    "json_path",
    "md_path",
    "tracelens_agent_report",
    # TraceLens analysis outputs surfaced by trace_analyze_handler.
    "trace_report_path",
    "analysis_report_path",
    "tracelens_summary_path",
    "kernel_roofline_path",
    "cli_log_path",
)


def lifecycle_paths(payload: Any) -> dict[str, str]:
    """Extract present, non-empty path-like fields from a kernel handler payload or result dict."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for key in _LIFECYCLE_PATH_KEYS:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val
    return out


class KernelRequest(NamedTuple):
    """A kernel_agent request, with the payload its programmatic handler is called on."""

    message: Message
    kind: str
    payload: dict[str, Any]


class KernelRequestOutcome(NamedTuple):
    """What a kernel request produced, and what the reply names as the source of it."""

    result: dict[str, Any]
    reply_source: str
    from_cache: bool


def remembered_outcome(router: Any, request: KernelRequest) -> KernelRequestOutcome | None:
    """The outcome shared state already holds for *request*, or ``None`` when its handler must run."""
    # Roofline data is read from the last_trace_analyze cache rather than auto-injected here.
    cached = router._coord.phase_kernel.cached_kernel_request(request.kind, request.payload)
    if cached is not None:
        # A cache hit never runs the handler; emit a single END (detail=cache_hit) so the lifecycle log
        # records the step.
        router._coord.writeback.emit_lifecycle(
            step=request.kind,
            status="END",
            artifacts=lifecycle_paths(cached),
            detail="cache_hit",
        )
        return KernelRequestOutcome(cached, "shared_state_cache", True)
    if request.kind != "integrate":
        return None
    rejected = router.shared_state.find_rejected_kernel_patch(request.payload)
    if rejected is None:
        return None
    result = {
        "status": "skipped",
        "decision": "REVERT",
        "error_class": "kernel_patch_rejected",
        "error": "same kernel patch already exhausted E2E attempts",
        "kernel_id": rejected.get("kernel_id"),
        "patch_path": rejected.get("patch_path"),
        "target_file": rejected.get("target_file"),
        "extra_server_args": rejected.get("extra_server_args", ""),
        "attempt_count": rejected.get("attempt_count"),
        "best_gain_pct": rejected.get("best_gain_pct"),
        "reason": rejected.get("reason"),
    }
    # A short-circuited integrate never runs the handler; emit a lone END recording the rejection.
    router._coord.writeback.emit_lifecycle(
        step=request.kind,
        status="END",
        artifacts=lifecycle_paths(result),
        detail="rejected",
    )
    return KernelRequestOutcome(result, "shared_state_kernel_rejection", True)


def seed_integrate_base_tput(router: Any, request: KernelRequest) -> None:
    """Fill in the baseline throughput an integrate request left out, from the session's current best."""
    if request.kind != "integrate" or request.payload.get("base_tput"):
        return
    cb_tput = (router.shared_state.current_best or {}).get("tput")
    if isinstance(cb_tput, (int, float)) and cb_tput > 0:
        request.payload["base_tput"] = float(cb_tput)


@asynccontextmanager
async def kernel_step_heartbeat(router: Any, kind: str, started: float):
    """Keep orchestration's bus timestamp moving through an inline step."""

    # Re-stamped per beat rather than once at the start, so a stamp that outlives its process expires instead of
    # muting the KERNEL idle guard.
    def _mark_running() -> None:
        router.shared_state.kernel_inline_step_seen_unix = time.time()

    async def _beat() -> None:
        while True:
            await asyncio.sleep(KERNEL_HEARTBEAT_SEC)
            _mark_running()
            await router.bus.append_and_seq(
                Message.new(
                    "orchestration",
                    "*",
                    "observation",
                    {
                        "kind": "kernel_step_running",
                        "step": kind,
                        "elapsed_sec": round(time.monotonic() - started, 1),
                    },
                )
            )

    _mark_running()
    task = asyncio.create_task(_beat())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        router.shared_state.kernel_inline_step_seen_unix = 0.0


def close_kernel_step(router: Any, kind: str, result: dict[str, Any], started: float) -> None:
    """Close the lifecycle bracket a programmatic kernel step opened, recording what it decided."""
    status = "ERROR" if str(result.get("status", "")).lower() in ("failed", "error") else "END"
    detail = " ".join(
        str(p)
        for p in (
            result.get("decision"),
            result.get("status"),
            f"kernel={result.get('kernel_id')}" if result.get("kernel_id") else "",
        )
        if p
    )
    router._coord.writeback.emit_lifecycle(
        step=kind,
        status=status,
        artifacts=lifecycle_paths(result),
        detail=detail,
        duration_s=time.monotonic() - started,
    )


async def run_kernel_handler(router: Any, request: KernelRequest, handler: Any) -> KernelRequestOutcome | None:
    """Run *request* through *handler*, under the catalogue lanes of the action that owns its kind.

    Returns ``None`` once the lanes the handler needs were busy and the request has been
    answered with a deferral instead.
    """
    seed_integrate_base_tput(router, request)
    # A handler that benchmarks runs under its action's catalogue lanes, so it waits out the
    # kernel_agent task instead of sharing the GPUs with it.
    action = REQUEST_KIND_TO_OWNED_ACTION.get(request.kind, request.kind)
    lanes, ttl = router._coord.dispatcher.registry_lanes_ttl(action)
    handler_lease = None
    if lanes:
        handler_lease = await router.locks.try_acquire_many(
            lanes,
            holder_id=request.message.msg_id,
            task_id=request.message.msg_id,
            action=action,
            ttl_sec=ttl or 60,
        )
        if handler_lease is None:
            await router._reply(
                request.message,
                {"status": "deferred", "reason": "lanes_busy", "lanes": lanes},
                source="lanes_busy",
            )
            return None

    started = time.monotonic()
    router._coord.writeback.emit_lifecycle(
        step=request.kind,
        status="START",
        artifacts=lifecycle_paths(request.payload),
    )
    try:
        async with kernel_step_heartbeat(router, request.kind, started):
            result = await handler(request.payload, session_dir=router.session_dir)
    except Exception as exc:
        log.exception(
            "kernel_request_handler[%s] crashed for source=%s",
            request.kind,
            request.message.from_agent,
        )
        result = {
            "status": "failed",
            "error_class": "handler_exception",
            "error": repr(exc),
        }
    finally:
        if handler_lease is not None:
            await router.locks.release(handler_lease)
    close_kernel_step(router, request.kind, result, started)
    return KernelRequestOutcome(result, "programmatic_handler", False)
