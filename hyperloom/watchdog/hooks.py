"""Watchdog hooks — emit events from Hyperloom subsystems into the event log.

Called from bench.py, dispatch.py, and state.py to automatically record
significant events. The watchdog scanner picks them up on the next poll cycle.

All hooks are fire-and-forget: they never raise and never block the caller.

Usage:
    from hyperloom.watchdog.hooks import emit_benchmark_event, emit_gate_event
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("hyperloom.watchdog.hooks")


def _session_dir() -> str | None:
    return os.environ.get("SESSION_DIR")


def _safe_emit(fn):
    """Decorator that silences all errors from event emission."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.debug("Watchdog hook failed", exc_info=True)
    return wrapper


@_safe_emit
def emit_benchmark_event(
    bench_result: dict[str, Any],
    tag: str = "",
    session_dir: str | None = None,
) -> None:
    """Emit a benchmark completion event."""
    from hyperloom.watchdog.event_log import append_event

    sd = session_dir or _session_dir()
    if not sd:
        return

    throughput = bench_result.get("output_throughput", 0)
    completed = bench_result.get("completed", 0)
    num_prompts = bench_result.get("num_prompts", 0)

    severity = "info"
    if completed == 0:
        severity = "error"
    elif num_prompts > 0 and completed / num_prompts < 0.9:
        severity = "warning"

    append_event(
        session_dir=sd,
        source="orchestrator",
        event_type="benchmark",
        severity=severity,
        promising=throughput > 0,
        details={
            "output_throughput": throughput,
            "input_throughput": bench_result.get("input_throughput", 0),
            "mean_ttft_ms": bench_result.get("mean_ttft_ms", 0),
            "mean_tpot_ms": bench_result.get("mean_tpot_ms", 0),
            "completed": completed,
            "num_prompts": num_prompts,
            "tag": tag,
        },
    )


@_safe_emit
def emit_gate_event(
    gate_result: dict[str, Any],
    action_description: str = "",
    session_dir: str | None = None,
) -> None:
    """Emit a gate (keep/revert) decision event."""
    from hyperloom.watchdog.event_log import append_event

    sd = session_dir or _session_dir()
    if not sd:
        return

    throughput_passed = gate_result.get("throughput_passed", True)
    accuracy_passed = gate_result.get("accuracy_passed", True)
    kept = throughput_passed and accuracy_passed

    severity = "info" if kept else "warning"

    append_event(
        session_dir=sd,
        source="orchestrator",
        event_type="gate_result",
        severity=severity,
        promising=kept,
        details={
            **gate_result,
            "action": action_description,
            "decision": "keep" if kept else "revert",
        },
    )


@_safe_emit
def emit_agent_event(
    agent_id: str,
    event_type: str,
    details: dict[str, Any] | None = None,
    session_dir: str | None = None,
) -> None:
    """Emit an agent lifecycle event."""
    from hyperloom.watchdog.event_log import append_event

    sd = session_dir or _session_dir()
    if not sd:
        return

    severity = "info"
    if event_type in ("agent_crash", "agent_error", "agent_timeout"):
        severity = "warning"

    append_event(
        session_dir=sd,
        source="dispatch",
        event_type=event_type,
        severity=severity,
        details={"agent_id": agent_id, **(details or {})},
    )


@_safe_emit
def emit_server_event(
    event_type: str,
    details: dict[str, Any] | None = None,
    session_dir: str | None = None,
) -> None:
    """Emit a server lifecycle event (start, stop, crash)."""
    from hyperloom.watchdog.event_log import append_event

    sd = session_dir or _session_dir()
    if not sd:
        return

    severity = "info"
    if event_type in ("server_crash", "server_died"):
        severity = "error"

    append_event(
        session_dir=sd,
        source="server",
        event_type=event_type,
        severity=severity,
        details=details or {},
    )


@_safe_emit
def emit_config_change(
    change: dict[str, Any],
    session_dir: str | None = None,
) -> None:
    """Emit a configuration change event."""
    from hyperloom.watchdog.event_log import append_event

    sd = session_dir or _session_dir()
    if not sd:
        return

    append_event(
        session_dir=sd,
        source="orchestrator",
        event_type="config_change",
        severity="info",
        details=change,
    )
