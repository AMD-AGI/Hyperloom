# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SBD V6 install event: begin it, record each preflight step, mark failures and persist it into the session."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.timeutil import now_iso

log = logging.getLogger("hyperloom.inference_optimizer.cli")


def _begin_install_event(args: argparse.Namespace | None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "install",
        "kind": "install",
        "status": "succeeded",
        "start_time": now_iso(timespec="seconds"),
        "end_time": "",
        "ext": {
            "run_kind": "resume" if bool(getattr(args, "resume_from", None)) else "fresh",
            "hard_fail_step_id": None,
            "runtime_snapshot": {},
            "steps": [],
        },
    }
    from ..session.sbd_v6 import set_pending_install_event

    set_pending_install_event(args, event)
    return event


@dataclass(frozen=True)
class InstallStep:
    """One install-event step: its id and category, the status it ended in, and why it skipped or what it said."""

    step_id: str
    category: str
    status: str
    skip_reason: str | None = None
    message: str | None = None


def _record_install_step(event: dict[str, Any], step: InstallStep, **fields: Any) -> None:
    row: dict[str, Any] = {
        "step_id": step.step_id,
        "category": step.category,
        "status": step.status,
        "skip_reason": step.skip_reason,
    }
    if step.message is not None:
        row["message"] = step.message
    row.update(fields)
    event["ext"]["steps"].append(row)


def _fail_install_step(
    event: dict[str, Any],
    *,
    step_id: str,
    category: str,
    exc: BaseException,
) -> None:
    failure_fields: dict[str, Any] = {}
    if isinstance(exc, SystemExit):
        failure_fields["detail"] = {"exit_code": exc.code}
    _record_install_step(
        event,
        InstallStep(step_id, category, "failed", message=str(exc) or type(exc).__name__),
        error_class=type(exc).__name__,
        **failure_fields,
    )
    event["status"] = "failed"
    event["end_time"] = now_iso(timespec="seconds")
    event["ext"]["hard_fail_step_id"] = step_id


def _mark_pending_install_event_failed(
    args: argparse.Namespace | None,
    exc: BaseException,
) -> None:
    """Mark an unwrapped preflight exception on the pending install event."""
    from ..session.sbd_v6 import pending_install_event

    event = pending_install_event(args)
    if event is None:
        event = _begin_install_event(args)
    if str(event.get("status") or "") != "failed":
        _fail_install_step(
            event,
            step_id="unhandled_preflight",
            category="check",
            exc=exc,
        )


def _run_install_step(
    event: dict[str, Any],
    *,
    step_id: str,
    category: str,
    action: Callable[[], Any],
    **success_fields: Any,
) -> Any:
    try:
        result = action()
    except BaseException as exc:
        _fail_install_step(event, step_id=step_id, category=category, exc=exc)
        raise
    outcome = dict(result) if isinstance(result, dict) else {}
    status = str(outcome.pop("status", "applied") or "applied")
    skip_reason = outcome.pop("skip_reason", None)
    message = outcome.pop("message", None)
    fields = {**success_fields, **outcome}
    _record_install_step(
        event,
        InstallStep(step_id, category, status, skip_reason=skip_reason, message=message),
        **fields,
    )
    return result


def _persist_install_event(args: argparse.Namespace | None, session_dir: Path) -> None:
    """Persist the pre-session install trace without changing launch behavior."""
    from ..session.sbd_v6 import persist_pending_install_event, record_write_warning

    try:
        path = persist_pending_install_event(args, session_dir)
    except (OSError, ValueError, TypeError) as exc:
        log.warning("failed to persist SBD V6 install event", exc_info=True)
        if not record_write_warning(session_dir, component="install.event", exc=exc):
            log.debug("failed to persist SBD V6 install-event write warning", exc_info=True)
        return
    if path is None:
        exc = RuntimeError("pending install event is unavailable")
        log.warning("failed to persist SBD V6 install event: %s", exc)
        if not record_write_warning(session_dir, component="install.event", exc=exc):
            log.debug("failed to persist SBD V6 install-event write warning", exc_info=exc)
