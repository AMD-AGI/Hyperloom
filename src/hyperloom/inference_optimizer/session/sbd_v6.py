"""Small write/read surface for additive SBD V6 timeline events."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import read_json

from .session_paths import sbd_v6_install_path, sbd_v6_model_gate_path


SCHEMA_VERSION_V6 = "hyperloom.session_breakdown.v6.0"
_PENDING_INSTALL_ATTR = "_sbd_v6_install_event"


def _event_path(session_dir: Path | str, event_type: str) -> Path:
    root = Path(session_dir)
    if event_type == "install":
        return sbd_v6_install_path(root)
    if event_type == "model_gate":
        return sbd_v6_model_gate_path(root)
    raise ValueError(f"unsupported SBD V6 timeline event type: {event_type!r}")


def write_timeline_event(session_dir: Path | str, event: dict[str, Any]) -> Path:
    """Atomically persist one V6 timeline event by its ``type``."""
    event_type = str(event.get("type") or "").strip()
    target = _event_path(session_dir, event_type)
    atomic_write_json(
        target,
        event,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        trailing_newline=True,
        make_parents=True,
    )
    return target


def read_timeline_event(
    session_dir: Path | str,
    event_type: str,
) -> dict[str, Any] | None:
    """Read one persisted event, returning ``None`` when absent or invalid."""
    path = _event_path(session_dir, event_type)
    if not path.is_file():
        return None
    try:
        return read_json(path, require_dict=True, strict=True)
    except Exception:
        return None


def read_timeline_events(session_dir: Path | str) -> list[dict[str, Any]]:
    """Read the currently implemented V6 timeline stages in business order."""
    events: list[dict[str, Any]] = []
    for event_type in ("install", "model_gate"):
        event = read_timeline_event(session_dir, event_type)
        if event is not None:
            events.append(event)
    return events


def set_pending_install_event(args: Namespace | None, event: dict[str, Any]) -> None:
    """Attach the pre-session install event to the parsed CLI namespace."""
    if args is not None:
        setattr(args, _PENDING_INSTALL_ATTR, event)


def pending_install_event(args: Namespace | None) -> dict[str, Any] | None:
    """Return the in-memory install event captured before session creation."""
    if args is None:
        return None
    event = getattr(args, _PENDING_INSTALL_ATTR, None)
    return event if isinstance(event, dict) else None


def persist_pending_install_event(args: Namespace | None, session_dir: Path | str) -> Path | None:
    """Persist the install event once the session directory exists."""
    event = pending_install_event(args)
    if event is None:
        return None
    return write_timeline_event(session_dir, event)


__all__ = [
    "SCHEMA_VERSION_V6",
    "pending_install_event",
    "persist_pending_install_event",
    "read_timeline_event",
    "read_timeline_events",
    "set_pending_install_event",
    "write_timeline_event",
]
