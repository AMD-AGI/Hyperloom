"""Small write/read surface for additive SBD V6 timeline events."""

from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path
from typing import Any

from hyperloom.common.io import append_jsonl, atomic_write_json
from hyperloom.common.jsonio import read_json, read_jsonl
from hyperloom.common.timeutil import now_iso

from .session_paths import (
    sbd_v6_timeline_dir,
    sbd_v6_timeline_event_path,
    sbd_v6_write_warnings_path,
)


SCHEMA_VERSION_V6 = "hyperloom.session_breakdown.v6.0"
_PENDING_INSTALL_ATTR = "_sbd_v6_install_event"
_STORAGE_SEQUENCE_KEY = "__sbd_v6_timeline_sequence"
_EVENT_TYPES = ("install", "model_gate")
_EVENT_FILE_RE = re.compile(r"^(?P<sequence>\d+)-(?P<event_type>[a-z0-9_]+)\.json$")


def _validate_event_type(event_type: str) -> str:
    if event_type not in _EVENT_TYPES:
        raise ValueError(f"unsupported SBD V6 timeline event type: {event_type!r}")
    return event_type


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    value = dict(event)
    value.pop(_STORAGE_SEQUENCE_KEY, None)
    return value


def _write_event(path: Path, event: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        _public_event(event),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        trailing_newline=True,
        make_parents=True,
    )


def _history_files(session_dir: Path | str) -> list[tuple[int, str, Path]]:
    root = sbd_v6_timeline_dir(Path(session_dir))
    if not root.is_dir():
        return []
    files: list[tuple[int, str, Path]] = []
    for path in root.glob("*.json"):
        match = _EVENT_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        event_type = match.group("event_type")
        if event_type not in _EVENT_TYPES:
            continue
        files.append((int(match.group("sequence")), event_type, path))
    files.sort(key=lambda row: row[0])
    return files


def _read_event_file(
    path: Path,
    event_type: str,
    warnings: list[str] | None = None,
) -> dict[str, Any] | None:
    try:
        event = read_json(path, require_dict=True, strict=True)
    except Exception as exc:
        if warnings is not None:
            warnings.append(f"timeline.{event_type}: failed to parse {path}: {exc!r}")
        return None
    if str(event.get("type") or "") != event_type:
        if warnings is not None:
            warnings.append(f"timeline.{event_type}: ignored event with type={event.get('type')!r}")
        return None
    return _public_event(event)


def write_timeline_event(session_dir: Path | str, event: dict[str, Any]) -> Path:
    """Persist one event without replacing an earlier run of the same stage."""
    event_type = _validate_event_type(str(event.get("type") or "").strip())
    history = _history_files(session_dir)

    raw_sequence = event.get(_STORAGE_SEQUENCE_KEY)
    try:
        sequence = int(raw_sequence) if raw_sequence is not None else 0
    except (TypeError, ValueError):
        sequence = 0
    if sequence > 0:
        conflicting_type = next(
            (stored_type for stored_sequence, stored_type, _ in history if stored_sequence == sequence),
            None,
        )
        if conflicting_type is not None and conflicting_type != event_type:
            raise ValueError(f"SBD V6 timeline sequence {sequence} belongs to {conflicting_type!r}, not {event_type!r}")
    else:
        sequence = max((stored_sequence for stored_sequence, _, _ in history), default=0) + 1
        event[_STORAGE_SEQUENCE_KEY] = sequence

    path = sbd_v6_timeline_event_path(Path(session_dir), sequence, event_type)
    _write_event(path, event)
    return path


def read_timeline_event(
    session_dir: Path | str,
    event_type: str,
) -> dict[str, Any] | None:
    """Read the latest persisted event of one type."""
    _validate_event_type(event_type)
    for _, stored_type, path in reversed(_history_files(session_dir)):
        if stored_type != event_type:
            continue
        event = _read_event_file(path, event_type)
        if event is not None:
            return event
    return None


def read_timeline_event_for_update(
    session_dir: Path | str,
    event_type: str,
) -> dict[str, Any] | None:
    """Read the latest event with its private storage sequence attached."""
    _validate_event_type(event_type)
    for sequence, stored_type, path in reversed(_history_files(session_dir)):
        if stored_type != event_type:
            continue
        event = _read_event_file(path, event_type)
        if event is not None:
            event[_STORAGE_SEQUENCE_KEY] = sequence
            return event
    return None


def read_timeline_events(
    session_dir: Path | str,
    *,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Read all persisted V6 events in execution order."""
    if warnings is not None:
        for warning in read_write_warnings(session_dir, warnings=warnings):
            if warning not in warnings:
                warnings.append(warning)
    events: list[dict[str, Any]] = []
    for _, event_type, path in _history_files(session_dir):
        event = _read_event_file(path, event_type, warnings)
        if event is not None:
            events.append(event)
    return events


def record_write_warning(
    session_dir: Path | str,
    *,
    component: str,
    exc: BaseException,
) -> bool:
    """Best-effort persist a V6 writer failure for the next export."""
    try:
        append_jsonl(
            sbd_v6_write_warnings_path(Path(session_dir)),
            {
                "ts": now_iso(timespec="seconds"),
                "component": str(component or "unknown"),
                "error_class": type(exc).__name__,
                "message": str(exc) or repr(exc),
            },
            make_parents=True,
            ensure_ascii=False,
            sort_keys=True,
        )
    except Exception:  # noqa: BLE001 — warning persistence cannot change runtime behavior
        return False
    return True


def read_write_warnings(
    session_dir: Path | str,
    *,
    warnings: list[str] | None = None,
) -> list[str]:
    """Read durable V6 writer failures without mutating the session."""
    path = sbd_v6_write_warnings_path(Path(session_dir))
    if not path.is_file():
        return []
    rows = read_jsonl(
        path,
        require_dict=True,
        skip_malformed=True,
        on_error=(
            (lambda exc: warnings.append(f"timeline.write_warnings: failed to parse {path}: {exc!r}"))
            if warnings is not None
            else None
        ),
    )
    result: list[str] = []
    for row in rows:
        component = str(row.get("component") or "unknown")
        error_class = str(row.get("error_class") or "Error")
        message = str(row.get("message") or "write failed")
        result.append(f"sbd_v6.write.{component}: {error_class}: {message}")
    return result


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
    "read_write_warnings",
    "read_timeline_event",
    "read_timeline_event_for_update",
    "read_timeline_events",
    "record_write_warning",
    "set_pending_install_event",
    "write_timeline_event",
]
