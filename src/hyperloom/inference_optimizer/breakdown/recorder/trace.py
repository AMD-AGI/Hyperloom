# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write-path trace log for the breakdown recorder."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.env import env_bool
from hyperloom.common.env_safety import redact_secret_values

#: Below ``DEBUG`` because every recorder write emits an event.
TRACE = 5

#: Sole switch for the write trace. Independent of global verbosity, which the
#: CLI floors at ``DEBUG``, so this can be turned on for a run without also
#: turning on everything else.
TRACE_ENV = "HYPERLOOM_BREAKDOWN_TRACE"

logging.addLevelName(TRACE, "TRACE")

log = logging.getLogger(__name__)

#: Whether this process asked for the trace. Held separately from the logger's
#: level because "off" and "not configured" are different answers: clearing the
#: level would hand the decision to whatever the root logger happens to be set
#: to, and a root at ``NOTSET`` enables everything.
_enabled = False

#: Fields that carry a v4 entity's stable identity, most specific first, so a
#: nested entity is named by its own id rather than its parent's.
_ID_FIELDS = (
    "measurement_id",
    "adoption_id",
    "operation_id",
    "subject_id",
    "artifact_id",
    "attempt_id",
    "substep_id",
    "gate_id",
    "decision_id",
    "relation_id",
    "kernel_id",
)

#: Frames inside this directory are the SDK itself, not the caller of it.
_SDK_DIR = str(Path(__file__).resolve().parent)

_VALUE_LIMIT = 48
#: Ids are long and end in a content hash, so cutting one at the value limit
#: would remove the part that distinguishes it from its neighbours.
_ID_LIMIT = 120
_CHANGED_LIMIT = 8


def enable_trace(enabled: bool = True) -> None:
    """Turn the write trace on or off for this process.

    The level is set on this logger rather than the root so the trace can be
    read without lowering everything else, and so it survives a ``basicConfig``
    that floors the root level above it.
    """
    global _enabled
    _enabled = bool(enabled)
    if _enabled:
        log.setLevel(TRACE)


def trace_enabled() -> bool:
    """Report whether the write trace would be emitted."""
    return _enabled and log.isEnabledFor(TRACE)


def _short(value: Any, limit: int = _VALUE_LIMIT) -> str:
    """Render one value for a single trace line, bounded and newline-free."""
    text = str(value)
    if len(text) > limit:
        text = f"{text[:limit]}..."
    return text.replace("\n", " ").replace("\r", " ")


def _entity(payload: Mapping[str, Any]) -> str:
    """Name the entity a payload is about, by its most specific stable id."""
    for field in _ID_FIELDS:
        value = payload.get(field)
        if value:
            return f"{field}={_short(value, _ID_LIMIT)}"
    name = payload.get("name") or payload.get("tool")
    return f"name={_short(name)}" if name else "id=none"


def _transition(previous: Any, new: Any) -> str:
    """Render a field's change, spelling out the values only when they are scalar."""
    if isinstance(previous, (str, int, float, bool, type(None))) and isinstance(
        new, (str, int, float, bool, type(None))
    ):
        return f"{_short(previous)}->{_short(new)}"
    return "changed"


def _changed(previous: Mapping[str, Any], merged: Mapping[str, Any]) -> str:
    """Summarise what a merging write did to the fragment already on disk."""
    added: list[str] = []
    changed: list[str] = []
    for key in sorted(set(previous) | set(merged)):
        if key not in merged:
            continue
        new = merged[key]
        if key not in previous:
            added.append(f"+{key}")
        elif previous[key] != new:
            changed.append(f"{key}:{_transition(previous[key], new)}")
    if not added and not changed:
        return "changed=none"
    shown_changed = changed[:_CHANGED_LIMIT]
    shown_added = added[:_CHANGED_LIMIT]
    # Both lists are capped, so both have to be counted; a write that only adds fields is otherwise reported as if
    # nothing had been left out.
    hidden = (len(changed) - len(shown_changed)) + (len(added) - len(shown_added))
    parts = shown_changed + shown_added
    if hidden > 0:
        parts.append(f"(+{hidden} more)")
    return "changed=" + ",".join(parts)


def _call_site() -> str:
    """Locate the code responsible for a write, on both sides of the SDK."""
    via = ""
    try:
        frame: Any = sys._getframe(1)  # noqa: SLF001 - cheap, and guarded by the level check
    except (ValueError, AttributeError):
        return "via=? from=?"
    while frame is not None:
        filename = frame.f_code.co_filename
        where = f"{os.path.basename(filename)}:{frame.f_lineno}:{frame.f_code.co_name}"
        if os.path.dirname(os.path.abspath(filename)) == _SDK_DIR:
            via = where
        elif via:
            return f"via={via} from={where}"
        frame = frame.f_back
    return f"via={via or '?'} from=?"


def _emit(section: str, fields: list[str]) -> None:
    """Mask one assembled trace line and log it."""
    log.log(TRACE, "breakdown %s", redact_secret_values(" ".join(fields)))


def trace_write(
    *,
    section: str,
    kind: str,
    operation: str,
    target: Path,
    payload: Mapping[str, Any],
    producer: str,
    seq: int,
    ts: str,
    size: int,
    existed: bool,
    previous: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    """Log one fragment write, or the failure of one."""
    if not trace_enabled():
        return
    try:
        fields = [
            f"section={section}",
            f"kind={kind}",
            f"op={operation}",
            "outcome=" + ("failed" if error is not None else "replaced" if existed else "created"),
            _entity(payload),
            f"producer={producer}",
            f"seq={seq}",
            f"ts={ts}",
            f"bytes={size}",
            f"file={target.name}",
        ]
        if previous is not None:
            fields.append(_changed(previous, payload))
        fields.append(_call_site())
        if error is not None:
            fields.append(f"error={type(error).__name__}:{_short(error)}")
        _emit(section, fields)
    except Exception:  # noqa: BLE001 - a trace must never break a recording
        log.log(TRACE, "breakdown trace failed for section=%s", section, exc_info=True)


def trace_skip(
    *,
    reason: str,
    section: str,
    producer: str = "",
    entity: Any = None,
    error: BaseException | None = None,
) -> None:
    """Log a recording that was wanted but never attempted."""
    if not trace_enabled():
        return
    try:
        fields = [
            f"section={section}",
            "outcome=skipped",
            f"reason={_short(reason)}",
        ]
        if entity:
            fields.append(f"id={_short(entity, _ID_LIMIT)}")
        if producer:
            fields.append(f"producer={producer}")
        fields.append(_call_site())
        if error is not None:
            fields.append(f"error={type(error).__name__}:{_short(error)}")
        _emit(section, fields)
    except Exception:  # noqa: BLE001 - a trace must never break a recording
        log.log(TRACE, "breakdown trace failed for section=%s", section, exc_info=True)


if env_bool(TRACE_ENV):
    enable_trace()


__all__ = [
    "TRACE",
    "TRACE_ENV",
    "enable_trace",
    "trace_enabled",
    "trace_skip",
    "trace_write",
]
