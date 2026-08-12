# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write-path trace log for the breakdown recorder.

The recorder is deliberately fire-and-forget: producers call it from deep
inside their own work, every call site swallows exceptions, and the result is
a spool directory of fragments that only becomes a breakdown much later, in a
different process. That is the right shape for recording facts without holding
up the run, and it is the wrong shape for answering "who wrote this number,
from where, and what did it replace" -- which is the question asked every time
an exported figure is doubted.

Enabling this turns each write into one line naming the section, the fragment
file, the entity, the call site, and, for the merging writes, the fields whose
values changed. An overwrite is the interesting case: a fragment id is stable
per entity so a second write merges into the first, which is what lets a later
re-measure land on the readings an earlier decision was made on. Traced, that
shows up as ``changed=value:5081.01->5100.76`` at the moment it happens instead
of as a contradiction in an archive weeks later.

The level is below ``DEBUG`` on purpose and is switched on by
``HYPERLOOM_BREAKDOWN_TRACE`` alone, because a long session writes tens of
thousands of fragments and this must not arrive mixed into output someone
enabled to read something else. Nothing here may raise or slow down a run that
has it switched off: every entry point is guarded by one level check and
wrapped against its own failure, since a broken trace must never be the reason
a fact went unrecorded.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.env import env_bool

#: Below ``DEBUG`` (10). See the module docstring: this is a per-write firehose,
#: so it cannot share a level with output read for any other purpose.
TRACE = 5

#: Sole switch for the write trace. Independent of global verbosity, which the
#: CLI floors at ``DEBUG``, so this can be turned on for a run without also
#: turning on everything else.
TRACE_ENV = "HYPERLOOM_BREAKDOWN_TRACE"

logging.addLevelName(TRACE, "TRACE")

log = logging.getLogger(__name__)

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

    Args:
        enabled (bool): whether to emit the write trace.
    """
    log.setLevel(TRACE if enabled else logging.NOTSET)


def trace_enabled() -> bool:
    """Report whether the write trace would be emitted.

    Returns:
        bool: ``True`` when a call to :func:`trace_write` would log.
    """
    return log.isEnabledFor(TRACE)


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
    """Render a field's change, spelling out the values only when they are scalar.

    A measurement being overwritten is the case worth reading in full, and it
    is always a number. Nested payloads are named but not dumped: a line long
    enough to hold two of them is one nobody reads, and the fragment file has
    the detail anyway.
    """
    if isinstance(previous, (str, int, float, bool, type(None))) and isinstance(
        new, (str, int, float, bool, type(None))
    ):
        return f"{_short(previous)}->{_short(new)}"
    return "changed"


def _changed(previous: Mapping[str, Any], merged: Mapping[str, Any]) -> str:
    """Summarise what a merging write did to the fragment already on disk.

    Only the top level is compared, which is enough to know that the write was
    not a no-op and which record to go read.
    """
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
    shown = changed[:_CHANGED_LIMIT]
    hidden = len(changed) - len(shown)
    parts = shown + added[:_CHANGED_LIMIT]
    if hidden > 0:
        parts.append(f"(+{hidden} more)")
    return "changed=" + ",".join(parts)


def _call_site() -> str:
    """Locate the code responsible for a write, on both sides of the SDK.

    Reports the innermost frame still inside the recorder package as ``via``
    (the SDK helper that built the payload) and the first frame outside it as
    ``from`` (the producer that decided to record something). The second is
    what identifies the owner of a questionable number; the first is what
    identifies the helper to go read.
    """
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
    """Log one fragment write, or the failure of one.

    Args:
        section (str): the breakdown section written.
        kind (str): the declared section shape (``item`` / ``singleton``).
        operation (str): ``write`` when the fragment is replaced wholesale,
            ``upsert`` when it is merged into what was already there.
        target (Path): the fragment file written.
        payload (Mapping[str, Any]): the payload as written, post-merge.
        producer (str): the producer label owning the fragment.
        seq (int): the writer's per-process sequence number.
        ts (str): the timestamp stamped into the fragment envelope.
        size (int): the serialized size in bytes.
        existed (bool): whether the target already existed, making this write a
            replacement of an earlier fact rather than a new one.
        previous (Mapping[str, Any] | None): for a merging write, the payload
            found on disk, so the line can name what the write changed.
        error (BaseException | None): the failure, when the write did not land.
    """
    if not log.isEnabledFor(TRACE):
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
        log.log(TRACE, "breakdown %s", " ".join(fields))
    except Exception:  # noqa: BLE001 - a trace must never break a recording
        log.log(TRACE, "breakdown trace failed for section=%s", section, exc_info=True)


if env_bool(TRACE_ENV):
    enable_trace()


__all__ = [
    "TRACE",
    "TRACE_ENV",
    "enable_trace",
    "trace_enabled",
    "trace_write",
]
