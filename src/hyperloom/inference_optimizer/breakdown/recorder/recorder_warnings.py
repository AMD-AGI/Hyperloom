# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a recording that failed says so.

Recording is best-effort by design: the run outranks its own record, so a
writer that raises is caught and the run carries on. What that left behind was
a breakdown quietly missing facts -- the export looked complete, and the only
trace of the loss was a ``DEBUG`` line in a log nobody keeps.

This routes those losses into the write-warning sidecar the V6 timeline
writers already park their failures in, so a lost fragment reaches
``metadata.warnings`` by the same path a lost timeline event does, and there
is one place to look rather than two.

Two rules hold the rest together. Noting a failure must never raise, because
it runs inside the handler for one. And it must never recurse: the note is
itself a write, so if it fails there is nothing left to do but log.
"""

from __future__ import annotations

import logging
import threading

from .trace import trace_skip

log = logging.getLogger(__name__)

#: One warning per section and error class. A writer on a hot path fails the
#: same way every tick, and an unbounded sidecar of identical lines would bury
#: the distinct failures that matter.
_seen: set[tuple[str, str]] = set()
_seen_lock = threading.Lock()

#: Set while a note is being parked, so a failure inside the note is logged
#: rather than noted, which would fail again in the same way forever.
_parking = threading.local()


def reset_for_tests() -> None:
    """Forget which failures were already noted."""
    with _seen_lock:
        _seen.clear()


def _first_time(section: str, error_class: str) -> bool:
    """Whether this section has yet to report this class of failure."""
    with _seen_lock:
        key = (section, error_class)
        if key in _seen:
            return False
        _seen.add(key)
        return True


def note_failure(
    *,
    section: str,
    error: BaseException,
    producer: str = "recorder",
    detail: str = "",
) -> None:
    """Report that ``section`` could not be recorded, durably and once.

    Args:
        section: The breakdown section whose write was lost.
        error: What the writer raised.
        producer: The component that was writing.
        detail: What the writer was doing, when the section alone is too
            coarse to locate it.
    """
    trace_skip(reason=detail or "writer raised", section=section, producer=producer, error=error)
    if getattr(_parking, "active", False):
        log.debug("recorder: a failure note for %s failed to park", section, exc_info=True)
        return
    if not _first_time(section, type(error).__name__):
        return

    component = f"recorder.{section}"
    log.warning(
        "recorder: %s could not be recorded (%s: %s)%s; the exported breakdown will be missing it",
        section,
        type(error).__name__,
        error,
        f" while {detail}" if detail else "",
    )

    _parking.active = True
    try:
        from ...session.sbd_v6 import record_write_warning
        from ...session.session_binding import bound_session_or_none

        session = bound_session_or_none()
        if session is None:
            # Nowhere to park it: the warning above is the whole record.
            return
        record_write_warning(session, component=component, exc=error)
    except Exception:  # noqa: BLE001 — the note is the last thing that may fail quietly
        log.debug("recorder: could not park the failure note for %s", section, exc_info=True)
    finally:
        _parking.active = False


__all__ = ["note_failure", "reset_for_tests"]
