# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trace recordings the orchestrator never managed to hand to the recorder.

The recorder traces its own coverage: every entry guard that returns early and
every writer failure it swallows leaves a line saying what was wanted and why
nothing was written. All of that is inside the recorder, and the orchestrator
calls it from inside ``try``/``except`` blocks that are deliberately wider than
the call itself -- they cover the lazy import, the arguments, and the
pre-conditions checked before calling. A failure in any of those means the
recorder is never reached, so none of its own tracing fires and the record is
lost in silence.

That silence is worse than it sounds. A lost adoption does not leave a hole in
the report: the export skips the step it cannot see, the workload has still
moved, and the difference is booked as gain belonging to no step -- a plausible
figure standing where a missing record should be. This is the only thing that
distinguishes the two.

Kept as its own module because it is called from several producers, and the
rule it encodes -- what a producer says when it could not even try -- is one
rule.
"""

from __future__ import annotations

from typing import Any

__all__ = ["trace_recording_skipped"]


def trace_recording_skipped(
    section: str,
    *,
    reason: str,
    entity: Any = None,
    error: BaseException | None = None,
) -> None:
    """Report a recorder call the producer could not make.

    Safe to call from any exception handler: it never raises, and it does
    nothing at all unless the breakdown write trace is switched on.

    Args:
        section: The breakdown section the lost record belonged to.
        reason: Why the call was not made, in a few words.
        entity: The id the record would have been keyed by, when known.
        error: The exception that stopped it, for the raising paths.
    """
    try:
        # Imported here for the same reason the calls this reports on are:
        # the breakdown package pulls in the exporter, and the orchestrator
        # does not carry that at import time. If the import is itself what
        # failed, there is no trace machinery to report to and nothing to say.
        from hyperloom.inference_optimizer.breakdown.recorder.trace import trace_skip

        trace_skip(reason=reason, section=section, entity=entity, error=error)
    except Exception:  # noqa: BLE001 - a trace must never break the caller
        pass
