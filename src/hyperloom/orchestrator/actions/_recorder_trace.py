# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trace recordings the orchestrator never managed to hand to the recorder."""

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
    """Report a recorder call the producer could not make."""
    try:
        # Imported here for the same reason the calls this reports on are: the breakdown package pulls in the
        # exporter, and the orchestrator does not carry that at import time.
        from hyperloom.inference_optimizer.breakdown.recorder.trace import trace_skip

        trace_skip(reason=reason, section=section, entity=entity, error=error)
    except Exception:  # noqa: BLE001 - a trace must never break the caller
        pass
