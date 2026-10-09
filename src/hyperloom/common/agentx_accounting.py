# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measured request errors from InferenceX's phase-overlapping accounting."""

from __future__ import annotations

from typing import Any


def measured_request_errors(accounting: dict[str, Any]) -> int | float | None:
    """Exclude warmup errors when the dropped union is available.

    InferenceX counts errors across both phases, while dropped rows are the
    union of warmup and error rows. Older artifacts lack that union, so retain
    their conservative all-phase error count instead of guessing an overlap.
    """
    errors = accounting.get("records_error_dropped")
    if isinstance(errors, bool) or not isinstance(errors, (int, float)) or not errors >= 0:
        return None
    if "records_dropped_total" not in accounting:
        return errors

    dropped = accounting["records_dropped_total"]
    warmup = accounting.get("records_warmup_dropped")
    if not all(type(value) is int for value in (dropped, warmup, errors)):
        return None
    if not (0 <= warmup <= dropped and 0 <= dropped - warmup <= errors <= dropped):
        return None
    return dropped - warmup
