# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded exception facts for the local KB publication audit."""

from __future__ import annotations

from urllib.error import HTTPError, URLError


def publish_error_fields(error: Exception) -> dict[str, str | int]:
    """Keep typed failure facts without exception text or response content."""
    fields: dict[str, str | int] = {"type": type(error).__name__}
    cause = error.__cause__ or error
    if cause is not error:
        fields["cause_type"] = type(cause).__name__
    if isinstance(cause, HTTPError):
        if type(cause.code) is int:
            fields["http_status"] = cause.code
    else:
        if isinstance(cause, URLError) and isinstance(cause.reason, OSError):
            cause = cause.reason
            fields["reason_type"] = type(cause).__name__
        if isinstance(cause, OSError) and type(cause.errno) is int:
            fields["errno"] = cause.errno
    return fields
