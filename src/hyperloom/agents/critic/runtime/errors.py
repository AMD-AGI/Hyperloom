# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed errors raised by the Critic runtime adapter."""

from __future__ import annotations

from hyperloom.common.subprocess_bridge import RuntimeAdapterError as RuntimeAdapterError


class RequestValidationError(RuntimeAdapterError):
    """The incoming request JSON is structurally invalid."""


class ReviewValidationError(RuntimeAdapterError):
    """The Critic-produced review JSON does not match the expected schema."""


class SessionMemoryError(RuntimeAdapterError):
    """An I/O or schema error happened while accessing session memory."""


class IntentEnvelopeValidationError(RuntimeAdapterError):
    """The intent envelope produced for the Coordinator is invalid."""


__all__ = [
    "IntentEnvelopeValidationError",
    "RequestValidationError",
    "ReviewValidationError",
    "RuntimeAdapterError",
    "SessionMemoryError",
]
