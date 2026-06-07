# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Typed errors raised by the Critic runtime adapter.

The errors are deliberately granular so the CLI / SKILL can map them to
deterministic recovery paths (e.g. dead-letter, fall back to
``needs_review``, or surface ``required_context`` to the caller).
"""

from __future__ import annotations


class RuntimeAdapterError(RuntimeError):
    """Base class for all Critic runtime errors."""


class RequestValidationError(RuntimeAdapterError):
    """The incoming request JSON is structurally invalid."""


class ReviewValidationError(RuntimeAdapterError):
    """The Critic-produced review JSON does not match the expected schema."""


class SessionMemoryError(RuntimeAdapterError):
    """An I/O or schema error happened while accessing session memory."""


class ScopeError(RuntimeAdapterError):
    """The packet/context cannot be turned into a valid KB scope."""


class SlugifyError(RuntimeAdapterError):
    """The slugify input cannot be reduced to a non-empty deterministic slug."""


class KBError(RuntimeAdapterError):
    """Base class for KB transport errors."""


class KBValidationError(KBError):
    """The KB rejected the payload (HTTP 400 / 422)."""


class KBNotFoundError(KBError):
    """The KB returned 404 (e.g. edge target absent)."""


class KBConflictError(KBError):
    """The KB returned 409 — should not occur with upsert mode."""


class KBTransportError(KBError):
    """Network / 5xx / timeout errors after exhausting retries."""


class IntentEnvelopeValidationError(RuntimeAdapterError):
    """The intent envelope produced for the Coordinator is invalid."""


class InboxParseError(RuntimeAdapterError):
    """The Coordinator-style inbox prompt could not be parsed."""


__all__ = [
    "InboxParseError",
    "IntentEnvelopeValidationError",
    "KBConflictError",
    "KBError",
    "KBNotFoundError",
    "KBTransportError",
    "KBValidationError",
    "RequestValidationError",
    "ReviewValidationError",
    "RuntimeAdapterError",
    "ScopeError",
    "SessionMemoryError",
    "SlugifyError",
]
