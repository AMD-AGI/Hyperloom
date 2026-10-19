# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shape a ``SEND_MESSAGE`` intent's payload must have before the bus stores it."""

from __future__ import annotations

from typing import Any

from .denial import PolicyDenied

__all__ = ["deny_malformed_send_message"]


def deny_malformed_send_message(payload: dict[str, Any]) -> None:
    """Refuse a ``SEND_MESSAGE`` without a ``topic``, or with a ``to`` that is not a non-empty string.

    Unknown topics are intentionally not rejected here: the Coordinator soft-degrades them to
    ``"observation"``, so agents can still surface unstructured observations. An absent ``to``
    broadcasts; a present one names a single recipient. Other shapes (a JSON array of recipients
    among them) cannot be bound as the ``to_agent`` column of the ``events`` row, so they are denied
    here rather than crashing the intent handler.

    Raises:
        PolicyDenied: with ``rule='payload'`` when ``topic`` is missing or blank, or ``to`` is
            present but not a non-empty string.
    """
    if not str(payload.get("topic", "")).strip():
        raise PolicyDenied("send_message missing topic", rule="payload")
    to = payload.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise PolicyDenied(
            f"send_message to must be a non-empty string, got {to!r}",
            rule="payload",
            hint="address one recipient per send_message, or omit 'to' to broadcast.",
        )
