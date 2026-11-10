# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shape a ``SEND_MESSAGE`` intent's payload must have before the bus stores it."""

from __future__ import annotations

from typing import Any

from .denial import PolicyDenied

__all__ = ["deny_malformed_send_message"]


def _names_recipients(to: Any) -> bool:
    """Whether ``to`` is a non-empty string or a non-empty list of non-empty strings."""
    names = to if isinstance(to, list) else [to]
    return bool(names) and all(isinstance(name, str) and name.strip() for name in names)


def deny_malformed_send_message(payload: dict[str, Any]) -> None:
    """Refuse a ``SEND_MESSAGE`` without a ``topic``, or with a ``to`` that names no recipient.

    Unknown topics are intentionally not rejected here: the Coordinator soft-degrades them to
    ``"observation"``, so agents can still surface unstructured observations. An absent ``to``
    broadcasts; a string names one recipient and a list several, each written as its own bus
    message (an ``events`` row's ``to_agent`` is one name). Other shapes, or a list holding an
    empty or non-string name, are denied here rather than crashing the intent handler.

    Raises:
        PolicyDenied: with ``rule='payload'`` when ``topic`` is missing or blank, or ``to`` is
            present but neither a non-empty string nor a non-empty list of them.
    """
    if not str(payload.get("topic", "")).strip():
        raise PolicyDenied("send_message missing topic", rule="payload")
    to = payload.get("to")
    if to is not None and not _names_recipients(to):
        raise PolicyDenied(
            f"send_message to must be a non-empty string or a non-empty list of them, got {to!r}",
            rule="payload",
            hint="name one recipient or a list of them, or omit 'to' to broadcast.",
        )
