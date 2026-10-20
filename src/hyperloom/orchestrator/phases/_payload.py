# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reads of the model-authored payloads a phase routes: specialist deliverables, executor results, task params.

Every one of those dicts is written by a model or crosses a process boundary as
JSON, so a field may be absent, null, or carry another type than the phase
expects. These hold that one rule, leaving each call site reading as the
decision it makes rather than as a coercion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def as_text(*values: Any) -> str:
    """The first of ``values`` carrying anything, as text; empty when none of them does."""
    for value in values:
        if value:
            return str(value)
    return ""


def as_flag(value: Any) -> bool:
    """``value`` as a plain yes/no; a field the payload left out or wrote as null reads as no."""
    return bool(value)


def as_mapping(value: Any) -> dict[str, Any]:
    """``value`` as a mapping of its own; empty when it is absent or carries another type."""
    return dict(value) if isinstance(value, Mapping) else {}


def as_entries(value: Any) -> list[Any]:
    """``value`` as a list of its own; empty when it is absent or carries another type."""
    return list(value) if isinstance(value, list) else []


def as_texts(value: Any) -> list[str]:
    """The non-empty texts ``value`` holds, each stripped of surrounding space."""
    return [text for text in (str(entry).strip() for entry in as_entries(value)) if text]


def task_id_of(task: Any) -> str:
    """The registry id of ``task``; empty when the registry handed back no row."""
    return str(getattr(task, "task_id", "") or "")


def params_of(task: Any) -> dict[str, Any]:
    """The dispatch params of ``task``; empty when the registry handed back no row."""
    return dict(getattr(task, "params", None) or {})


def payload_of(message: Any) -> dict[str, Any]:
    """The body of a bus message; empty when the entry carries none."""
    return dict(getattr(message, "payload", None) or {})
