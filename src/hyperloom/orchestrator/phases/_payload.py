# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Field reads over the model-authored payloads a phase routes: specialist deliverables, executor results, task params.

Every one of those dicts is written by a model or crosses a process boundary as
JSON, so any key may be absent, null, or carry another type than the phase
expects. These readers hold that rule in one place, leaving each call site
reading as the decision it makes rather than as a coercion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def text_field(payload: Mapping[str, Any], *keys: str) -> str:
    """The first of ``keys`` carrying a value, as text; empty when none of them does."""
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def mapping_field(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    """``key`` as a mapping of its own; empty when absent or carrying another type."""
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def list_field(payload: Mapping[str, Any], key: str) -> list[Any]:
    """``key`` as a list of its own; empty when absent or carrying another type."""
    value = payload.get(key)
    return list(value) if isinstance(value, list) else []


def text_list_field(payload: Mapping[str, Any], key: str) -> list[str]:
    """``key`` as the non-empty texts it holds, each stripped of surrounding space."""
    return [text for text in (str(entry).strip() for entry in list_field(payload, key)) if text]


def task_id_of(task: Any) -> str:
    """The registry id of ``task``; empty when the registry handed back no row."""
    return str(getattr(task, "task_id", "") or "")


def params_of(task: Any) -> dict[str, Any]:
    """The dispatch params of ``task``; empty when the registry handed back no row."""
    return dict(getattr(task, "params", None) or {})


def payload_of(message: Any) -> dict[str, Any]:
    """The body of a bus message; empty when the entry carries none."""
    return dict(getattr(message, "payload", None) or {})
