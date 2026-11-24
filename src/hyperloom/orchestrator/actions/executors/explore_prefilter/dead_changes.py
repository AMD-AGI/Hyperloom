# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The changes this session saw stop a server before it was ready, and the variants that would launch one again.

A start death counts only when the framework refused the configuration (``_CONFIG_REFUSALS``); it then holds for the rest
of the session, a later source patch included. The experience corpus's attribution is left as it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from inference_suite.server_log import cause_line
from hyperloom.common.failure_signature import DEVICE_MEMORY_SHORTFALL_LINE
from hyperloom.inference_optimizer.canonical_fingerprint import change_items

from ....state.failure_evidence import (
    FAILURE_ATTRIBUTION_ENVIRONMENT,
    SERVER_INIT_DEAD,
    classify_failure_attribution,
    classify_root_cause_type,
    failure_text,
)
from ..grid._grid_base import TS_FAILED

__all__ = ["KNOWN_DEAD_CHANGE", "DeadChange", "carried_dead_change", "dead_changes"]

# ``skipped_dup`` reason for a variant that carries a change this session saw kill a server before it was ready.
KNOWN_DEAD_CHANGE: str = "known_dead_change"

# The root cause of a server that exited before it was ready whose log names no out-of-memory or capacity limit.
_ENGINE_INIT = "engine_core_init"

# Exceptions a framework raises when it rejects the configuration it was given. A host fault raised as another type
# (OSError, MemoryError, a RuntimeError with no device, a hub HTTP error) never retires a change; one raised under these
# names does, unless its log names the host or a card short of free memory.
_CONFIG_REFUSALS: frozenset[str] = frozenset(
    {"ValueError", "NotImplementedError", "AssertionError", "TypeError", "ValidationError"}
)


def _exception_name(cause: str) -> str:
    """The unqualified exception name a cause line starts with, empty for a runtime line that names none."""
    head, sep, _ = cause.partition(":")
    return head.rsplit(".", 1)[-1].strip() if sep else ""


@dataclass(frozen=True)
class DeadChange:
    """A change that stopped a server before it was ready, the variant it came from and the cause its log named."""

    name: str
    round_id: str
    items: frozenset[tuple[str, ...]]
    cause: str

    def detail(self) -> str:
        return f"carries the whole change of {self.name} ({self.round_id}), whose server died before it was ready: {self.cause}"


def _items_of(change: Mapping[str, Any]) -> frozenset[tuple[str, ...]]:
    return change_items(
        str(change.get("extra_server_args") or ""),
        change.get("extra_envs") or {},
        remove_args=change.get("remove_args"),
        unset_envs=change.get("unset_envs"),
        args_mode=str(change.get("args_mode") or "append"),
    )


def _start_death_cause(row: Mapping[str, Any]) -> str | None:
    """The fault line of a row whose server exited before it was ready because the framework refused its change.

    A log that names no fault (an exit code alone), or one whose fault is not a configuration refusal, cannot tell the
    change from memory, capacity or the host, so it yields None.
    """
    error_class = str(row.get("error_class") or "")
    excerpt = failure_text(row)
    cause = cause_line(excerpt) or ""
    if (
        _exception_name(cause) not in _CONFIG_REFUSALS
        or error_class != SERVER_INIT_DEAD
        or row.get("framework_changed")
        or DEVICE_MEMORY_SHORTFALL_LINE.search(excerpt)
    ):
        return None
    if classify_root_cause_type(error_class, excerpt) != _ENGINE_INIT:
        return None
    attribution = classify_failure_attribution(error_class=error_class, framework_changed=False, error_excerpt=excerpt)
    return None if attribution == FAILURE_ATTRIBUTION_ENVIRONMENT else cause


def dead_changes(rows: Iterable[Mapping[str, Any]], *, workload_signature: str) -> list[DeadChange]:
    """The changes the ``rows`` of this workload saw stop a server before it was ready, oldest first."""
    out: list[DeadChange] = []
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("outcome") or "") != TS_FAILED:
            continue
        if str(row.get("workload_signature") or "") != workload_signature:
            continue
        items = _items_of(row)
        cause = _start_death_cause(row) if items else None
        if cause:
            out.append(
                DeadChange(
                    name=str(row.get("name") or "?"),
                    round_id=str(row.get("round_id") or "an earlier round"),
                    items=items,
                    cause=cause,
                )
            )
    return out


def carried_dead_change(
    variant: Mapping[str, Any], dead: Iterable[DeadChange], *, stack: Mapping[str, Any]
) -> DeadChange | None:
    """The first dead change ``variant`` carries whole, unless the stack, which serves, already carries it."""
    items = _items_of(variant)
    running = _items_of(stack)
    return next((d for d in dead if d.items <= items and not d.items <= running), None)
