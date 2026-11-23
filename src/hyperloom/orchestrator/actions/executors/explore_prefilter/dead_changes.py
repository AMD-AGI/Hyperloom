# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The changes this session saw stop a server before it was ready, and the variants that would launch one again.

Within one session the host and build stay fixed, so such a change kills any launch that carries it whole; the
experience corpus's attribution is left as conservative as ``classify_failure_attribution`` makes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from inference_suite.server_log import cause_line
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

# The root cause a server that exited before it was ready reads as when nothing names memory, capacity or the host.
_ENGINE_INIT = "engine_core_init"


@dataclass(frozen=True)
class DeadChange:
    """A change that stopped a server before it was ready, the variant it came from and the cause its log named."""

    name: str
    round_id: str
    items: frozenset[tuple[str, ...]]
    cause: str

    def detail(self) -> str:
        cause = f": {self.cause}" if self.cause else ""
        return (
            f"carries the whole change of {self.name} ({self.round_id}), whose server died before it was ready{cause}"
        )


def _items_of(change: Mapping[str, Any]) -> frozenset[tuple[str, ...]]:
    return change_items(
        str(change.get("extra_server_args") or ""),
        change.get("extra_envs") or {},
        remove_args=change.get("remove_args"),
        unset_envs=change.get("unset_envs"),
        args_mode=str(change.get("args_mode") or "append"),
    )


def _died_at_init(row: Mapping[str, Any]) -> bool:
    """Whether the row's server exited before it was ready on its own change, not on memory, capacity or the host."""
    error_class = str(row.get("error_class") or "")
    excerpt = failure_text(row)
    if error_class != SERVER_INIT_DEAD or row.get("framework_changed"):
        return False
    if classify_root_cause_type(error_class, excerpt) != _ENGINE_INIT:
        return False
    attribution = classify_failure_attribution(error_class=error_class, framework_changed=False, error_excerpt=excerpt)
    return attribution != FAILURE_ATTRIBUTION_ENVIRONMENT


def dead_changes(rows: Iterable[Mapping[str, Any]], *, workload_signature: str) -> list[DeadChange]:
    """The changes the ``rows`` of this workload saw stop a server before it was ready, oldest first."""
    out: list[DeadChange] = []
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("outcome") or "") != TS_FAILED:
            continue
        if str(row.get("workload_signature") or "") != workload_signature:
            continue
        items = _items_of(row)
        if items and _died_at_init(row):
            out.append(
                DeadChange(
                    name=str(row.get("name") or "?"),
                    round_id=str(row.get("round_id") or "an earlier round"),
                    items=items,
                    cause=cause_line(failure_text(row)) or "",
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
