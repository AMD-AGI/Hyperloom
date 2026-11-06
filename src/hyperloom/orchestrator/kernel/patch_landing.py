# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Landing N sibling patches from one nomination without cross-contamination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from hyperloom_common.coerce import optional_positive_int, to_float

#: How many patches one nomination round may land by default. The integrate lane
#: is serial -- benchmark capacity one, ~25 minutes modelled per patch -- so this
#: is a wall-clock ceiling, not a preference. Three patches ~= 75 minutes of
#: serial GPU, which is about what one macro cycle can afford.
DEFAULT_PATCH_BUDGET = 3

#: Terminal records kept for triage are capped at this multiple of the patch
#: budget; beyond it the weakest (lowest ``micro_speedup``) are dropped. Two
#: rounds' worth is enough context to explain a decision without the dict growing
#: without bound.
TERMINAL_RETENTION_MULTIPLE = 2

#: Terminal statuses that record a settled judgement on the patch itself. The
#: remaining terminal, ``dispatch_failed``, is a fault: the drain crashed before
#: any judgement, so that record may be re-offered for another attempt.
VERDICT_STATUSES = frozenset({"integrated", "rejected"})

#: Statuses that mean the record will never be dispatched again. Pending is the
#: only live state; everything else is eligible for eviction.
TERMINAL_STATUSES = VERDICT_STATUSES | {"dispatch_failed"}

#: Stack labels whose KEEP overwrote a whole kernel source file, so a queued
#: patch on that file can no longer be measured on its own. Writeback labels a
#: lane's KEEP by its own name, so a label missing here re-drains a spent patch.
INTEGRATING_STACK_ACTIONS = frozenset({"integrate", "fusion"})


def record_source_path(record: Mapping[str, Any]) -> str:
    """The one spelling of a record's source path, read the same everywhere."""
    if not isinstance(record, Mapping):
        return ""
    return str(record.get("target_file") or record.get("source_file") or "").strip()


def record_matches_task(
    record: Mapping[str, Any],
    *,
    kernel_id: str,
    task_group_key: str,
    source_file: str,
    task_group_aliases: set[str] | None = None,
) -> bool:
    """Match persisted integration state to the current stable task identity."""
    recorded_key = str(record.get("task_group_key") or "")
    if task_group_key and recorded_key:
        accepted_keys = {task_group_key, *(task_group_aliases or set())}
        return recorded_key in accepted_keys
    if str(record.get("kernel_id") or "") != kernel_id:
        return False
    recorded_source = record_source_path(record)
    if source_file and recorded_source:
        return source_file == recorded_source
    return True


@dataclass(frozen=True)
class IntegratedStack:
    """The optimization_stack rows whose KEEP already landed a whole kernel source file."""

    _entries: tuple[Mapping[str, Any], ...]
    _source_files: frozenset[str]

    @classmethod
    def of(cls, optimization_stack: Iterable[Any] | None) -> IntegratedStack:
        entries = tuple(
            entry
            for entry in (optimization_stack or [])
            if isinstance(entry, Mapping) and entry.get("action") in INTEGRATING_STACK_ACTIONS
        )
        return cls(entries, frozenset(path for path in map(record_source_path, entries) if path))

    def covers(
        self,
        *,
        kernel_ids: Iterable[str],
        task_group_key: str,
        source_file: str,
        task_group_aliases: set[str] | None = None,
    ) -> bool:
        """Whether a landed row already retired this task, by identity or by owning its source file."""
        if source_file and source_file in self._source_files:
            return True
        return any(
            record_matches_task(
                entry,
                kernel_id=kernel_id,
                task_group_key=task_group_key,
                source_file=source_file,
                task_group_aliases=task_group_aliases,
            )
            for kernel_id in kernel_ids
            for entry in self._entries
        )


def patch_budget(configured: object) -> int:
    """Resolve the per-round patch ceiling, never below one."""
    return optional_positive_int(configured, DEFAULT_PATCH_BUDGET)


def evict_terminal(
    queue: Mapping[str, Any],
    *,
    budget: int = DEFAULT_PATCH_BUDGET,
    retention_multiple: int = TERMINAL_RETENTION_MULTIPLE,
) -> dict[str, Any]:
    """Return the queue with stale terminal records reaped."""
    if not isinstance(queue, Mapping):
        return {}
    live: dict[str, Any] = {}
    terminal: list[tuple[str, dict[str, Any]]] = []
    for integration_id, record in queue.items():
        if not isinstance(record, Mapping):
            # Non-dict entries carry no lifecycle we can reason about; keep them verbatim rather than silently
            # discarding foreign state.
            live[str(integration_id)] = record
            continue
        status = str(record.get("status") or "pending")
        if status not in TERMINAL_STATUSES:
            live[str(integration_id)] = dict(record)
        else:
            terminal.append((str(integration_id), dict(record)))
    keep = max(0, int(budget)) * max(0, int(retention_multiple))
    if len(terminal) > keep:
        # Weakest first, so the strongest survivors are the most informative.
        terminal.sort(key=lambda item: to_float(item[1].get("micro_speedup"), 0.0))
        terminal = terminal[len(terminal) - keep :] if keep else []
    for integration_id, record in terminal:
        live[integration_id] = record
    return live


def bundle_belongs_to(bundle: Mapping[str, Any], integration_id: object) -> bool:
    """Whether a recorded artifact bundle may be merged into this integrate."""
    if not isinstance(bundle, Mapping):
        return False
    resolved_id = str(integration_id or "").strip()
    bundle_id = str(bundle.get("integration_id") or "").strip()
    if not resolved_id or not bundle_id:
        return True
    return bundle_id == resolved_id
