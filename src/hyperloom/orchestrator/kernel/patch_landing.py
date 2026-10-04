# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Landing N sibling patches from one nomination without cross-contamination."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

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


def record_source_path(record: Mapping[str, Any]) -> str:
    """The one spelling of a record's source path, read the same everywhere."""
    if not isinstance(record, Mapping):
        return ""
    return str(record.get("target_file") or record.get("source_file") or "").strip()


def patch_budget(configured: object = None, *, default: int = DEFAULT_PATCH_BUDGET) -> int:
    """Resolve the per-round patch ceiling, never below one."""
    value = _positive_int(configured)
    if value is not None:
        return value
    resolved = _positive_int(default)
    return resolved if resolved is not None else DEFAULT_PATCH_BUDGET


def clamp_by_budget(
    records: Iterable[Mapping[str, Any]],
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ranked pending records into the ones that fit and the deferred rest."""
    rows = [dict(row) for row in records if isinstance(row, Mapping)]
    ceiling = max(0, int(budget))
    return rows[:ceiling], rows[ceiling:]


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
        terminal.sort(key=lambda item: _micro(item[1]))
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


def _micro(record: Mapping[str, Any]) -> float:
    """A record's micro speedup for ranking; unusable values sort weakest."""
    try:
        value = float(record.get("micro_speedup") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value


def _positive_int(value: object) -> int | None:
    """Coerce to a positive int, or ``None`` when that is not possible."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
