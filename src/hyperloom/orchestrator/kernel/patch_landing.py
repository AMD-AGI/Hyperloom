# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Landing N sibling patches from one nomination without cross-contamination.

The queue and its bookkeeping were built when a lane returned exactly one patch
per round. Once forge nominates several kernels in a single call, that batch
lands as a set of *siblings* -- independent patches that share nothing but the
round they came from. Five assumptions from the one-patch era turn into bugs the
moment a second sibling exists, and they are the reason this module exists:

* **The source-file key had two spellings.** The write side recorded
  ``target_file or source_file``; the read side compared only ``source_file``.
  When a record's real path was stored under ``target_file`` the read saw an
  empty string, and same-source exclusion silently no-opped -- so a second
  whole-file overwrite of the same file could slip through. One spelling,
  :func:`record_source_path`, is now used on both sides.
* **The queue never shrank.** Terminal records (integrated / rejected /
  dispatch-failed) were only ever status-flipped, never removed, so the dict
  grew without bound and was fully rescanned every round. :func:`evict_terminal`
  reaps them, keeping at most a bounded tail for post-mortem.
* **There was no ceiling on how many patches one round could push.**
  :func:`patch_budget` and :func:`clamp_by_budget` cap the batch so a run of
  low-value proposals cannot swamp the integrate lane, whose serial cost is real.

Everything here is a pure function over plain dicts so it can be tested without
standing up a SharedState. The callers in ``_kernel_decisions`` and
``kernel_stack`` forward to these; they hold no logic of their own.
"""

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

#: Statuses that mean the record will never be dispatched again. Pending is the
#: only live state; everything else is eligible for eviction.
TERMINAL_STATUSES = frozenset({"integrated", "rejected", "dispatch_failed"})


def record_source_path(record: Mapping[str, Any]) -> str:
    """The one spelling of a record's source path, read the same everywhere.

    Both a queued integration record and an optimization-stack entry may carry
    the path under ``target_file`` or ``source_file`` depending on which producer
    wrote it. Reading only one of the two is how same-source exclusion used to
    fail silently. Callers on both the write and read sides go through here so the
    two can never disagree again.

    Args:
        record: Any mapping that might carry a source path.

    Returns:
        The resolved path, or ``""`` when neither field is set.
    """
    if not isinstance(record, Mapping):
        return ""
    return str(record.get("target_file") or record.get("source_file") or "").strip()


def patch_budget(configured: object = None, *, default: int = DEFAULT_PATCH_BUDGET) -> int:
    """Resolve the per-round patch ceiling, never below one.

    Args:
        configured: An override (env value, payload field, ...). ``None`` or an
            unparsable value falls back to ``default``.
        default: The ceiling when nothing is configured.

    Returns:
        A positive integer ceiling.
    """
    value = _positive_int(configured)
    if value is not None:
        return value
    resolved = _positive_int(default)
    return resolved if resolved is not None else DEFAULT_PATCH_BUDGET


def clamp_by_budget(
    records: Iterable[Mapping[str, Any]],
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ranked pending records into the ones that fit and the deferred rest.

    The input is assumed already ordered strongest-first (impact, then micro
    speedup) by the caller. Everything past the budget is *deferred*, not
    dropped: it stays pending and is reconsidered next macro cycle rather than
    being thrown away, because a patch below this round's cut may be above the
    next round's.

    Args:
        records: Pending records, strongest first.
        budget: How many may land this round.

    Returns:
        ``(fit, deferred)`` -- the leading ``budget`` records and the remainder.
    """
    rows = [dict(row) for row in records if isinstance(row, Mapping)]
    ceiling = max(0, int(budget))
    return rows[:ceiling], rows[ceiling:]


def evict_terminal(
    queue: Mapping[str, Any],
    *,
    budget: int = DEFAULT_PATCH_BUDGET,
    retention_multiple: int = TERMINAL_RETENTION_MULTIPLE,
) -> dict[str, Any]:
    """Return the queue with stale terminal records reaped.

    Pending records are always kept -- they are live work. Terminal records
    (integrated / rejected / dispatch-failed) are kept only up to
    ``budget * retention_multiple`` for post-mortem, and when there are more the
    weakest by ``micro_speedup`` are dropped first. This is the queue's only
    deletion point; without it the dict grew every round and was rescanned in
    full each time.

    Args:
        queue: The ``pending_kernel_integrations`` mapping.
        budget: The per-round patch budget the retention cap scales from.
        retention_multiple: How many budgets' worth of terminal records to keep.

    Returns:
        A new dict safe to assign back onto state.
    """
    if not isinstance(queue, Mapping):
        return {}
    live: dict[str, Any] = {}
    terminal: list[tuple[str, dict[str, Any]]] = []
    for integration_id, record in queue.items():
        if not isinstance(record, Mapping):
            # Non-dict entries carry no lifecycle we can reason about; keep them
            # verbatim rather than silently discarding foreign state.
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
    """Whether a recorded artifact bundle may be merged into this integrate.

    A bundle is stamped with the ``integration_id`` of the sibling that produced
    it. When the integrate being resolved names a specific sibling, only that
    sibling's bundle -- or a legacy bundle that carries no id at all -- may be
    merged; a bundle stamped with a *different* id belongs to another sibling of
    the same nomination round and would otherwise land its write set under the
    wrong integrate. When the integrate names no sibling (legacy, id-less path)
    any bundle is accepted, matching the one-patch-era behaviour where there was
    only ever one bundle per kernel to choose from.

    Args:
        bundle: The candidate artifact bundle.
        integration_id: The id of the sibling this integrate resolved to, if any.

    Returns:
        ``True`` when the bundle is safe to merge, ``False`` when it is a
        cross-sibling bundle that must be refused.
    """
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
