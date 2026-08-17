# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-round staging of the kernel agent's KB sub-columns.

The CLOSE-time scrape in :mod:`remote_recipe.values` reverse-engineers the
``kernel.{gemm,fusion,rewrite}`` columns from the final ``SharedState``. This
module lets the kernel paths stage the *same* columns incrementally through
:class:`KernelAgentKB`, so a run owns its columns explicitly and carries the
per-patch outcome/gain the scrape leaves implicit.

Field selection deliberately mirrors ``values.build_kernel_{gemm,fusion,
rewrite}_value`` so a staged column and a scraped one agree in shape; the one
difference is transport: the facade stages local files and returns refs, so the
builders here hand back ``(knowledge, file_sources, fold)`` and let the driver
run the facade's two-write dance (stage files -> fold refs -> replace).

Every call is best-effort. A run with no draft directory leaves
:class:`KernelAgentKB` inactive and turns the whole module into a no-op, and no
failure here is allowed to reach the kernel optimization it records.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .agent_kb import KernelAgentKB
from .remote_recipe.values import match_rewrite_attempt

log = logging.getLogger(__name__)

# Local-path keys stripped from a scraped record before staging; the refs that
# replace them are folded back in after the files are staged. Mirrors
# ``values._PATH_KEYS`` / ``values._PATH_LIST_KEYS``.
_LOCAL_PATH_KEYS = frozenset(
    {
        "artifact_path",
        "artifact_files",
        "artifacts",
        "changed_files",
        "final_report_path",
        "patch",
        "patch_path",
        "patches",
        "patches_applied",
        "report_path",
        "source_file",
        "source_files",
        "target_file",
        "target_files",
        "tuned_file",
    }
)

# A column builder returns the pre-file knowledge map, the ordered list of local
# file sources to stage, and a fold that reinserts the returned refs.
_Built = tuple[dict[str, Any], list[str], Callable[[list[str]], dict[str, Any]]]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _readable_file(value: Any) -> str:
    """Return a real, non-symlink file path, or '' when it cannot be staged."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    return raw if path.is_file() and not path.is_symlink() else ""


def _scrub(record: Mapping[str, Any]) -> dict[str, Any]:
    """Drop host-local path fields; the fold reinserts the ones we manage."""
    return {k: v for k, v in record.items() if k not in _LOCAL_PATH_KEYS}


def _stack_rows(state: Any, action: str) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in (getattr(state, "optimization_stack", []) or [])
        if isinstance(item, Mapping) and str(item.get("action") or "").lower() == action
    ]


def build_gemm(state: Any) -> _Built | None:
    """Stage accepted GEMM optimizations (params + optional tuned file)."""
    rows = _stack_rows(state, "gemm_tuning")
    last = _mapping(getattr(state, "last_gemm_tuning", {}))
    accepted_last = str(last.get("decision") or "").upper() == "KEEP" or str(
        last.get("status") or ""
    ).lower() == "kept"
    if rows and accepted_last:
        rows[-1] = {**last, **rows[-1]}
    if not rows:
        return None
    records: list[dict[str, Any]] = []
    sources: list[str] = []
    slots: list[int | None] = []
    for row in rows:
        record = _scrub(row)
        record.setdefault("phase", row.get("phase") or "KERNEL_AGENT")
        tuned = _readable_file(row.get("tuned_file"))
        if tuned:
            slots.append(len(sources))
            sources.append(tuned)
        else:
            slots.append(None)
        records.append(record)

    def fold(refs: list[str]) -> dict[str, Any]:
        for slot, record in zip(slots, records):
            if slot is not None and slot < len(refs):
                record["tuned_file"] = refs[slot]
                if not refs[slot]:
                    record.pop("tuned_file", None)
        return {"optimizations": records, "files": [ref for ref in refs if ref]}

    return ({"optimizations": records}, sources, fold)


def build_fusion(state: Any) -> _Built | None:
    """Stage one E2E-accepted fusion solution (patch + target + e2e verdict)."""
    rows = _stack_rows(state, "fusion")
    result = _mapping(getattr(state, "last_fusion", {}))
    integrated = _mapping(getattr(state, "last_fusion_integrate", {}))
    if not rows or str(integrated.get("decision") or "").upper() != "KEEP":
        return None
    patch = _readable_file(rows[-1].get("patch_path") or result.get("patch"))
    target = _readable_file(rows[-1].get("target_file") or result.get("source_file"))
    if not patch or not target:
        return None
    record = {
        **_scrub(result),
        **_scrub(rows[-1]),
        "phase": str(rows[-1].get("phase") or "KERNEL_AGENT"),
        "e2e": _scrub(integrated),
    }

    def fold(refs: list[str]) -> dict[str, Any]:
        patch_ref = refs[0] if refs else ""
        source_ref = refs[1] if len(refs) > 1 else ""
        if not patch_ref or not source_ref:
            return {"items": [], "files": []}
        record["patch"] = patch_ref
        record["source_file"] = source_ref
        return {"items": [record], "files": [patch_ref, source_ref]}

    return ({"items": [record]}, [patch, target], fold)


def build_rewrite(state: Any) -> _Built | None:
    """Stage E2E-integrated rewrite rows (speedup/gain + patch + source)."""
    attempts = getattr(state, "kernel_opt_task_attempts", {}) or {}
    if not isinstance(attempts, Mapping):
        attempts = {}
    integrated = _stack_rows(state, "integrate")
    items: list[dict[str, Any]] = []
    sources: list[str] = []
    slots: list[tuple[int, int]] = []
    for entry in integrated:
        raw = match_rewrite_attempt(entry, attempts)
        patch = _readable_file(
            entry.get("patch_path") or raw.get("last_artifact_path") or raw.get("artifact_path")
        )
        source = _readable_file(
            entry.get("target_file") or raw.get("last_source_file") or raw.get("source_file")
        )
        # Per-round staging is tolerant: an entry missing its files is skipped
        # rather than raising, unlike the CLOSE scrape which owns validation.
        if not patch or not source:
            continue
        kernel_name = str(
            raw.get("kernel_name")
            or raw.get("current_kernel_id")
            or raw.get("kernel_id")
            or entry.get("kernel_id")
            or "unknown"
        )
        item = {
            "id": str(
                entry.get("integration_id")
                or entry.get("task_group_key")
                or entry.get("kernel_id")
                or kernel_name
            ),
            "phase": "KERNEL_AGENT",
            "kernel_name": kernel_name,
            "speedup": _number(raw.get("last_micro_speedup") or raw.get("speedup")),
            "e2e_gain_pct": _number(entry.get("gain_pct")),
            "optimized_throughput": _number(entry.get("tput")),
        }
        slots.append((len(sources), len(sources) + 1))
        sources.extend((patch, source))
        items.append(item)
    if not items:
        return None

    def fold(refs: list[str]) -> dict[str, Any]:
        # refs is positional over ``sources`` — an empty slot means that artifact
        # never staged, so the item it belongs to is dropped rather than
        # published pointing at a neighbour's file.
        folded: list[dict[str, Any]] = []
        for item, (patch_slot, source_slot) in zip(items, slots):
            patch = refs[patch_slot] if patch_slot < len(refs) else ""
            source = refs[source_slot] if source_slot < len(refs) else ""
            if not patch or not source:
                continue
            item["patch"] = patch
            item["source_files"] = [source]
            folded.append(item)
        return {"items": folded, "files": [ref for ref in refs if ref]}

    return ({"items": items}, sources, fold)


_BUILDERS: tuple[tuple[str, str, Callable[[Any], _Built | None]], ...] = (
    ("gemm", "write_gemm", build_gemm),
    ("fusion", "write_fusion", build_fusion),
    ("rewrite", "write_rewrite", build_rewrite),
)


def stage_kernel_columns(state: Any, *, kb: KernelAgentKB | None = None) -> dict[str, Any]:
    """Stage every non-empty kernel sub-column via :class:`KernelAgentKB`.

    Best-effort and idempotent: a write replaces its own sub-column, so
    re-staging after each round keeps the draft in step with the live state.
    Returns a small summary for logging/tests; never raises — a staging failure
    is logged at warning level rather than surfaced, because knowledge is
    advisory and must not fail the optimization round that produced it.
    """
    facade = kb or KernelAgentKB.open()
    if not facade.active:
        return {"active": False}
    summary: dict[str, Any] = {"active": True}
    for column, method, builder in _BUILDERS:
        try:
            built = builder(state)
        except Exception:  # noqa: BLE001 — a bad record must not break the round
            log.warning("kernel kb: building %s column failed", column, exc_info=True)
            continue
        if built is None:
            continue
        knowledge, sources, fold = built
        try:
            write = getattr(facade, method)
            refs = write(knowledge, files=sources)
            write(fold(refs))
            summary[column] = {"refs": refs}
        except Exception:  # noqa: BLE001 — knowledge is advisory
            log.warning("kernel kb: staging %s column failed", column, exc_info=True)
    return summary


__all__ = [
    "build_fusion",
    "build_gemm",
    "build_rewrite",
    "stage_kernel_columns",
]
