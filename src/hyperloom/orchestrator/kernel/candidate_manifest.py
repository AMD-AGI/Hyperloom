# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hot-kernel list as handed to a self-nominating forge.

Three things separate this from ``kernel_candidates.json`` itself:

* **Undispatchable rows are kept.** The dispatch path filters them out before
  forge is ever called, which is how a hot kernel disappears with no error and
  no counter. A nominator that never sees those rows cannot rescue them, and
  cannot judge how much of the GPU is out of its reach.
* **Every row carries a judgeable class.** The routing gate's prose is reduced
  to ``reason_class`` by the shared classifier, so a consumer can skip the
  classes nothing can rescue without measuring them.
* **Session history is merged in.** How many times a kernel was already tried,
  and whether it was rejected, live in orchestrator state -- forge has no way to
  know either.

This is a projection, not a replacement: ``kernel_candidates.json`` keeps its
shape and its existing consumers, and the classifier is the same pure function
the resolution artifact uses, so the two can never disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.kernel_source_contract import classify_skip_reason

#: Bump on any field removal or meaning change. ``kernelforge.nomination``
#: refuses a version it does not know, so both halves move together.
MANIFEST_VERSION = 2

MANIFEST_FILENAME = "forge_candidate_manifest.json"


class CandidateManifestError(ValueError):
    """A manifest that cannot be built is a contract violation, not a skip."""


@dataclass(frozen=True)
class ManifestStats:
    """What went into one manifest, for the log line and for triage."""

    total: int
    resolved: int
    rejected: int
    undispatchable_gpu_pct: float


def build_manifest(
    candidates_path: str | Path,
    *,
    rejected_kernel_ids: object = (),
    attempts_by_kernel_id: object = None,
    trace_path: str = "",
    trace_captured_after: str = "",
) -> tuple[dict[str, Any], ManifestStats]:
    """Project the candidate artifact onto the list forge receives.

    Args:
        candidates_path: Path to ``kernel_candidates.json``.
        rejected_kernel_ids: Kernel ids already rejected this session.
        attempts_by_kernel_id: Attempt counts keyed by kernel id.
        trace_path: Trace the rows were derived from, recorded for triage.
        trace_captured_after: Code-state marker for that trace.

    Returns:
        The manifest document and its stats.

    Raises:
        CandidateManifestError: When the artifact is unreadable or carries no
            recognizable row array. An empty array is valid: it means the trace
            found nothing, which a nominator should be told rather than guess.
    """
    payload = _load(candidates_path)
    rows = payload.get("hot_kernels")
    if rows is None:
        rows = payload.get("hot_kernels_top15")
    if not isinstance(rows, list):
        raise CandidateManifestError(f"candidate artifact has no hot_kernels array: {candidates_path}")

    rejected = {str(item).strip() for item in _as_iterable(rejected_kernel_ids) if str(item or "").strip()}
    attempts = attempts_by_kernel_id if isinstance(attempts_by_kernel_id, dict) else {}

    entries: list[dict[str, Any]] = []
    resolved = 0
    undispatchable_gpu_pct = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry = _project_row(row, rejected=rejected, attempts=attempts)
        if entry is None:
            continue
        entries.append(entry)
        if entry["reason_class"] == "resolved":
            resolved += 1
        else:
            undispatchable_gpu_pct += entry["gpu_pct"]

    document = {
        "manifest_version": MANIFEST_VERSION,
        "trace_path": str(trace_path or ""),
        "trace_captured_after": str(trace_captured_after or ""),
        # Named to match what the consumer reads; this is the full set, including
        # rows the dispatch path would have dropped.
        "hot_kernels": entries,
    }
    stats = ManifestStats(
        total=len(entries),
        resolved=resolved,
        rejected=sum(1 for entry in entries if entry["rejected"]),
        undispatchable_gpu_pct=undispatchable_gpu_pct,
    )
    return document, stats


def write_manifest(directory: Path, document: dict[str, Any]) -> Path:
    """Persist a manifest as UTF-8 JSON and return its path."""
    from hyperloom.common.io import atomic_write_json

    path = Path(directory) / MANIFEST_FILENAME
    atomic_write_json(path, document, ensure_ascii=False)
    return path


def _project_row(
    row: dict[str, Any],
    *,
    rejected: set[str],
    attempts: dict,
) -> dict[str, Any] | None:
    """Reduce one candidate row to what a nominator needs, or drop it."""
    kernel_id = str(row.get("kernel_id") or "").strip()
    name = str(row.get("name") or "").strip()
    # A row with neither identity cannot be reported back against, so it could
    # never become a patch even if it were nominated.
    if not (kernel_id or name):
        return None
    kernel_name = name or kernel_id
    return {
        # Provenance only: reranking reassigns the ordinal between operators.
        "kernel_id": kernel_id,
        # The accounting identity on this lane, and what the consumer keys on.
        "kernel_name": kernel_name,
        "gpu_pct": _finite_float(row.get("gpu_pct")),
        "duration_us": _finite_float(row.get("duration_us")),
        "call_count": _non_negative_int(row.get("call_count")),
        "source_file": str(row.get("source_file") or "").strip(),
        "reason_class": classify_skip_reason(
            reusable=row.get("reusable_native_kernel"),
            skip_reason=row.get("skip_reason"),
            source_file=row.get("source_file"),
        ),
        "resolution_method": str(row.get("source_resolution_method") or "").strip(),
        "resolution_reason": str(row.get("source_resolution_reason") or "").strip(),
        "kernel_category": str(row.get("kernel_category") or "").strip(),
        "kernel_repo": str(row.get("kernel_repo") or "").strip(),
        "patch_strategy": str(row.get("patch_strategy") or "").strip(),
        "shapes": row.get("shapes") if isinstance(row.get("shapes"), list) else [],
        "trace_report_path": str(row.get("trace_report_path") or "").strip(),
        # Orchestrator-only knowledge: forge cannot derive either of these.
        # Keyed on the name this lane records under; the ordinal is the fallback
        # that keeps the legacy selector's history readable.
        "attempts": _non_negative_int(_attempts_for(attempts, kernel_name, kernel_id)),
        "rejected": bool((kernel_name and kernel_name in rejected) or (kernel_id and kernel_id in rejected)),
    }


def _attempts_for(attempts: dict, *keys: str) -> Any:
    """Read an attempt count under the first key that has one.

    Each key may hold a bare number or a ledger entry. Keys are tried in the
    order given, so the caller decides which identity is authoritative.
    """
    for key in keys:
        if not key:
            continue
        value = attempts.get(key)
        if isinstance(value, dict):
            value = value.get("attempts", 0)
        if value is not None:
            return value
    return 0


def _load(path: str | Path) -> dict[str, Any]:
    """Read the candidate artifact, turning every failure into one error."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateManifestError(f"could not read candidate artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CandidateManifestError(f"candidate artifact must be a JSON object: {path}")
    return payload


def _as_iterable(value: object) -> tuple:
    """Accept a list, tuple or set of ids; anything else means none."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return ()


def _finite_float(value: object) -> float:
    """Non-finite or non-numeric measurements rank as zero rather than crashing."""
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _non_negative_int(value: object) -> int:
    """A missing or unusable counter reads as zero; counters are advisory."""
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, number)
