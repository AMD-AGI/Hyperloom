# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Hyperloom reads back when forge picked the kernels itself.

Three rules shape this module, all of them about not letting one bad entry
cost a whole round:

1. A patch missing any of its three required fields is dropped; its siblings
   still land. The batch is N independent results, not a transaction.
2. Duplicate kernel names collapse to the strongest one, because the ledger
   keys on the name -- duplicates would share one retry budget and one
   rejection.
3. An empty ``patches`` array with a clean exit is a valid answer, not a
   failure. Phase exit is a latch, not an inference from emptiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Dropping an entry is a contract problem worth surfacing, so every drop
#: carries one of these instead of vanishing into a log line.
DROP_MISSING_KERNEL_NAME = "missing_kernel_name"
DROP_MISSING_PATCH_PATH = "missing_patch_path"
DROP_MISSING_TARGET_FILE = "missing_target_file"
DROP_DUPLICATE_KERNEL_NAME = "duplicate_kernel_name"
DROP_NOT_AN_OBJECT = "not_an_object"


@dataclass(frozen=True)
class NominatedPatch:
    """One patch from a self-nominating run, ready for the integrate queue."""

    kernel_name: str
    patch_path: str
    target_file: str
    kernel_repo: str = ""
    snapshot_dir: str = ""
    base_commit: str = ""
    micro_speedup: float = 0.0
    write_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DroppedPatch:
    """One entry that could not be used, kept so the drop is reportable."""

    reason: str
    kernel_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NominationOutcome:
    """The whole of what one forge call returned."""

    patches: tuple[NominatedPatch, ...] = ()
    dropped: tuple[DroppedPatch, ...] = ()
    candidates_seen: int = 0
    resolved: int = 0
    selected: int = 0

    @property
    def is_empty(self) -> bool:
        """No usable patch. A valid outcome, and not the same as a failure."""
        return not self.patches


def parse_outcome(payload: Any) -> NominationOutcome:
    """Read a forge result envelope into usable patches plus explained drops.

    Never raises on content: a malformed entry becomes a drop so the rest of
    the batch survives. Only a non-mapping envelope yields an empty outcome.

    Args:
        payload: The parsed forge result envelope.

    Returns:
        The outcome, with patches ordered strongest-first by micro speedup.
    """
    if not isinstance(payload, dict):
        return NominationOutcome()
    entries = payload.get("patches")
    rows = entries if isinstance(entries, list) else []
    kept: dict[str, NominatedPatch] = {}
    dropped: list[DroppedPatch] = []
    for row in rows:
        patch, reason = _read_entry(row)
        if patch is None:
            dropped.append(
                DroppedPatch(
                    reason=reason,
                    kernel_name=str(row.get("kernel_name") or "") if isinstance(row, dict) else "",
                    raw=row if isinstance(row, dict) else {},
                )
            )
            continue
        incumbent = kept.get(patch.kernel_name)
        if incumbent is None:
            kept[patch.kernel_name] = patch
            continue
        # Same name twice would share one ledger key. Keep the stronger claim
        # and report the other rather than letting either win silently.
        weaker = patch if patch.micro_speedup <= incumbent.micro_speedup else incumbent
        stronger = incumbent if weaker is patch else patch
        kept[patch.kernel_name] = stronger
        dropped.append(
            DroppedPatch(
                reason=DROP_DUPLICATE_KERNEL_NAME,
                kernel_name=weaker.kernel_name,
                raw={"patch_path": weaker.patch_path},
            )
        )
    summary = payload.get("nomination")
    summary = summary if isinstance(summary, dict) else {}
    ordered = sorted(kept.values(), key=lambda patch: patch.micro_speedup, reverse=True)
    return NominationOutcome(
        patches=tuple(ordered),
        dropped=tuple(dropped),
        candidates_seen=_int_or_zero(summary.get("candidates_seen")),
        resolved=_int_or_zero(summary.get("resolved")),
        selected=_int_or_zero(summary.get("selected")),
    )


def _read_entry(row: Any) -> tuple[NominatedPatch | None, str]:
    """Validate one entry, returning either a patch or why it was dropped."""
    if not isinstance(row, dict):
        return None, DROP_NOT_AN_OBJECT
    kernel_name = str(row.get("kernel_name") or "").strip()
    if not kernel_name:
        return None, DROP_MISSING_KERNEL_NAME
    patch_path = str(row.get("patch_path") or "").strip()
    if not patch_path:
        return None, DROP_MISSING_PATCH_PATH
    # Interchangeable on the return path: integrate reads whichever is set.
    target_file = str(row.get("target_file") or row.get("source_file") or "").strip()
    if not target_file:
        return None, DROP_MISSING_TARGET_FILE
    return (
        NominatedPatch(
            kernel_name=kernel_name,
            patch_path=patch_path,
            target_file=target_file,
            kernel_repo=str(row.get("kernel_repo") or row.get("repo_root") or "").strip(),
            snapshot_dir=str(row.get("snapshot_dir") or "").strip(),
            base_commit=str(row.get("base_commit") or "").strip(),
            micro_speedup=_finite_float(row.get("micro_speedup")),
            write_paths=_string_tuple(row.get("write_paths")),
        ),
        "",
    )


def _finite_float(value: Any) -> float:
    """Non-finite or non-numeric speedups sort last rather than crashing."""
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _int_or_zero(value: Any) -> int:
    """A missing or unparsable count reports zero; counts are advisory."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Keep non-empty strings only; the import-confirmation gate reads these."""
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item or "").strip())
