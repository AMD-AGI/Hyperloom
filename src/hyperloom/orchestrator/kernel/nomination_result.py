# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What Hyperloom reads back when forge picked the kernels itself."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


log = logging.getLogger(__name__)

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
    #: Space-joined env flag(s) that gate the fused path. An authored fusion is
    #: env-gated: unset, the patched code is bit-for-bit the eager path, so the
    #: flag has to reach the e2e re-baseline or integrate measures the un-fused
    #: path and REVERTs a real win. Empty for a self-activating compile pass.
    env_flag: str = ""


@dataclass(frozen=True)
class DroppedPatch:
    """One entry that could not be used, kept so the drop is reportable."""

    reason: str
    kernel_name: str = ""


@dataclass(frozen=True)
class NominationOutcome:
    """The whole of what one forge call returned."""

    patches: tuple[NominatedPatch, ...] = ()
    dropped: tuple[DroppedPatch, ...] = ()
    #: Why the envelope itself could not be trusted, empty when it could. An
    #: envelope that is unreadable is not a nomination that selected nothing.
    schema_error: str = ""

    @property
    def is_empty(self) -> bool:
        """No usable patch. A valid outcome, and not the same as a failure."""
        return not self.patches


def summarize_dropped_patches(dropped: Any) -> dict[str, int]:
    """Count dropped patch entries by reason, logging each one as it is counted.

    ``parse_outcome`` names why every unusable entry was refused, but a reason
    nobody reports is indistinguishable from forge never having offered the
    entry at all.
    """
    counts: dict[str, int] = {}
    for entry in dropped or ():
        reason = str(getattr(entry, "reason", "") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
        log.warning(
            "nomination landing: refused patch kernel=%s reason=%s",
            str(getattr(entry, "kernel_name", "") or "<unnamed>"),
            reason,
        )
    return counts


def parse_outcome(payload: Any) -> NominationOutcome:
    """Read a forge result envelope into usable patches plus explained drops."""
    if not isinstance(payload, dict):
        return NominationOutcome(schema_error=f"result envelope is {type(payload).__name__}, not an object")
    entries = payload.get("patches")
    if entries is None:
        return NominationOutcome(schema_error="result envelope carries no patches array")
    if not isinstance(entries, list):
        return NominationOutcome(schema_error=f"patches is {type(entries).__name__}, not an array")
    kept: dict[str, NominatedPatch] = {}
    dropped: list[DroppedPatch] = []
    for row in entries:
        patch, reason = _read_entry(row)
        if patch is None:
            dropped.append(
                DroppedPatch(
                    reason=reason,
                    kernel_name=str(row.get("kernel_name") or "") if isinstance(row, dict) else "",
                )
            )
            continue
        incumbent = kept.get(patch.kernel_name)
        if incumbent is None:
            kept[patch.kernel_name] = patch
            continue
        # Same name twice would share one ledger key.
        weaker = patch if patch.micro_speedup <= incumbent.micro_speedup else incumbent
        stronger = incumbent if weaker is patch else patch
        kept[patch.kernel_name] = stronger
        dropped.append(DroppedPatch(reason=DROP_DUPLICATE_KERNEL_NAME, kernel_name=weaker.kernel_name))
    ordered = sorted(kept.values(), key=lambda patch: patch.micro_speedup, reverse=True)
    # Offered patches of which none survived: forge believed it nominated, so this is a contract disagreement rather
    # than a round that selected nothing.
    schema_error = (
        f"every one of the {len(entries)} offered patch entries was refused" if entries and not ordered else ""
    )
    return NominationOutcome(
        patches=tuple(ordered),
        dropped=tuple(dropped),
        schema_error=schema_error,
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
    target_file = str(row.get("target_file") or "").strip()
    if not target_file:
        return None, DROP_MISSING_TARGET_FILE
    return (
        NominatedPatch(
            kernel_name=kernel_name,
            patch_path=patch_path,
            target_file=target_file,
            kernel_repo=str(row.get("kernel_repo") or "").strip(),
            snapshot_dir=str(row.get("snapshot_dir") or "").strip(),
            base_commit=str(row.get("base_commit") or "").strip(),
            micro_speedup=_finite_float(row.get("micro_speedup")),
            env_flag=str(row.get("env_flag") or "").strip(),
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
