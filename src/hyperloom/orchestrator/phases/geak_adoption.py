# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Construct canonical GEAK adoption ledger rows."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdoptionMeasurement:
    """One rebench outcome shared by every kernel in the adopted stack."""

    measured_tput: float
    rebench_gain: float | None
    attributable: bool
    provenance: str
    ts: str
    cycle: int
    overlay_loaded: bool | None
    basis: str
    alignment_status: str


def adoption_entry(
    previous: dict[str, Any],
    *,
    kid: str,
    spec: dict[str, Any],
    measurement: AdoptionMeasurement,
) -> dict[str, Any]:
    """Preserve the best measured gain while recording this adoption's attribution."""
    entry = dict(previous or {})
    attempts = list(entry.get("attempts") or [])
    attempt_decision = "KEEP" if measurement.attributable else "UNATTRIBUTED"
    attempt_status = "ok" if measurement.attributable else "unvalidated"
    attempts.append(
        {
            "decision": attempt_decision,
            "status": attempt_status,
            "new_tput": measurement.measured_tput,
            "gain_pct": measurement.rebench_gain if measurement.attributable else None,
            "decision_reason": measurement.provenance,
            "artifact_kind": str(spec.get("kind") or "authored"),
            "ts": measurement.ts,
            "cycle": measurement.cycle,
        }
    )
    # Max over attempts, matching the canonical ledger writer in ``kernel_decisions.py``: a second, worse rebench
    # must not lower the kernel's best.
    gains = [
        float(a["gain_pct"]) for a in attempts if isinstance(a, dict) and isinstance(a.get("gain_pct"), (int, float))
    ]
    entry.update(
        {
            "key": kid,
            "kernel_id": kid,
            "source": "geak_e2e",
            "attempts": attempts,
            "attempt_count": len(attempts),
            "best_gain_pct": max(gains) if gains else None,
            "last_decision": attempt_decision,
            "last_status": attempt_status,
            "validated": measurement.attributable,
            "overlay_loaded": bool(measurement.overlay_loaded),
            "basis": measurement.basis,
            "alignment_status": measurement.alignment_status,
            "updated_at": measurement.ts,
        }
    )
    return entry
