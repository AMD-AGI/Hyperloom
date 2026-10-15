# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Construct canonical GEAK adoption ledger rows."""

from typing import Any


def adoption_entry(
    previous: dict[str, Any],
    *,
    kid: str,
    spec: dict[str, Any],
    measured_tput: float,
    rebench_gain: float | None,
    attributable: bool,
    provenance: str,
    ts: str,
    cycle: int,
    overlay_loaded: bool | None,
    basis: str,
    alignment_status: str,
) -> dict[str, Any]:
    """Preserve the best measured gain while recording this adoption's attribution."""
    entry = dict(previous or {})
    attempts = list(entry.get("attempts") or [])
    attempt_decision = "KEEP" if attributable else "UNATTRIBUTED"
    attempt_status = "ok" if attributable else "unvalidated"
    attempts.append(
        {
            "decision": attempt_decision,
            "status": attempt_status,
            "new_tput": measured_tput,
            "gain_pct": rebench_gain if attributable else None,
            "decision_reason": provenance,
            "artifact_kind": str(spec.get("kind") or "authored"),
            "ts": ts,
            "cycle": cycle,
        }
    )
    # Max over attempts, matching the canonical ledger writer in ``_kernel_decisions.py`` -- ``by_kernel`` and
    # ``kernel_lifecycle`` read this one field from both writers, so a second, worse rebench must not lower
    # the kernel's best.
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
            "validated": attributable,
            "overlay_loaded": bool(overlay_loaded),
            "basis": basis,
            "alignment_status": alignment_status,
            # GEAK's own same-config A/B, kept beside the orchestrator number so the two are never confused
            # for each other.
            "geak_same_config_delta_pct": spec.get("e2e_delta_pct"),
            "geak_isolated_speedup": spec.get("isolated"),
            "updated_at": ts,
        }
    )
    return entry
