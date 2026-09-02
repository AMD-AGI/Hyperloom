# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Typed measurement claim: the single object that carries a benchmark reading to the promotion gate.

A claim wraps the three facts a gate decision requires — the measured value,
how it was obtained, and where the on-disk artifact lives — into one frozen
object. Writeback reads claims rather than plucking individual keys from a
result dict, so every key that enters the gate is visible from a single
call site and zero-reader fields cannot accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_BASES: frozenset[str] = frozenset(
    {
        "e2e_rebench",
        "e2e_rebench_unpaired",
        "e2e_paired",
        "e2e_decision_round",
        "e2e_decision_round_unpaired",
        "geak_same_harness_geak",
    }
)


@dataclass(frozen=True)
class MeasurementClaim:
    """One benchmark reading offered for promotion.

    Attributes:
        value: The measured throughput (tok/s or img/s, whole-server total).
            Must be strictly positive for a claim to be promotable.
        basis: How the reading was obtained.  Must belong to the closed
            vocabulary in ``_VALID_BASES``.
        artifact_ref: Path to the on-disk ``benchmark_report.json`` (or
            equivalent) that corroborates the number. ``None`` when no
            artifact exists — callers must decide whether to gate on this.
    """

    value: float
    basis: str
    artifact_ref: str | None = None

    def is_promotable(self) -> bool:
        """Return True when value is positive and basis is in the valid set."""
        return isinstance(self.value, (int, float)) and self.value > 0 and self.basis in _VALID_BASES


def claim_from_result(result: dict) -> MeasurementClaim | None:
    """Extract a MeasurementClaim from an executor result dict.

    Understands both ``output_throughput`` (explore / integrate_patch path)
    and ``new_tput`` (kernel integrate path) so callers need not branch on
    the producer.

    Args:
        result: The executor result mapping.

    Returns:
        A :class:`MeasurementClaim` when a positive throughput is present,
        else ``None``.
    """
    raw = result.get("output_throughput") or result.get("new_tput")
    if not isinstance(raw, (int, float)) or raw <= 0:
        return None
    basis = str(result.get("measurement_basis") or "e2e_rebench")
    if basis not in _VALID_BASES:
        basis = "e2e_rebench"
    artifact_ref = result.get("raw_result_path") or result.get("benchmark_report_path")
    return MeasurementClaim(
        value=float(raw),
        basis=basis,
        artifact_ref=str(artifact_ref) if artifact_ref else None,
    )
