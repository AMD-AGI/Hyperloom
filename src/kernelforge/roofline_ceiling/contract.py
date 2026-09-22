# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ceiling output contract: a number per shape, and the prose that defends it.

The analyst owns the whole estimate, composition included. An earlier revision
had it declare a stage-by-stage work model and let this module compose the
latency from it, on one fixed rule::

    t_stage = dispatch_count * dispatch_floor + max(flops / peak, bytes / bw)
    t_ideal = sum over stages

That rule does not generalize. It cannot express a stage that partially overlaps
its predecessor, a stage whose arithmetic is split across MFMA and SFU paths, a
shape whose real limit is that it fills eight CUs out of two hundred and
fifty-six, or a working set that lives in Infinity Cache rather than HBM. Each
gap wanted another schema field, and a model that needs an escape hatch per
operator family is the wrong model. Validating a candidate against it produced
the self-consistency of a formula that was already wrong -- confidence, not
safety.

So the composition is the analyst's judgement too, and the audit trail is prose:
``analysis_md``, written to the structure the methodology prescribes, carrying
the formulas, the hardware figures used and the assumptions behind them. A human
can check it. This module cannot, and no longer pretends to.

What is still enforced here needs no view of the model at all: the case set is
the driver's, every scored case is answered and no other, a latency is finite
and positive, and a ceiling above the latency the box was observed reaching is
reported rather than shipped. Those hold whatever composition the analyst chose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from kernelforge.roofline_ceiling.specs import (
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_REFERENCE,
)

#: Bumped from 1 when the stage-level work model was dropped. A cached v1 report
#: describes a differently-derived number and is refused rather than read.
SCHEMA_VERSION = 2

BOUND_COMPUTE = "compute"
BOUND_MEMORY = "memory"
BOUND_LATENCY = "latency"
BOUND_MIXED = "mixed"
BOUNDS = (BOUND_COMPUTE, BOUND_MEMORY, BOUND_LATENCY, BOUND_MIXED)

CONFIDENCE_LEVELS = ("high", "medium", "low")

#: Memory levels a ceiling may be taken against. Handed to the analyst with
#: whatever figures were measured for them, because a kernel whose working set
#: is Infinity-Cache resident rides a higher roof than HBM and one bounded by
#: LDS bank throughput rides a lower one.
BANDWIDTH_TIERS = ("hbm", "mall", "l2", "l1", "lds")


class CeilingContractError(ValueError):
    """Raised when an analyst response cannot be read as a ceiling."""


@dataclass(frozen=True)
class Hardware:
    """The measured figures one ceiling run was handed."""

    arch: str
    #: instruction path -> peak FLOP/s (or OP/s for the integer paths).
    peak_flops: dict[str, float]
    #: memory level -> bytes/s.
    bandwidth: dict[str, float]
    peak_source: str
    dispatch_floor_s: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_measured(self) -> bool:
        """Whether these came from a real card rather than off a datasheet."""
        return self.peak_source == PEAK_SOURCE_REFERENCE

    @property
    def hbm_bw_bytes_per_s(self) -> float:
        """The HBM roof, the one figure every report states."""
        return float(self.bandwidth.get("hbm", 0.0))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the published report."""
        return asdict(self)


@dataclass(frozen=True)
class CaseCeiling:
    """The deliverable for one test shape: its theoretical achievable latency."""

    case_id: str
    t_ideal_ms: float
    bound: str
    profiler_observed_ms: float | None = None
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the published report."""
        payload = asdict(self)
        payload["issues"] = list(self.issues)
        return payload


@dataclass(frozen=True)
class CeilingReport:
    """Per-shape theoretical achievable latency for one kernel on one box."""

    schema_version: int
    canonical_id: str
    hardware: Hardware
    cases: tuple[CaseCeiling, ...]
    confidence: str
    #: The analyst's own derivation, in the structure the methodology prescribes.
    #: The only audit trail there is, which is why it is required rather than
    #: optional: a latency with nothing behind it cannot be argued with, only
    #: believed or discarded.
    analysis_md: str = ""
    caveats: tuple[str, ...] = ()

    def case(self, case_id: str) -> CaseCeiling | None:
        """Look one case up by id."""
        for entry in self.cases:
            if entry.case_id == case_id:
                return entry
        return None

    def ideal_ms(self) -> dict[str, float]:
        """The headline answer: ``case_id -> theoretical achievable latency``."""
        return {entry.case_id: entry.t_ideal_ms for entry in self.cases}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the published report."""
        return {
            "schema_version": self.schema_version,
            "canonical_id": self.canonical_id,
            "hardware": self.hardware.to_dict(),
            "confidence": self.confidence,
            "analysis_md": self.analysis_md,
            "caveats": list(self.caveats),
            "cases": [entry.to_dict() for entry in self.cases],
        }


def response_schema() -> dict[str, Any]:
    """The JSON shape the analyst must return, also used to ask for a repair."""
    return {
        "type": "object",
        "required": ["cases", "confidence", "analysis_md"],
        "properties": {
            "cases": {
                "type": "array",
                "description": "One entry per scored case id, no more and no fewer.",
                "items": {
                    "type": "object",
                    "required": ["case_id", "t_ideal_ms", "bound"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "t_ideal_ms": {
                            "type": "number",
                            "description": (
                                "Theoretical achievable latency for this shape, in milliseconds. "
                                "Your own composition of the terms you judged relevant, against "
                                "the hardware figures supplied in this request."
                            ),
                        },
                        "bound": {
                            "type": "string",
                            "enum": list(BOUNDS),
                            "description": "What limits this shape at its ceiling.",
                        },
                    },
                },
            },
            "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
            "analysis_md": {
                "type": "string",
                "description": (
                    "The full derivation as Markdown, in the structure the role document "
                    "prescribes. This is the only record of how each latency was reached, so "
                    "it must carry the formulas, the hardware figures used, the per-case "
                    "arithmetic and the assumptions -- enough for a reader to recompute every "
                    "number without rerunning you."
                ),
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
    }


def _finite(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` when the value is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_report(
    payload: Mapping[str, Any],
    *,
    canonical_id: str,
    hardware: Hardware,
    expected_case_ids: Sequence[str],
    observed_ms: Mapping[str, float] | None = None,
) -> CeilingReport:
    """Validate one analyst response and publish it as a ceiling report.

    Raises :class:`CeilingContractError` for anything that makes the response
    unusable -- a missing case, a latency that is not a number, an empty
    derivation. Findings that leave the answer usable but suspect ride along on
    the case's ``issues``, because a ceiling whose doubts are invisible is worse
    than one that names them.
    """
    if hardware.peak_source not in {PEAK_SOURCE_REFERENCE, PEAK_SOURCE_DATASHEET}:
        raise CeilingContractError(f"unknown peak_source {hardware.peak_source!r}")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, str) or not raw_cases:
        raise CeilingContractError("response must carry a non-empty 'cases' array")

    expected = list(dict.fromkeys(str(case_id) for case_id in expected_case_ids))
    if not expected:
        raise CeilingContractError("no scored case ids to produce a ceiling for")

    analysis_md = str(payload.get("analysis_md") or "").strip()
    if not analysis_md:
        raise CeilingContractError(
            "response must carry 'analysis_md'; it is the only record of how the latencies were reached"
        )

    observed = dict(observed_ms or {})
    by_id: dict[str, CaseCeiling] = {}

    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise CeilingContractError("every entry of 'cases' must be an object")
        case_id = str(raw_case.get("case_id") or "").strip()
        if not case_id:
            raise CeilingContractError("a case entry has no 'case_id'")
        if case_id in by_id:
            raise CeilingContractError(f"case {case_id!r} appears twice")

        t_ideal_ms = _finite(raw_case.get("t_ideal_ms"))
        if t_ideal_ms is None or t_ideal_ms <= 0:
            raise CeilingContractError(f"case {case_id!r} needs a finite positive 't_ideal_ms'")

        bound = str(raw_case.get("bound") or "").strip().lower()
        if bound not in BOUNDS:
            raise CeilingContractError(f"case {case_id!r} needs a 'bound' from: {', '.join(BOUNDS)}")

        issues: list[str] = []
        # The one check that survives dropping the work model, and the one the
        # methodology leans on hardest: a ceiling above what the box was seen
        # doing is not a ceiling. Report it rather than clamping -- the estimate
        # is wrong somewhere, and clamping hides which part.
        seen = observed.get(case_id)
        if seen is not None and t_ideal_ms > float(seen):
            issues.append(
                f"ideal latency {t_ideal_ms:.6g} ms exceeds the observed {float(seen):.6g} ms; "
                "the estimate overstates the minimum legal work"
            )
        # A shape the derivation never mentions has a number and nothing behind
        # it. Not fatal -- the latency may still be right -- but it is exactly
        # the case nobody can check.
        if case_id not in analysis_md:
            issues.append(f"case {case_id!r} is never mentioned in the derivation, so its latency is unaudited")

        by_id[case_id] = CaseCeiling(
            case_id=case_id,
            t_ideal_ms=t_ideal_ms,
            bound=bound,
            profiler_observed_ms=(float(seen) if seen is not None else None),
            issues=tuple(issues),
        )

    missing = [case_id for case_id in expected if case_id not in by_id]
    if missing:
        raise CeilingContractError("no ceiling for scored case(s): " + ", ".join(missing))
    extra = sorted(set(by_id) - set(expected))
    if extra:
        raise CeilingContractError("ceiling for case(s) the driver never scored: " + ", ".join(extra))

    confidence = str(payload.get("confidence") or "").strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        raise CeilingContractError(f"'confidence' must be one of {', '.join(CONFIDENCE_LEVELS)}")

    caveats = [str(entry).strip() for entry in (payload.get("caveats") or ()) if str(entry).strip()]
    if hardware.peak_source == PEAK_SOURCE_DATASHEET:
        caveats.append(
            "Peaks are vendor datasheet figures, not measured on any card: these latencies are an "
            "absolute lower bound that no implementation reaches, not an achievable target."
        )
    elif hardware.peak_source == PEAK_SOURCE_REFERENCE:
        caveats.append(
            "Peaks come from a committed profile measured on a card of this configuration, not from "
            "this box on this day. Clocks, power cap and cooling move them by a few percent, so read "
            "these latencies as close rather than exact."
        )
    if hardware.dispatch_floor_s <= 0:
        caveats.append(
            "Dispatch floor was not measured, so any launch-latency term in these estimates is the "
            "analyst's assumption rather than this box's."
        )

    return CeilingReport(
        schema_version=SCHEMA_VERSION,
        canonical_id=canonical_id,
        hardware=hardware,
        cases=tuple(by_id[case_id] for case_id in expected),
        confidence=confidence,
        analysis_md=analysis_md,
        caveats=tuple(dict.fromkeys(caveats)),
    )


def load_report(payload: Mapping[str, Any]) -> CeilingReport:
    """Rebuild a report from its published JSON, for a consumer reading the cache."""
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CeilingContractError(f"unsupported ceiling schema {version!r}; this build reads {SCHEMA_VERSION}")

    raw_hardware = payload.get("hardware")
    if not isinstance(raw_hardware, Mapping):
        raise CeilingContractError("published report has no 'hardware' record")
    hardware = Hardware(
        arch=str(raw_hardware.get("arch") or ""),
        peak_flops={str(key): float(value) for key, value in (raw_hardware.get("peak_flops") or {}).items()},
        bandwidth={str(key): float(value) for key, value in (raw_hardware.get("bandwidth") or {}).items()},
        peak_source=str(raw_hardware.get("peak_source") or ""),
        dispatch_floor_s=float(raw_hardware.get("dispatch_floor_s") or 0.0),
        provenance=dict(raw_hardware.get("provenance") or {}),
    )

    cases: list[CaseCeiling] = []
    for raw_case in payload.get("cases") or ():
        if not isinstance(raw_case, Mapping):
            raise CeilingContractError("published report has a malformed case entry")
        observed = raw_case.get("profiler_observed_ms")
        cases.append(
            CaseCeiling(
                case_id=str(raw_case.get("case_id") or ""),
                t_ideal_ms=float(raw_case.get("t_ideal_ms") or 0.0),
                bound=str(raw_case.get("bound") or BOUND_MIXED),
                profiler_observed_ms=(float(observed) if observed is not None else None),
                issues=tuple(str(entry) for entry in (raw_case.get("issues") or ())),
            )
        )

    return CeilingReport(
        schema_version=SCHEMA_VERSION,
        canonical_id=str(payload.get("canonical_id") or ""),
        hardware=hardware,
        cases=tuple(cases),
        confidence=str(payload.get("confidence") or ""),
        analysis_md=str(payload.get("analysis_md") or ""),
        caveats=tuple(str(entry) for entry in (payload.get("caveats") or ())),
    )


__all__ = [
    "BANDWIDTH_TIERS",
    "BOUNDS",
    "BOUND_COMPUTE",
    "BOUND_LATENCY",
    "BOUND_MEMORY",
    "BOUND_MIXED",
    "CONFIDENCE_LEVELS",
    "SCHEMA_VERSION",
    "CaseCeiling",
    "CeilingContractError",
    "CeilingReport",
    "Hardware",
    "build_report",
    "load_report",
    "response_schema",
]
