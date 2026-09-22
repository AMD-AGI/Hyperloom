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
    CANONICAL_INSTRUCTION_PATHS,
    KNOWN_INSTRUCTION_PATHS,
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_MEASURED,
    arch_spec,
    equal_rate_paths,
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
        return self.peak_source == PEAK_SOURCE_MEASURED

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


def hardware_schema() -> dict[str, Any]:
    """The JSON shape the analyst reports the roofs it established in."""
    return {
        "type": "object",
        "required": ["peak_source", "peak_flops", "bandwidth", "dispatch_floor_s"],
        "properties": {
            "peak_source": {
                "type": "string",
                "enum": [PEAK_SOURCE_MEASURED, PEAK_SOURCE_DATASHEET],
                "description": (
                    f"'{PEAK_SOURCE_MEASURED}' only if you ran a saturating benchmark on this box "
                    f"this session. '{PEAK_SOURCE_DATASHEET}' if you could not and fell back to "
                    "published peaks. Never claim the first when you did the second."
                ),
            },
            "peak_flops": {
                "type": "object",
                "description": (
                    "instruction path -> FLOP/s (OP/s for integer paths), for the paths you "
                    "established. Omit a path you could not measure rather than guessing it."
                ),
            },
            "bandwidth": {
                "type": "object",
                "description": "memory level -> bytes/s, from: " + ", ".join(BANDWIDTH_TIERS),
            },
            "dispatch_floor_s": {
                "type": "number",
                "description": (
                    "Seconds for one unavoidable kernel dispatch, timed from a captured graph. "
                    "Zero if you could not measure it; the report then says every latency term is "
                    "an assumption."
                ),
            },
            "method": {
                "type": "string",
                "description": (
                    "How each figure was obtained, in enough detail to be repeated: the tool and "
                    "version, the command, the column each roof was read from, and any correction "
                    "you applied."
                ),
            },
        },
    }


def response_schema() -> dict[str, Any]:
    """The JSON shape the analyst must return, also used to ask for a repair."""
    return {
        "type": "object",
        "required": ["hardware", "cases", "confidence", "analysis_md"],
        "properties": {
            "hardware": hardware_schema(),
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


#: How far a reported figure may sit above its published peak and still be read
#: as rounding. Nothing measured beats the vendor peak, so past this the figure
#: was read off wrongly -- a column taken by the wrong name, a unit left
#: unscaled, or a row taken for the wrong device.
_OVER_DATASHEET_TOLERANCE = 1.02

#: How far two instruction paths the vendor rates identically may drift.
_EQUAL_RATE_TOLERANCE = 0.05


def build_hardware(payload: Mapping[str, Any], *, arch: str) -> Hardware:
    """Read the roofs the analyst established, refusing figures that cannot be.

    The analyst measures the machine itself now, which means nothing
    deterministic stands between a mistyped column and the denominator of every
    ceiling. These checks are what is left, and they are deliberately narrow:
    they reject only what the vendor's own published peaks contradict, and say
    nothing about figures the datasheet does not cover.

    A roof that reads low is the dangerous direction -- the ceiling derived from
    it is too loose, the kernel reads as closer to done than it is, and a
    campaign with an attainment target stops with the work half finished. So a
    figure above its published peak is refused outright, and two paths the
    vendor rates as one must not arrive apart.
    """
    if not isinstance(payload, Mapping):
        raise CeilingContractError("response must carry a 'hardware' object describing the roofs used")

    peak_source = str(payload.get("peak_source") or "").strip()
    if peak_source not in {PEAK_SOURCE_MEASURED, PEAK_SOURCE_DATASHEET}:
        raise CeilingContractError(
            f"hardware.peak_source must be {PEAK_SOURCE_MEASURED!r} or {PEAK_SOURCE_DATASHEET!r}, "
            f"not {peak_source!r}"
        )

    peak_flops: dict[str, float] = {}
    for name, raw in (payload.get("peak_flops") or {}).items():
        value = _finite(raw)
        if value is None or value <= 0:
            raise CeilingContractError(f"hardware.peak_flops[{name!r}] is not a positive number")
        if str(name) not in KNOWN_INSTRUCTION_PATHS:
            raise CeilingContractError(
                f"hardware.peak_flops names {name!r}, which is not an instruction path this build "
                "knows; use one of: " + ", ".join(CANONICAL_INSTRUCTION_PATHS)
            )
        peak_flops[str(name)] = value
    if not peak_flops:
        raise CeilingContractError("hardware.peak_flops is empty; no ceiling can be divided by nothing")

    bandwidth: dict[str, float] = {}
    for name, raw in (payload.get("bandwidth") or {}).items():
        value = _finite(raw)
        if value is None or value <= 0:
            raise CeilingContractError(f"hardware.bandwidth[{name!r}] is not a positive number")
        if str(name) not in BANDWIDTH_TIERS:
            raise CeilingContractError(
                f"hardware.bandwidth names memory level {name!r}; use one of: " + ", ".join(BANDWIDTH_TIERS)
            )
        bandwidth[str(name)] = value
    if bandwidth.get("hbm", 0.0) <= 0:
        raise CeilingContractError("hardware.bandwidth needs an 'hbm' figure; every report states that roof")

    dispatch_floor_s = _finite(payload.get("dispatch_floor_s")) or 0.0
    if dispatch_floor_s < 0:
        raise CeilingContractError("hardware.dispatch_floor_s cannot be negative")

    for problem in _roofs_contradicting_the_datasheet(peak_flops, bandwidth, arch):
        raise CeilingContractError(problem)

    provenance: dict[str, Any] = {"reported_by": "ceiling analyst"}
    method = str(payload.get("method") or "").strip()
    if method:
        provenance["method"] = method
    return Hardware(
        arch=arch,
        peak_flops=peak_flops,
        bandwidth=bandwidth,
        peak_source=peak_source,
        dispatch_floor_s=dispatch_floor_s,
        provenance=provenance,
    )


def _roofs_contradicting_the_datasheet(
    peak_flops: Mapping[str, float],
    bandwidth: Mapping[str, float],
    arch: str,
) -> list[str]:
    """Reported roofs the vendor's own published peaks rule out."""
    spec = arch_spec(arch)
    if spec is None:
        return []

    problems: list[str] = []
    for path, value in sorted(peak_flops.items()):
        published = float(spec.peak_flops.get(path) or 0.0)
        if published > 0 and value > published * _OVER_DATASHEET_TOLERANCE:
            problems.append(
                f"hardware.peak_flops[{path!r}] is {value:.6g}, above the published {arch} peak of "
                f"{published:.6g}; no measurement beats the vendor peak, so this was read off wrongly"
            )
    hbm = float(bandwidth.get("hbm") or 0.0)
    if spec.hbm_bw_bytes_per_s > 0 and hbm > spec.hbm_bw_bytes_per_s * _OVER_DATASHEET_TOLERANCE:
        problems.append(
            f"hardware.bandwidth['hbm'] is {hbm:.6g}, above the published {arch} peak of "
            f"{spec.hbm_bw_bytes_per_s:.6g}; no measurement beats the vendor peak"
        )
    for left, right in equal_rate_paths(arch):
        a, b = float(peak_flops.get(left) or 0.0), float(peak_flops.get(right) or 0.0)
        if a <= 0 or b <= 0:
            continue
        if abs(a / b - 1.0) > _EQUAL_RATE_TOLERANCE:
            problems.append(
                f"hardware.peak_flops has {left} at {a:.6g} and {right} at {b:.6g}, but {arch} runs "
                "both at one rate; the profiler halves one of them and the correction was not applied"
            )
    return problems


def build_report(
    payload: Mapping[str, Any],
    *,
    canonical_id: str,
    arch: str,
    expected_case_ids: Sequence[str],
    observed_ms: Mapping[str, float] | None = None,
) -> CeilingReport:
    """Validate one analyst response and publish it as a ceiling report.

    The analyst establishes the roofs as well as the work model, so the
    ``hardware`` block it returns is validated here rather than supplied here.

    Raises :class:`CeilingContractError` for anything that makes the response
    unusable -- a roof above the vendor's own peak, a missing case, a latency
    that is not a number, an empty derivation. Findings that leave the answer
    usable but suspect ride along on the case's ``issues``, because a ceiling
    whose doubts are invisible is worse than one that names them.
    """
    hardware = build_hardware(payload.get("hardware") or {}, arch=arch)

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
            "Peaks are vendor datasheet figures, measured on no card: these latencies are an absolute "
            "lower bound no implementation reaches. The gap to a real card is not a fixed discount -- "
            "on gfx950 it runs from 1.2% for FP32 matrix to 50.8% for FP16 matrix -- so cases of "
            "different dtypes are not comparable and no correction makes them so."
        )
    missing_paths = sorted(set(CANONICAL_INSTRUCTION_PATHS) - set(hardware.peak_flops))
    if missing_paths:
        caveats.append(
            "No roof was established for: "
            + ", ".join(missing_paths)
            + ". A case whose arithmetic runs on one of those was priced against a substitute."
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
    "build_hardware",
    "build_report",
    "hardware_schema",
    "load_report",
    "response_schema",
]
