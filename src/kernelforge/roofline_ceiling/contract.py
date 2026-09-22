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
    #: Findings that leave the latency usable but suspect. Surfaced to the
    #: operator when the report is built and written into the derivation
    #: document; deliberately absent from the published file, which carries the
    #: answer and nothing to weigh it against.
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class CeilingReport:
    """Per-shape theoretical achievable latency for one kernel on one box.

    The published file is the latencies and their mean. Everything else here --
    the derivation, the findings, where the roofs came from -- exists for the
    document beside it and for the operator watching the run, and is not
    serialized. A consumer needs the numbers; a reader deciding whether to
    believe them needs the document.
    """

    canonical_id: str
    #: ``measured_on_this_box`` or ``datasheet``, for the document. Not
    #: published: a consumer gets the latencies, and a reader deciding how much
    #: to trust them reads the derivation instead.
    peak_source: str
    cases: tuple[CaseCeiling, ...]
    #: The analyst's own derivation. The only audit trail there is: nothing
    #: recomputes these latencies, so a reader who doubts one has this and
    #: nothing else.
    analysis_md: str = ""
    #: Roofs the analyst established, kept for the document. Never published.
    hardware: Hardware | None = None

    def case(self, case_id: str) -> CaseCeiling | None:
        """Look one case up by id."""
        for entry in self.cases:
            if entry.case_id == case_id:
                return entry
        return None

    def ideal_ms(self) -> dict[str, float]:
        """The headline answer: ``case_id -> theoretical achievable latency``."""
        return {entry.case_id: entry.t_ideal_ms for entry in self.cases}

    def mean_ideal_ms(self) -> float | None:
        """The equal-weight mean across cases, or ``None`` when there are none.

        Equal weight because that is how the campaign scores the suite: a mean
        that weighted by latency would let one large shape speak for all of
        them, and then the aggregate here would describe something the
        objective does not.
        """
        if not self.cases:
            return None
        return sum(entry.t_ideal_ms for entry in self.cases) / len(self.cases)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the published report: the latencies and their mean, nothing else."""
        return {
            "mean_ideal_ms": self.mean_ideal_ms(),
            "cases": self.ideal_ms(),
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
        "required": ["hardware", "cases", "analysis_md"],
        "properties": {
            "hardware": hardware_schema(),
            "cases": {
                "type": "array",
                "description": "One entry per scored case id, no more and no fewer.",
                "items": {
                    "type": "object",
                    "required": ["case_id", "t_ideal_ms"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "t_ideal_ms": {
                            "type": "number",
                            "description": (
                                "Theoretical achievable latency for this shape, in milliseconds. "
                                "Your own composition of the terms you judged relevant, against "
                                "the roofs you established in 'hardware'."
                            ),
                        },
                    },
                },
            },
            "analysis_md": {
                "type": "string",
                "description": (
                    "The full derivation as Markdown, in the structure the role document "
                    "prescribes. This is the only record of how each latency was reached, so "
                    "it must carry the formulas, the roofs used, the per-case arithmetic, what "
                    "bounds each shape and every assumption -- enough for a reader to recompute "
                    "every number without rerunning you."
                ),
            },
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

        issues: list[str] = []
        # The check the methodology leans on hardest: a ceiling above what the
        # box was seen doing is not a ceiling. Reported, never clamped -- the
        # estimate is wrong somewhere and clamping hides which part.
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

        by_id[case_id] = CaseCeiling(case_id=case_id, t_ideal_ms=t_ideal_ms, issues=tuple(issues))

    missing = [case_id for case_id in expected if case_id not in by_id]
    if missing:
        raise CeilingContractError("no ceiling for scored case(s): " + ", ".join(missing))
    extra = sorted(set(by_id) - set(expected))
    if extra:
        raise CeilingContractError("ceiling for case(s) the driver never scored: " + ", ".join(extra))

    return CeilingReport(
        canonical_id=canonical_id,
        peak_source=hardware.peak_source,
        cases=tuple(by_id[case_id] for case_id in expected),
        analysis_md=analysis_md,
        hardware=hardware,
    )


def load_report(payload: Mapping[str, Any]) -> CeilingReport:
    """Rebuild a report from its published file.

    The file carries the latencies and their mean. Everything that would let a
    reader weigh them -- the derivation, the roofs, where those roofs came from
    -- is in the document published beside it, so a report loaded from disk can
    answer what the ceiling is and not how much to trust it.
    """
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Mapping):
        raise CeilingContractError("published report has no 'cases' mapping")

    cases: list[CaseCeiling] = []
    for case_id, raw in raw_cases.items():
        latency = _finite(raw)
        if latency is None or latency <= 0:
            raise CeilingContractError(f"published report has no usable latency for case {case_id!r}")
        cases.append(CaseCeiling(case_id=str(case_id), t_ideal_ms=latency))

    return CeilingReport(canonical_id="", peak_source="", cases=tuple(cases))


__all__ = [
    "BANDWIDTH_TIERS",
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
