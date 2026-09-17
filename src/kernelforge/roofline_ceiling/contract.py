# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ceiling output contract: what the analyst declares, and what is derived.

The split this module encodes is the whole design. The analyst owns every
*judgement*: how many serial stages a case really has, the minimum legal FLOPs
and semantic bytes of each, which instruction path it runs on, how many
dispatches cannot be overlapped, and which assumptions all of that rests on.
Nothing there is enumerable, which is why it is an agent's job and not a table's.

The analyst owns no *arithmetic* and supplies no hardware constant. Peaks come
from the resolved :class:`Hardware` record -- measured on this box when
``--roof-only`` ran, datasheet otherwise -- and the ideal latency is derived
here:

    t_stage  = dispatch_count * dispatch_floor + extra_latency
               + max(flops / peak, bytes / bandwidth)
    t_ideal  = sum over stages

Serial stages are summed rather than merged before one ``max`` on purpose: a
single ``max`` over pooled FLOPs and bytes asserts that the compute of one stage
overlaps the memory traffic of another, which is exactly what "serial" denies.

Deriving the number instead of accepting it also means a mismatch between the
model and the answer cannot exist: there is one answer, and it is reproducible
from the declared model by anyone who disagrees with it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from kernelforge.roofline_ceiling.specs import (
    KNOWN_INSTRUCTION_PATHS,
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_EMPIRICAL,
)

SCHEMA_VERSION = 1

BOUND_COMPUTE = "compute"
BOUND_MEMORY = "memory"
BOUND_LATENCY = "latency"
BOUND_MIXED = "mixed"

CONFIDENCE_LEVELS = ("high", "medium", "low")

#: A term is called dominant when it supplies more of a case's ideal latency
#: than every other term combined. Below that the case is genuinely contested
#: and "mixed" is the honest label -- guide section 8.4.
_DOMINANCE_SHARE = 0.5


class CeilingContractError(ValueError):
    """Raised when an analyst response cannot be read as a ceiling model."""


@dataclass(frozen=True)
class Hardware:
    """The hardware constants one ceiling run was computed against."""

    arch: str
    hbm_bw_bytes_per_s: float
    peak_flops: dict[str, float]
    peak_source: str
    dispatch_floor_s: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empirical(self) -> bool:
        """Whether the peaks were measured on this box rather than read off a datasheet."""
        return self.peak_source == PEAK_SOURCE_EMPIRICAL

    def peak_for(self, instruction_path: str) -> float:
        """Peak FLOP/s for one instruction path, ``0.0`` when it has none."""
        return float(self.peak_flops.get(str(instruction_path or "").strip().lower(), 0.0))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the published report."""
        return asdict(self)


@dataclass(frozen=True)
class Stage:
    """One serial stage of a case, as modelled by the analyst."""

    name: str
    flops: float
    bytes_moved: float
    instruction_path: str
    dispatch_count: int
    formula_flops: str
    formula_bytes: str
    extra_latency_s: float = 0.0
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the published report.

        The traffic field is published as ``bytes``, the name the response schema
        uses, so a published report reads back through the same parser that read
        the analyst's answer. Publishing the Python attribute name instead left
        the module unable to load its own output.
        """
        payload = asdict(self)
        payload["bytes"] = payload.pop("bytes_moved")
        payload["assumptions"] = list(self.assumptions)
        return payload


@dataclass(frozen=True)
class StageTiming:
    """The derived service times of one stage."""

    name: str
    t_compute_s: float
    t_memory_s: float
    t_latency_s: float
    t_stage_s: float
    peak_flops_used: float
    bw_used: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the published report."""
        return asdict(self)


@dataclass(frozen=True)
class CaseCeiling:
    """The deliverable for one test shape: its theoretical achievable latency."""

    case_id: str
    t_ideal_ms: float
    bound: str
    stages: tuple[Stage, ...]
    timings: tuple[StageTiming, ...]
    bound_note: str = ""
    profiler_observed_ms: float | None = None
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the published report."""
        return {
            "case_id": self.case_id,
            "t_ideal_ms": self.t_ideal_ms,
            "bound": self.bound,
            "bound_note": self.bound_note,
            "profiler_observed_ms": self.profiler_observed_ms,
            "issues": list(self.issues),
            "derivation": {
                "stages": [stage.to_dict() for stage in self.stages],
                "timings": [timing.to_dict() for timing in self.timings],
            },
        }


@dataclass(frozen=True)
class CeilingReport:
    """Per-shape theoretical achievable latency for one kernel on one box."""

    schema_version: int
    canonical_id: str
    hardware: Hardware
    cases: tuple[CaseCeiling, ...]
    confidence: str
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
            "caveats": list(self.caveats),
            "cases": [entry.to_dict() for entry in self.cases],
        }


def response_schema() -> dict[str, Any]:
    """The JSON shape the analyst must return, also used to ask for a repair."""
    return {
        "type": "object",
        "required": ["cases", "confidence"],
        "properties": {
            "cases": {
                "type": "array",
                "description": "One entry per scored case id, no more and no fewer.",
                "items": {
                    "type": "object",
                    "required": ["case_id", "stages"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "bound_note": {
                            "type": "string",
                            "description": "Why this case is bound the way it is, per stage.",
                        },
                        "stages": {
                            "type": "array",
                            "description": (
                                "The serial stages a best legal implementation cannot avoid, in "
                                "execution order. One stage when the whole case can be a single "
                                "fused dispatch."
                            ),
                            "items": {
                                "type": "object",
                                "required": [
                                    "name",
                                    "flops",
                                    "bytes",
                                    "instruction_path",
                                    "dispatch_count",
                                    "formula_flops",
                                    "formula_bytes",
                                ],
                                "properties": {
                                    "name": {"type": "string"},
                                    "flops": {
                                        "type": "number",
                                        "description": (
                                            "Minimum legal FLOPs by algorithm semantics, not the padded "
                                            "work the current implementation happens to execute."
                                        ),
                                    },
                                    "bytes": {
                                        "type": "number",
                                        "description": (
                                            "Minimum semantic bytes a best legal implementation must move "
                                            "through HBM. Excludes intermediates a legal fusion keeps in "
                                            "registers, LDS or cache."
                                        ),
                                    },
                                    "instruction_path": {
                                        "type": "string",
                                        "description": (
                                            "The hardware path this stage's arithmetic actually runs on. "
                                            "Use the dtype the MFMA sees, not the storage dtype: weights "
                                            "unpacked to bf16 before the MFMA run at the bf16 rate."
                                        ),
                                    },
                                    "dispatch_count": {
                                        "type": "integer",
                                        "description": (
                                            "Serial kernel dispatches this stage cannot avoid. Each is "
                                            "charged the measured dispatch floor."
                                        ),
                                    },
                                    "extra_latency_s": {
                                        "type": "number",
                                        "description": (
                                            "Unavoidable non-dispatch serialization (barriers, recurrence). "
                                            "Justify any nonzero value in assumptions."
                                        ),
                                    },
                                    "formula_flops": {"type": "string"},
                                    "formula_bytes": {"type": "string"},
                                    "assumptions": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
            },
            "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
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


def _parse_stage(raw: Any, *, case_id: str, index: int) -> Stage:
    """Read one stage out of the analyst payload."""
    where = f"case {case_id!r} stage {index}"
    if not isinstance(raw, Mapping):
        raise CeilingContractError(f"{where} is not an object")

    name = str(raw.get("name") or "").strip()
    if not name:
        raise CeilingContractError(f"{where} has no name")

    flops = _finite(raw.get("flops"))
    if flops is None or flops < 0:
        raise CeilingContractError(f"{where} needs a finite non-negative 'flops'")
    bytes_moved = _finite(raw.get("bytes"))
    if bytes_moved is None or bytes_moved < 0:
        raise CeilingContractError(f"{where} needs a finite non-negative 'bytes'")
    if flops <= 0 and bytes_moved <= 0:
        raise CeilingContractError(f"{where} declares neither work nor traffic")

    dispatch_count = raw.get("dispatch_count")
    try:
        dispatches = int(dispatch_count)
    except (TypeError, ValueError) as exc:
        raise CeilingContractError(f"{where} needs an integer 'dispatch_count'") from exc
    if dispatches < 1:
        raise CeilingContractError(f"{where} needs at least one dispatch")

    extra_latency = _finite(raw.get("extra_latency_s", 0.0))
    if extra_latency is None or extra_latency < 0:
        raise CeilingContractError(f"{where} needs a finite non-negative 'extra_latency_s'")

    assumptions = raw.get("assumptions") or ()
    if isinstance(assumptions, str) or not isinstance(assumptions, Sequence):
        raise CeilingContractError(f"{where} 'assumptions' must be a list of strings")

    return Stage(
        name=name,
        flops=flops,
        bytes_moved=bytes_moved,
        instruction_path=str(raw.get("instruction_path") or "").strip().lower(),
        dispatch_count=dispatches,
        formula_flops=str(raw.get("formula_flops") or "").strip(),
        formula_bytes=str(raw.get("formula_bytes") or "").strip(),
        extra_latency_s=extra_latency,
        assumptions=tuple(str(entry).strip() for entry in assumptions if str(entry).strip()),
    )


def _time_stage(stage: Stage, hardware: Hardware) -> tuple[StageTiming, list[str]]:
    """Derive one stage's service times, reporting what it could not be given."""
    issues: list[str] = []

    peak = hardware.peak_for(stage.instruction_path)
    if stage.flops > 0 and peak <= 0:
        # Silently pricing the arithmetic at zero would turn an unmodellable
        # compute term into a free one and report a ceiling below anything the
        # hardware can do. Charge nothing and say so instead -- and separate a
        # path name that does not exist from a real one this box has no peak
        # for, because the first is the analyst's mistake and the second is the
        # measurement's gap.
        known = stage.instruction_path in KNOWN_INSTRUCTION_PATHS
        reason = (
            f"{hardware.arch} has no peak for it under {hardware.peak_source}"
            if known
            else "it is not a canonical instruction path name"
        )
        issues.append(
            f"stage {stage.name!r}: instruction path {stage.instruction_path!r} priced at nothing "
            f"because {reason}; its compute term is missing from the ceiling"
        )
    t_compute = stage.flops / peak if (stage.flops > 0 and peak > 0) else 0.0

    bandwidth = float(hardware.hbm_bw_bytes_per_s)
    if stage.bytes_moved > 0 and bandwidth <= 0:
        issues.append(f"stage {stage.name!r}: no bandwidth for {hardware.arch}; its memory term is missing")
    t_memory = stage.bytes_moved / bandwidth if (stage.bytes_moved > 0 and bandwidth > 0) else 0.0

    t_latency = stage.dispatch_count * max(float(hardware.dispatch_floor_s), 0.0) + stage.extra_latency_s

    return (
        StageTiming(
            name=stage.name,
            t_compute_s=t_compute,
            t_memory_s=t_memory,
            t_latency_s=t_latency,
            t_stage_s=t_latency + max(t_compute, t_memory),
            peak_flops_used=peak,
            bw_used=bandwidth,
        ),
        issues,
    )


def classify_bound(timings: Sequence[StageTiming]) -> str:
    """Name the term that dominates a case's ideal latency."""
    compute = sum(timing.t_compute_s for timing in timings)
    memory = sum(timing.t_memory_s for timing in timings)
    latency = sum(timing.t_latency_s for timing in timings)
    # Compute and memory do not add: within a stage only the larger one is paid.
    served = sum(max(timing.t_compute_s, timing.t_memory_s) for timing in timings)
    total = served + latency
    if total <= 0:
        return BOUND_MIXED
    if latency / total > _DOMINANCE_SHARE:
        return BOUND_LATENCY
    if served <= 0:
        return BOUND_MIXED
    if compute > memory and compute / (compute + memory) > _DOMINANCE_SHARE:
        return BOUND_COMPUTE
    if memory > compute and memory / (compute + memory) > _DOMINANCE_SHARE:
        return BOUND_MEMORY
    return BOUND_MIXED


def build_report(
    payload: Mapping[str, Any],
    *,
    canonical_id: str,
    hardware: Hardware,
    expected_case_ids: Sequence[str],
    observed_ms: Mapping[str, float] | None = None,
) -> CeilingReport:
    """Turn one analyst response into a validated, derived ceiling report.

    Raises :class:`CeilingContractError` for anything that makes the response
    unreadable -- a missing case, a stage without a work model. Findings that
    leave the number computable but suspect ride along on the case's ``issues``
    and the report's ``caveats``, because a ceiling nobody can see the doubts of
    is worse than one that names them.
    """
    if hardware.peak_source not in {PEAK_SOURCE_EMPIRICAL, PEAK_SOURCE_DATASHEET}:
        raise CeilingContractError(f"unknown peak_source {hardware.peak_source!r}")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, str) or not raw_cases:
        raise CeilingContractError("response must carry a non-empty 'cases' array")

    expected = list(dict.fromkeys(str(case_id) for case_id in expected_case_ids))
    if not expected:
        raise CeilingContractError("no scored case ids to produce a ceiling for")

    observed = dict(observed_ms or {})
    by_id: dict[str, CaseCeiling] = {}
    caveats: list[str] = [str(entry).strip() for entry in (payload.get("caveats") or ()) if str(entry).strip()]

    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise CeilingContractError("every entry of 'cases' must be an object")
        case_id = str(raw_case.get("case_id") or "").strip()
        if not case_id:
            raise CeilingContractError("a case entry has no 'case_id'")
        if case_id in by_id:
            raise CeilingContractError(f"case {case_id!r} appears twice")

        raw_stages = raw_case.get("stages")
        if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, str) or not raw_stages:
            raise CeilingContractError(f"case {case_id!r} must declare at least one stage")

        stages = tuple(
            _parse_stage(raw_stage, case_id=case_id, index=index) for index, raw_stage in enumerate(raw_stages)
        )
        timings: list[StageTiming] = []
        issues: list[str] = []
        for stage in stages:
            timing, stage_issues = _time_stage(stage, hardware)
            timings.append(timing)
            issues.extend(stage_issues)

        t_ideal_ms = sum(timing.t_stage_s for timing in timings) * 1000.0
        if t_ideal_ms <= 0:
            raise CeilingContractError(f"case {case_id!r} derives a non-positive ideal latency")

        # Guide section 10: a ceiling above what the box was observed doing is
        # not a ceiling. Report it rather than clamping -- the model is wrong and
        # clamping would hide which part.
        seen = observed.get(case_id)
        if seen is not None and t_ideal_ms > float(seen):
            issues.append(
                f"ideal latency {t_ideal_ms:.6g} ms exceeds the observed {float(seen):.6g} ms; "
                "the work model overstates the minimum legal work"
            )

        by_id[case_id] = CaseCeiling(
            case_id=case_id,
            t_ideal_ms=t_ideal_ms,
            bound=classify_bound(timings),
            stages=stages,
            timings=tuple(timings),
            bound_note=str(raw_case.get("bound_note") or "").strip(),
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

    if not hardware.is_empirical:
        caveats.append(
            "Peaks are vendor datasheet figures, not measured on this box: these latencies are an "
            "absolute lower bound that no implementation reaches, not an achievable target."
        )
    if hardware.dispatch_floor_s <= 0:
        caveats.append("Dispatch floor was not measured; every latency term is charged at zero.")

    return CeilingReport(
        schema_version=SCHEMA_VERSION,
        canonical_id=canonical_id,
        hardware=hardware,
        cases=tuple(by_id[case_id] for case_id in expected),
        confidence=confidence,
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
        hbm_bw_bytes_per_s=float(raw_hardware.get("hbm_bw_bytes_per_s") or 0.0),
        peak_flops={str(key): float(value) for key, value in (raw_hardware.get("peak_flops") or {}).items()},
        peak_source=str(raw_hardware.get("peak_source") or ""),
        dispatch_floor_s=float(raw_hardware.get("dispatch_floor_s") or 0.0),
        provenance=dict(raw_hardware.get("provenance") or {}),
    )

    cases: list[CaseCeiling] = []
    for raw_case in payload.get("cases") or ():
        if not isinstance(raw_case, Mapping):
            raise CeilingContractError("published report has a malformed case entry")
        derivation = raw_case.get("derivation") or {}
        stages = tuple(
            _parse_stage(raw_stage, case_id=str(raw_case.get("case_id") or "?"), index=index)
            for index, raw_stage in enumerate(derivation.get("stages") or ())
        )
        timings = tuple(
            StageTiming(
                name=str(raw_timing.get("name") or ""),
                t_compute_s=float(raw_timing.get("t_compute_s") or 0.0),
                t_memory_s=float(raw_timing.get("t_memory_s") or 0.0),
                t_latency_s=float(raw_timing.get("t_latency_s") or 0.0),
                t_stage_s=float(raw_timing.get("t_stage_s") or 0.0),
                peak_flops_used=float(raw_timing.get("peak_flops_used") or 0.0),
                bw_used=float(raw_timing.get("bw_used") or 0.0),
            )
            for raw_timing in (derivation.get("timings") or ())
        )
        observed = raw_case.get("profiler_observed_ms")
        cases.append(
            CaseCeiling(
                case_id=str(raw_case.get("case_id") or ""),
                t_ideal_ms=float(raw_case.get("t_ideal_ms") or 0.0),
                bound=str(raw_case.get("bound") or BOUND_MIXED),
                stages=stages,
                timings=timings,
                bound_note=str(raw_case.get("bound_note") or ""),
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
        caveats=tuple(str(entry) for entry in (payload.get("caveats") or ())),
    )


__all__ = [
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
    "Stage",
    "StageTiming",
    "build_report",
    "classify_bound",
    "load_report",
    "response_schema",
]
