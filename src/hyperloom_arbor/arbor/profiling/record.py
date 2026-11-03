"""The profile record: what gets written, and how a share is read back out.

The record is the contract between a profiler run and every evaluator that
projects a component win onto the whole step. Three rules it exists to enforce:

**Emit every op, not a top-N slice.** ``evalctl.sikl_share()`` computes
``matched / sum(ops)``. When the writer truncates ``ops`` to the 25 most
expensive kernels, that denominator is a subtotal and every share comes out
inflated — silently, and by an amount nobody can recover from the file. Writing
the full list makes the *existing, pinned* reader correct without editing it.

**Emit both key spellings.** The pinned reader wants ``name``/``us``; the older
SIKL log parser wrote ``op``/``ms``. Carrying both costs a few bytes and means
neither reader has to be the one that changes.

**Carry coverage.** A share is only as trustworthy as the fraction of the step
the attribution could name. :func:`component_share` refuses to return a number
when coverage is too low, because a confidently wrong share is worse than the
``None`` the old code returned by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from arbor.profiling.attribution import (
    Attribution,
    Component,
    attribute,
    components_from_patterns,
)
from arbor.profiling.trace import KernelTotals

SCHEMA_VERSION = 2

# Below this, a share is not reported. Chosen so that the DSv4 precedent — SIKL
# leaving 16.6% of prefill in `<rest>` and still being useful — sits inside the
# band, while a table that names only half the step does not.
MIN_COVERAGE = 0.70


@dataclass
class ProfileRecord:
    """One phase-isolated per-op profile of one engine step."""

    totals: KernelTotals
    attribution: Attribution
    phase: str = "decode"
    phase_verification: dict = field(default_factory=dict)
    steps_profiled: int = 1
    method: str = ""
    profiler: str = ""
    engine: dict = field(default_factory=dict)
    status: str = "ok"

    def to_dict(self, top: int = 25) -> dict:
        total = self.totals.total_us
        ops = [
            {
                # Keys evalctl.sikl_share() reads. The list is COMPLETE — see
                # the module docstring; truncating it corrupts the denominator.
                "name": name,
                "us": round(us, 3),
                # Legacy spelling, for any reader of the old SIKL record.
                "op": name[:60],
                "ms": round(us / 1000.0, 5),
                "pct": round(100.0 * us / total, 3) if total else 0.0,
                "count_per_step": round(cnt, 2),
            }
            for name, (us, cnt) in sorted(
                self.totals.by_name.items(), key=lambda kv: -kv[1][0]
            )
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "profiler": self.profiler,
            "method": self.method,
            "phase": self.phase,
            "phase_verification": self.phase_verification,
            "engine": self.engine,
            "steps_profiled": self.steps_profiled,
            "trace_files": self.totals.n_files,
            "distinct_kernels": len(self.totals.by_name),
            "total_gpu_us_per_step": round(total, 3),
            "attribution": self.attribution.to_dict(),
            "ops": ops,
            "ops_top": [o["name"] for o in ops[:top]],
            "top_bottleneck": ops[0]["name"] if ops else None,
        }

    def write(self, path: str | Path, top: int = 25) -> dict:
        data = self.to_dict(top=top)
        Path(path).write_text(json.dumps(data, indent=2))
        return data

    @classmethod
    def build(
        cls,
        totals: KernelTotals,
        *,
        components: tuple[Component, ...] | None = None,
        **kwargs,
    ) -> "ProfileRecord":
        from arbor.profiling.attribution import DEFAULT_COMPONENTS

        comps = components if components is not None else DEFAULT_COMPONENTS
        return cls(totals=totals, attribution=attribute(totals, comps), **kwargs)


def load_record(path: str | Path) -> dict | None:
    """Read a profile record, or None when absent or unparseable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def component_share(
    record: dict | str | Path,
    patterns: list[str],
    *,
    name: str = "query",
    min_coverage: float = MIN_COVERAGE,
) -> dict:
    """Fraction of step GPU time held by kernels matching ``patterns``.

    Returns a dict rather than a bare float, because the number alone is not
    enough to decide whether to trust it:

    ``share``      the fraction, or None when it must not be used
    ``coverage``   fraction of the step the record's own attribution could name
    ``matched_us`` / ``total_us``  the arithmetic, so a reader can check it
    ``verdict``    why a None is a None

    The three ways this returns None are deliberately distinguishable:
    no record on disk, a record whose attribution covers too little of the
    step to be believed, and a query that matched nothing. The last is the one
    that used to be indistinguishable from "this component is cheap".
    """
    data = record if isinstance(record, dict) else load_record(record)
    out: dict = {"component": name, "share": None, "patterns": list(patterns)}

    if data is None:
        out["verdict"] = "no profile record — run the profiler first"
        return out
    if data.get("status") not in (None, "ok"):
        out["verdict"] = f"profile status is {data.get('status')!r}, not ok"
        return out

    ops = data.get("ops") or []
    if isinstance(ops, dict):  # tolerate an older mapping form
        ops = [{"name": k, **(v if isinstance(v, dict) else {"us": v})} for k, v in ops.items()]

    # Prefer the writer's own total: it is the whole step even if some reader
    # or transport truncated `ops`. Fall back to summing only when absent.
    total = float(data.get("total_gpu_us_per_step") or 0.0)
    summed = 0.0
    matched = 0.0
    for op in ops:
        us = float(op.get("us") or op.get("gpu_us") or op.get("duration_us") or 0.0)
        if not us:
            ms = op.get("ms")
            us = float(ms) * 1000.0 if ms is not None else 0.0
        summed += us
        kernel = str(op.get("name") or op.get("op") or "").lower()
        if any(p.lower() in kernel for p in patterns):
            matched += us
    if total <= 0:
        total = summed
    elif summed > 0 and summed < 0.99 * total:
        out["ops_truncated"] = True
        out["ops_sum_us"] = round(summed, 3)

    out["total_us"] = round(total, 3)
    out["matched_us"] = round(matched, 3)

    attribution = data.get("attribution") or {}
    coverage = attribution.get("coverage")
    if coverage is None:
        # A v1 record has no attribution block. Compute coverage for this query
        # alone: we cannot vouch for the rest of the table, so say so.
        out["coverage"] = None
        out["verdict_note"] = "record predates attribution coverage; share is unchecked"
    else:
        out["coverage"] = coverage
        if coverage < min_coverage:
            out["verdict"] = (
                f"attribution covers only {100 * coverage:.1f}% of the step "
                f"(floor {100 * min_coverage:.0f}%) — share withheld; "
                f"extend the component patterns, see attribution.rest_top"
            )
            return out

    if total <= 0:
        out["verdict"] = "profile total is zero — no GPU time captured"
        return out
    if matched <= 0:
        out["verdict"] = (
            "no kernel matched the query patterns. This is a MISS, not a small "
            "component — the patterns do not describe any symbol this build "
            "emits. Check attribution.rest_top before assuming the component "
            "is cheap."
        )
        return out

    out["share"] = matched / total
    out["verdict"] = "ok"

    phase = data.get("phase_verification") or {}
    if phase.get("isolation") == "contaminated":
        out["warning"] = (
            f"profile phase isolation is CONTAMINATED ({phase.get('reason', '')}); "
            "share includes work from another phase"
        )
    elif phase.get("isolation") == "unverified":
        out["warning"] = (
            "profile phase isolation is UNVERIFIED — the phase label rests on "
            "launch arguments alone"
        )
    return out


def share_from_patterns(
    record_path: str | Path, patterns: list[str], name: str = "query"
) -> float | None:
    """Bare-float adapter for callers that only want the number.

    Drops the verdict and every warning, so prefer :func:`component_share`.
    Provided so an existing ``sikl_share(...)`` call site can be pointed here
    with a one-line change.
    """
    return component_share(record_path, patterns, name=name)["share"]


__all__ = [
    "MIN_COVERAGE",
    "SCHEMA_VERSION",
    "ProfileRecord",
    "component_share",
    "components_from_patterns",
    "load_record",
    "share_from_patterns",
]
