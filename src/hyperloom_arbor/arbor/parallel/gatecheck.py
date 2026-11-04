"""The integration A/B's decision: an A/B job's runs plus the pinned gates -> a GateRecord.

An A/B job deploys the integrated tree (arm B) and it plus one addition (arm C) on
one node and replays the benchmark in the operator's order (e.g. B C C B), so
both arms see the same node and the same drift. This module turns those runs
into a verdict. It is a pure function: the job writes runs, this decides, and
the integrator applies.

The order of the checks is the point:

1. A run that cannot be trusted is excluded, never averaged: a deploy that did
   not match its blobs, a foreign process on the GPUs, requests that did not
   all complete, a missing metric.
2. The control arm is the instrument. If B fails to boot, fails correctness,
   or spreads wider than the operator's cap, nothing can be said about C:
   NOT_MEASURED: the integrator retries once on another node.
3. C failing to serve, or failing correctness while B passes, is a property of
   the candidate: REVERT.
4. Otherwise Arbor's own `run_gate` decides on the arm means, with the
   candidate's declared intent choosing the bar.
5. A change that declared a log marker and never printed it did not run
   (AGENTS.md section 10): a KEEP without it is downgraded to REVERT.

Ported verbatim from arbor/mesh3/gatecheck.py (<fork>/mesh3-confirm a691edc).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from arbor.measure.gate import _correctness_is_infrastructure, run_gate
from arbor.measure.intents import resolve

KEEP, REVERT, NOT_MEASURED = "KEEP", "REVERT", "NOT_MEASURED"


@dataclass(frozen=True)
class ArmRun:
    """One replay in a gate job, as the job recorded it."""
    arm: str                                  # "B" (integrated base) or "C" (candidate)
    order: int                                # position in the job's ORDER
    metric: float | None = None
    secondary: float | None = None
    booted: bool = True
    correctness: dict = field(default_factory=lambda: {"passed": True})
    num_requests_total: int | None = None
    num_requests_successful: int | None = None
    deploy_ok: bool = True
    foreign_gpu: bool = False
    marker_count: int | None = None           # matches of the declared marker in this run's log
    warmup: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArmRun":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class GateRecord:
    verdict: str                              # KEEP | REVERT | NOT_MEASURED
    reason: str
    intent: str
    base_mean: float | None = None
    cand_mean: float | None = None
    delta_pct: float | None = None
    base_spread_pct: float | None = None
    base_runs: int = 0
    cand_runs: int = 0
    booked: bool = False
    frontier: bool = False
    excluded: list = field(default_factory=list)   # [order, reason]
    gate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _exclusion(run: ArmRun) -> str | None:
    if run.warmup:
        return "warmup"
    if not run.deploy_ok:
        return "deploy did not match its blobs"
    if run.foreign_gpu:
        return "a foreign GPU process was present"
    if run.booted and (run.num_requests_total is not None
                       and run.num_requests_successful is not None
                       and run.num_requests_successful != run.num_requests_total):
        return (f"{run.num_requests_successful} of {run.num_requests_total} "
                f"requests completed")
    return None


def _correctness_failed(run: ArmRun) -> bool:
    """A real correctness failure, not a harness one (404, unreachable, inconclusive)."""
    c = run.correctness or {}
    if c.get("passed", False):
        return False
    return not _correctness_is_infrastructure(str(c.get("error", "")))


def _spread_pct(values: list[float]) -> float:
    m = mean(values)
    return (max(values) - min(values)) / m * 100.0 if m else 0.0


def evaluate(runs: list[ArmRun], gates: dict[str, Any], intent: str,
             marker: str | None = None) -> GateRecord:
    """Decide one candidate from its gate job's runs.

    ``gates`` is the operator's integration block: ``objective`` (metric,
    direction, kind, measurement as in a pack), ``min_runs_per_arm`` and
    ``max_control_spread_pct``.
    """
    intent_name = resolve(intent).name
    objective = gates["objective"]
    min_runs = int(gates.get("min_runs_per_arm", 2))
    spread_cap = float(gates.get("max_control_spread_pct", 100.0))

    excluded, kept = [], []
    for r in sorted(runs, key=lambda r: r.order):
        why = _exclusion(r)
        if why:
            excluded.append([r.order, why])
        else:
            kept.append(r)
    b_runs = [r for r in kept if r.arm == "B"]
    c_runs = [r for r in kept if r.arm == "C"]

    def record(verdict, reason, **kw) -> GateRecord:
        return GateRecord(verdict=verdict, reason=reason, intent=intent_name,
                          excluded=excluded, **kw)

    # 2. The control arm is the instrument.
    if any(not r.booted for r in b_runs):
        return record(NOT_MEASURED, "the base arm failed to boot: the node or the "
                                    "harness is suspect, not the candidate")
    if any(_correctness_failed(r) for r in b_runs):
        return record(NOT_MEASURED, "the base arm failed correctness: the instrument "
                                    "is suspect, not the candidate")

    # 3. The candidate failing to serve or to be correct is the candidate's.
    c_boot_failures = sum(1 for r in c_runs if not r.booted)
    if c_boot_failures >= 2:
        return record(REVERT, f"the candidate failed to boot in {c_boot_failures} "
                              f"runs while the base booted")
    if any(_correctness_failed(r) for r in c_runs):
        return record(REVERT, "the candidate failed correctness while the base passed")

    b_vals = [r.metric for r in b_runs if r.booted and r.metric is not None]
    c_vals = [r.metric for r in c_runs if r.booted and r.metric is not None]
    if len(b_vals) < min_runs or len(c_vals) < min_runs:
        return record(NOT_MEASURED, f"too few usable runs: base {len(b_vals)}, "
                                    f"candidate {len(c_vals)}, need {min_runs} each",
                      base_runs=len(b_vals), cand_runs=len(c_vals))

    # A harness that could not score the candidate has not shown it correct.
    if not any((r.correctness or {}).get("passed") for r in c_runs if r.booted):
        return record(NOT_MEASURED, "no candidate run produced a correctness result: the "
                                    "harness is suspect, not the candidate",
                      base_runs=len(b_vals), cand_runs=len(c_vals))

    b_mean, c_mean = mean(b_vals), mean(c_vals)
    spread = _spread_pct(b_vals)
    stats = dict(base_mean=b_mean, cand_mean=c_mean, base_spread_pct=spread,
                 base_runs=len(b_vals), cand_runs=len(c_vals))
    if spread > spread_cap:
        return record(NOT_MEASURED, f"the base arm spread {spread:.2f}% exceeds the "
                                    f"{spread_cap:.2f}% cap: the node cannot resolve this",
                      **stats)

    # 4. Arbor's gate on the arm means, the candidate's intent choosing the bar.
    metric_key = objective["metric"]
    sec_key = objective.get("measurement", {}).get("secondary_metric")
    bench = {"status": "ok", metric_key: c_mean}
    b_sec = [r.secondary for r in b_runs if r.booted and r.secondary is not None]
    c_sec = [r.secondary for r in c_runs if r.booted and r.secondary is not None]
    sec_best = None
    if sec_key and b_sec and c_sec:
        bench[sec_key] = mean(c_sec)
        sec_best = mean(b_sec)
    gate = run_gate(objective, b_mean, bench, {"passed": True}, True,
                    intent=intent_name, secondary_best=sec_best)
    verdict = gate.verdict if gate.verdict in (KEEP, REVERT) else NOT_MEASURED
    result = record(verdict, gate.reason, delta_pct=gate.delta_pct, booked=gate.booked,
                    frontier=gate.frontier, gate_reason=gate.reason, **stats)

    # 5. A declared change that never logged its marker did not run.
    if marker and verdict == KEEP:
        seen = sum(r.marker_count or 0 for r in c_runs if r.booted)
        if seen == 0:
            return record(REVERT, f"the candidate never logged its marker {marker!r}: "
                                  f"the change did not run", delta_pct=gate.delta_pct,
                          gate_reason=gate.reason, **stats)
    return result
