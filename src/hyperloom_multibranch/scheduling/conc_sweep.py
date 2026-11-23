# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The concurrency sweep of a closed session: its root and the commit FINAL chose, over the recipe's concurrencies.

The points are the concurrencies the InferenceX arm of the session's frozen base publishes, or the operator's
``--conc-sweep-concs`` for a base no InferenceX arm gave, with the workload's CONC among them
(:func:`hyperloom.orchestrator.conc_sweep.runner.sweep_points`). It runs only when FINAL chose a commit other than the root whose
reproduction gained over the baseline. Both arms are submitted at once, so they run side by side where the session
runs two tests at once and one after the other where it runs one. An AgentX point is a test of its own, booted cold as
every AgentX level is, so each lands as it ends; an InferenceX arm is one sweep test over all its points on one server,
whose attempt timeout scales with them (:meth:`SweepPlan.test_points`). The report is the conc-sweep report
(``MB-reports/conc_sweep_summary.json``), rewritten after every test; a later run measures only the points not yet
measured. The tests are session measurements of the commits as they are: no accuracy, no checks.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.gain_math import conc_pair_comparison
from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions
from hyperloom.orchestrator.conc_sweep.points import SCHEMA_VERSION, SweepPoints, ladder, sweep_points
from hyperloom.orchestrator.conc_sweep.report import SETTLED, read_summary, write_curve, write_report
from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom_common.jsonio import read_json_strict
from hyperloom_multibranch.blackboard.codec import decode
from hyperloom_multibranch.contracts.toolbox import RequestRejected, TestWithdrawn
from hyperloom_multibranch.scheduling.final import MEASURE_PHASE, FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload

logger = logging.getLogger(__name__)

BASELINE, OPTIMIZED = "baseline", "optimized"
NO_OPTIMIZATION = "no_optimization"
NO_SWEEP_POINTS = "no_sweep_points"
_SUCCEEDED = "succeeded"


@dataclass(frozen=True)
class SweepPlan:
    """The commit FINAL chose and the points both arms run."""

    chosen: str
    points: SweepPoints

    def test_points(self, workload: Workload) -> int:
        """How many points one test of the sweep measures: one in an AgentX test, every point in an InferenceX one."""
        return 1 if workload.inferencex is None else len(self.points.concs)


def sweep_plan(layout: SessionLayout, settings: RunSettings) -> SweepPlan | str:
    """What the sweep of the closed session in ``layout`` measures, or why it measures nothing."""
    document = read_json_strict(layout.report, require_dict=True)
    choice = None if document["final"] is None else decode(FinalChoice, document["final"])
    chosen = None if choice is None else choice.chosen
    if chosen is None or chosen.root:
        return NO_OPTIMIZATION
    check = next(check for check in choice.checks if check.candidate == chosen)
    if not (check.gain_pct or 0.0) > 0:
        return NO_OPTIMIZATION
    from hyperloom.orchestrator.state.base_launch import frozen_base

    frozen = frozen_base(layout.toolbox)
    workload = settings.workload
    points = sweep_points(None if frozen is None else frozen.launch, workload.conc, settings.conc_sweep_concs)
    return NO_SWEEP_POINTS if points is None else SweepPlan(chosen.commit, points)


def sweep_unfinished(layout: SessionLayout, settings: RunSettings) -> bool:
    """Whether the session in ``layout`` closed, asked for a conc sweep, and that sweep has not run its course."""
    if not (settings.conc_sweep and layout.report.is_file()):
        return False
    summary = read_summary(layout.reports)
    return summary is None or summary["status"] not in SETTLED


def _header(layout: SessionLayout, workload: Workload) -> dict[str, Any]:
    shape = workload.inferencex
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": layout.session_id,
        "benchmark_mode": "agentx" if shape is None else "synthetic",
        "isl": None if shape is None else shape.isl,
        "osl": None if shape is None else shape.osl,
        "tp": workload.tp,
    }


def write_skipped(layout: SessionLayout, workload: Workload, reason: str) -> dict[str, Any]:
    """The report of a sweep that measured nothing, written where a sweep's report goes."""
    payload = {
        **_header(layout, workload),
        "status": "skipped",
        "skip_reason": reason,
        "baseline": {"points": []},
        "optimized": {"points": []},
        "summary": {},
    }
    write_report(payload, layout.reports)
    logger.info("conc_sweep: skipped (%s)", reason)
    return payload


def points_of(arm: str, concs: Sequence[int], result: TestResult) -> list[dict[str, Any]]:
    """One report row per concurrency of ``concs``: the mean of every metric over the test's measure rounds there."""
    measured: defaultdict[int, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in result.rounds:
        if row.phase == MEASURE_PHASE and row.concurrency in concs:
            for key, value in row.metrics.items():
                measured[row.concurrency][key].append(value)
    rows = []
    for conc in concs:
        metrics = {key: statistics.fmean(values) for key, values in measured[conc].items()}
        ok = result.validity is Validity.SUCCEEDED and bool(metrics)
        error = None if ok else result.failure_reason or "the test measured nothing at this concurrency"
        status = _SUCCEEDED if ok else "failed"
        rows.append({"arm": arm, "conc": conc, "status": status, "request_id": result.request_id, "error": error})
        rows[-1].update(metrics)
    return rows


@dataclass
class ConcSweep:
    """One sweep's inputs and the points both arms have measured so far, by arm and concurrency."""

    plan: SweepPlan
    root: str
    layout: SessionLayout
    workload: Workload
    tests: SessionTests
    parallel: int
    per_point_sec: float
    estimated_sec: float = 0.0
    started_at: float = field(default_factory=time.time)
    measured: dict[str, dict[int, dict[str, Any]]] = field(default_factory=lambda: {BASELINE: {}, OPTIMIZED: {}})
    cancel_reason: str = ""

    def pending(self, arm: str) -> list[int]:
        done = self.measured[arm]
        order = ladder(self.plan.points.concs, self.workload.conc)
        return [conc for conc in order if done.get(conc, {}).get("status") != _SUCCEEDED]

    async def run(self, stop: Awaitable[int], grant_end_unix: float | None = None) -> dict[str, Any]:
        """Run both arms until they end or ``stop`` returns a signal, which takes their tests back."""
        self._resume()
        per_arm = [self.per_point_sec * len(self.pending(arm)) for arm in (BASELINE, OPTIMIZED)]
        self.estimated_sec = max(per_arm) if self.parallel > 1 else sum(per_arm)
        self._log_plan(grant_end_unix)
        arms = asyncio.gather(self._arm(BASELINE, self.root), self._arm(OPTIMIZED, self.plan.chosen))
        stopped = asyncio.ensure_future(stop)
        await asyncio.wait({arms, stopped}, return_when=asyncio.FIRST_COMPLETED)
        if stopped.done():
            self.cancel_reason = f"signal {stopped.result()}"
            await self.tests.withdraw_integration(self.cancel_reason)
            await arms
        else:
            stopped.cancel()
        payload = self.payload(final=True)
        write_report(payload, self.layout.reports)
        shape = self.workload.inferencex
        options = ConcCurveOptions(
            self.workload.model_basename,
            self.workload.gpu_type,
            self.workload.tp,
            0 if shape is None else shape.isl,
            0 if shape is None else shape.osl,
        )
        write_curve(payload, self.layout.reports, options)
        logger.info(
            "conc_sweep: done status=%s successful_pairs=%s", payload["status"], payload["summary"]["successful_pairs"]
        )
        return payload

    async def _arm(self, arm: str, commit: str) -> None:
        pending = self.pending(arm)
        groups = [(conc,) for conc in pending] if self.workload.inferencex is None else [tuple(sorted(pending))]
        for concs in (group for group in groups if group):
            request = integration_test_request(TestKind.SWEEP, Purpose.HEALTH, commit, concurrency=concs)
            try:
                result = await self.tests.result(request)
            except TestWithdrawn:
                return
            except RequestRejected as exc:
                self._record([{"arm": arm, "conc": conc, "status": "failed", "error": str(exc)} for conc in concs])
                return
            self._record(points_of(arm, concs, result))

    def _record(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.measured[row["arm"]][row["conc"]] = row
        write_report(self.payload(), self.layout.reports)

    def _resume(self) -> None:
        """Keep the points an earlier run of the same sweep measured, so only the rest are run."""
        earlier = read_summary(self.layout.reports)
        if not earlier or earlier.get("concs") != list(self.plan.points.concs):
            return
        for arm in (BASELINE, OPTIMIZED):
            for point in earlier[arm]["points"]:
                if point["status"] == _SUCCEEDED:
                    self.measured[arm][point["conc"]] = point

    def _log_plan(self, grant_end_unix: float | None) -> None:
        logger.info(
            "conc_sweep: plan source=%s concs=%s arms=baseline,optimized %s pending_points=%d estimated=%.1fh "
            "(each point priced as the session's longest root test of %.0fs)",
            self.plan.points.source,
            list(self.plan.points.concs),
            "side by side" if self.parallel > 1 else "one after the other",
            len(self.pending(BASELINE)) + len(self.pending(OPTIMIZED)),
            self.estimated_sec / 3600.0,
            self.per_point_sec,
        )
        if grant_end_unix is None:
            return
        left = grant_end_unix - time.time()
        logger.info("conc_sweep: the run's grant has %.1fh left", left / 3600.0)
        if self.estimated_sec > left:
            logger.warning("conc_sweep: the estimate outlasts the run's grant; the platform may end the sweep early")

    def payload(self, *, final: bool = False) -> dict[str, Any]:
        sections = {
            arm: {"commit": commit, "points": [self.measured[arm][conc] for conc in sorted(self.measured[arm])]}
            for arm, commit in ((BASELINE, self.root), (OPTIMIZED, self.plan.chosen))
        }
        comparison, summary = conc_pair_comparison(
            sections[BASELINE]["points"], sections[OPTIMIZED]["points"], metric_key=self.workload.metric
        )
        return {
            **_header(self.layout, self.workload),
            "status": self._status(summary) if final else "in_progress",
            "concs": list(self.plan.points.concs),
            "concs_source": self.plan.points.source,
            BASELINE: sections[BASELINE],
            OPTIMIZED: sections[OPTIMIZED],
            "comparison": comparison,
            "summary": summary,
            "estimated_sec": round(self.estimated_sec),
            "elapsed_sec": round(time.time() - self.started_at, 2),
            "cancel_reason": self.cancel_reason,
        }

    def _status(self, summary: dict[str, Any]) -> str:
        if self.cancel_reason:
            return "cancelled"
        if not (self.pending(BASELINE) or self.pending(OPTIMIZED)):
            return _SUCCEEDED
        return "partial" if summary["successful_pairs"] else "failed"


__all__ = [
    "BASELINE",
    "NO_OPTIMIZATION",
    "NO_SWEEP_POINTS",
    "OPTIMIZED",
    "ConcSweep",
    "SweepPlan",
    "points_of",
    "sweep_plan",
    "sweep_unfinished",
    "write_skipped",
]
