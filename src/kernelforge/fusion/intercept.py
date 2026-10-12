# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-check a fusion opportunity against a CUDA-graph-ON trace of the same workload.

Discovery reads a CUDA-graph-DISABLED trace, because graph replay amortizes the launches fusion exists to remove and
the tail vanishes with graphs on. That makes the cgoff trace the only place the opportunity is visible -- and also the
reason its size is not the opportunity production still has. :func:`predict_cuda_graph_on_gain` bridges the two with a
flat discount when no measured calibration is loaded, and the calibration finding recorded in :mod:`.diagnose` says the
share it discounts is a poor discriminator of real cg-ON gain.

This module answers the same question by measurement instead: given both traces, how much of the cgoff opportunity did
replay already take, and how much survives it. The split rests on fusion buying two separable things.

* **Launches.** The gaps between tiny kernels. Graph replay removes most of these, so this half is what "already
  intercepted" means, and it is visible as the rise in busy-fraction-of-wall between the two traces.
* **Memory traffic.** A fused kernel reads its inputs once and keeps intermediates in registers instead of round-
  tripping them through HBM. Replay does nothing for this, so this half survives and is what is still worth authoring.

Replay does not change *which* kernels run, only the gaps between them, so a launch-bound busy share that moves sharply
between the two traces means they did not capture the same code path. That is reported as non-comparable rather than
folded into a verdict, because every number below would otherwise be comparing two different workloads.

That same invariant is why the verdict cannot rest on the predicted gain alone. Absent measured calibration and op
shapes, :func:`predict_cuda_graph_on_gain` is a function of the launch-bound share -- the one quantity replay leaves
alone -- so running it on either trace returns the same answer and the second trace would add nothing. The two signals
that do come from the cg-ON trace are the gap it closed and the traffic the surviving tail still moves, and the verdict
is decided on those.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import click

from .calibration import DEFAULT_MIN_PREDICTED_GAIN, predict_cuda_graph_on_gain
from .diagnose import (
    LAUNCH_BOUND_CATEGORIES,
    load_op_busy_from_kineto_trace,
    load_op_bytes_from_kineto_trace,
)

#: How far the launch-bound busy share may move between the two traces before they stop describing one workload.
#: Relative to the cgoff share, so a 0.40 -> 0.22 move is tolerated and 0.40 -> 0.05 is not.
DEFAULT_MAX_SHARE_DRIFT = 0.5

#: Gap the cgoff trace must show before "replay reclaimed it" is a statement about anything. Below this the workload
#: was not dispatch-bound to begin with and the comparison has no launch overhead to attribute.
DEFAULT_MIN_CGOFF_GAP = 0.05

#: Idle wall still left with graphs on before the launch channel counts as open. Replay is supposed to close these
#: gaps; when this much is still idle it did not, and fusing the chain can still remove launches.
DEFAULT_MAX_RESIDUAL_GAP = 0.10

VERDICT_HEADROOM = "headroom_remains"
VERDICT_INTERCEPTED = "intercepted"
VERDICT_NOT_COMPARABLE = "not_comparable"
VERDICT_GRAPHS_NOT_ACTIVE = "graphs_not_active"
VERDICT_UNREADABLE = "unreadable"


@dataclass
class TraceFacts:
    """What one trace contributes to the cross-check."""

    path: str
    #: Fraction of GPU-busy time in the launch-bound categories -- the same sum ``diagnose_from_shares`` gates on.
    launch_bound_share: float
    #: ``None`` when the trace carried no usable timestamps, which leaves the gap half of the split unanswerable.
    busy_fraction_of_wall: Optional[float]
    #: Measured launch-bound memory-traffic share, or ``None`` when the trace exposed no op shapes to derive it from.
    launch_bound_mem_share: Optional[float]
    kernels: float

    @property
    def gap_fraction(self) -> Optional[float]:
        """Fraction of wall the GPU was idle -- launch and dispatch overhead, when decode is the only work."""
        if self.busy_fraction_of_wall is None:
            return None
        return max(0.0, 1.0 - float(self.busy_fraction_of_wall))

    @property
    def readable(self) -> bool:
        return self.launch_bound_share > 0.0 or self.kernels > 0.0


@dataclass
class InterceptReport:
    """What replay already took, what survives it, and how far the cgoff trace overstates the opportunity."""

    cgoff: TraceFacts
    cgon: TraceFacts
    #: Wall fraction replay reclaimed: the gap that closed between the two traces.
    intercepted_gap: Optional[float]
    #: Gap still open with graphs on. Replay did not amortize these launches, so fusion can still reach them.
    residual_gap: Optional[float]
    #: Gain still available measured on the cg-ON trace -- the memory-traffic half, which replay cannot take.
    headroom_gain: float
    #: What discovery predicts from the cgoff trace alone, i.e. the number forge-fuse ranks on today. Reported beside
    #: ``headroom_gain`` rather than differenced against it: both estimates read the same op shapes, which a second
    #: capture of one workload does not change, so their difference measures nothing.
    cgoff_predicted_gain: float
    #: Share of the idle time the cgoff trace showed that replay had already taken. This is the launch channel's
    #: answer, and the one number here that needs both traces to exist.
    gap_intercepted_fraction: Optional[float]
    verdict: str
    reason: str

    @property
    def comparable(self) -> bool:
        return self.verdict not in (VERDICT_NOT_COMPARABLE, VERDICT_UNREADABLE)

    @property
    def worth_authoring(self) -> bool:
        """Whether anything survives replay by the acceptance bar. False on every non-comparable verdict."""
        return self.verdict == VERDICT_HEADROOM

    def to_dict(self) -> dict[str, Any]:
        """The report as plain JSON-able data, gap properties included."""
        out = asdict(self)
        for key, facts in (("cgoff", self.cgoff), ("cgon", self.cgon)):
            out[key]["gap_fraction"] = facts.gap_fraction
        out["comparable"] = self.comparable
        out["worth_authoring"] = self.worth_authoring
        return out


def trace_facts(trace_path: str | Path) -> TraceFacts:
    """Read one kineto trace into the facts the cross-check needs."""
    shares, busy_of_wall, kernels = load_op_busy_from_kineto_trace(trace_path)
    launch_bound = sum(v for k, v in (shares or {}).items() if k in LAUNCH_BOUND_CATEGORIES)
    bytes_share = load_op_bytes_from_kineto_trace(trace_path)
    # Distinguish "no op shapes in the trace" from "shapes present, none launch-bound": the first leaves the memory
    # channel unmeasured and must fall back to the flat discount, the second is a measured zero.
    mem_share = sum(v for k, v in bytes_share.items() if k in LAUNCH_BOUND_CATEGORIES) if bytes_share else None
    return TraceFacts(
        path=str(trace_path),
        launch_bound_share=float(launch_bound),
        busy_fraction_of_wall=busy_of_wall,
        launch_bound_mem_share=mem_share,
        kernels=float(kernels),
    )


def _share_drift(cgoff: float, cgon: float, *, max_drift: float) -> float | None:
    """Relative move in launch-bound share, or ``None`` when it stays within *max_drift*."""
    if cgoff <= 0.0:
        return None
    drift = abs(cgoff - cgon) / cgoff
    return drift if drift > max_drift else None


def cross_check_intercept(
    cgoff_trace: str | Path,
    cgon_trace: str | Path,
    *,
    decode_batch: int = 16,
    min_gain: float = DEFAULT_MIN_PREDICTED_GAIN,
    max_share_drift: float = DEFAULT_MAX_SHARE_DRIFT,
    min_cgoff_gap: float = DEFAULT_MIN_CGOFF_GAP,
    max_residual_gap: float = DEFAULT_MAX_RESIDUAL_GAP,
) -> InterceptReport:
    """Decide whether CUDA-graph replay already took the opportunity the cgoff trace shows."""
    off = trace_facts(cgoff_trace)
    on = trace_facts(cgon_trace)

    # The cgoff prediction is reproduced exactly as discovery computes it, so the overstatement below is measured
    # against the number that actually ranks candidates rather than a restatement of it.
    cgoff_predicted = predict_cuda_graph_on_gain(
        off.launch_bound_share, decode_batch=decode_batch, mem_share=off.launch_bound_mem_share
    )
    # The same estimator on the cg-ON trace. Everything it reads -- the surviving launch-bound share and the traffic
    # that share still moves -- is measured with replay in effect, so what it returns is what replay left behind.
    headroom = predict_cuda_graph_on_gain(
        on.launch_bound_share, decode_batch=decode_batch, mem_share=on.launch_bound_mem_share
    )

    gap_off, gap_on = off.gap_fraction, on.gap_fraction
    intercepted = max(0.0, gap_off - gap_on) if (gap_off is not None and gap_on is not None) else None
    taken_fraction = intercepted / gap_off if (intercepted is not None and gap_off and gap_off > 0) else None

    def _report(verdict: str, reason: str) -> InterceptReport:
        return InterceptReport(
            cgoff=off,
            cgon=on,
            intercepted_gap=intercepted,
            residual_gap=gap_on,
            headroom_gain=headroom,
            cgoff_predicted_gain=cgoff_predicted,
            gap_intercepted_fraction=taken_fraction,
            verdict=verdict,
            reason=reason,
        )

    if not off.readable or not on.readable:
        missing = ", ".join(f.path for f in (off, on) if not f.readable)
        return _report(VERDICT_UNREADABLE, f"no kernel events parsed from {missing}")

    drift = _share_drift(off.launch_bound_share, on.launch_bound_share, max_drift=max_share_drift)
    if drift is not None:
        return _report(
            VERDICT_NOT_COMPARABLE,
            f"launch-bound share moved {off.launch_bound_share:.3f} -> {on.launch_bound_share:.3f} "
            f"({drift * 100:.0f}% > {max_share_drift * 100:.0f}%); replay changes the gaps between kernels, not "
            f"which kernels run, so these two traces did not capture the same code path",
        )

    # A capture mistake this cheap to make deserves naming rather than a verdict computed from it: graphs left off in
    # the "cg-ON" run leaves both gaps open, and the comparison would then read as "nothing was intercepted".
    if gap_off is not None and gap_on is not None and gap_off >= min_cgoff_gap and gap_on >= gap_off:
        return _report(
            VERDICT_GRAPHS_NOT_ACTIVE,
            f"the cg-ON trace is no busier than the cgoff one ({on.busy_fraction_of_wall:.2f} vs "
            f"{off.busy_fraction_of_wall:.2f} of wall), so replay closed no gap; check that graphs were enabled "
            f"for that capture",
        )

    # The memory channel is the half replay cannot touch, so a measured one clearing the bar settles the question on
    # its own.
    if headroom >= min_gain:
        return _report(
            VERDICT_HEADROOM,
            f"{headroom:.3f} predicted gain survives replay (bar {min_gain:.3f}); the launch-bound tail still "
            f"holds {on.launch_bound_share:.3f} of GPU-busy time with graphs on, and fusion saves its memory "
            f"traffic whatever replay did to the launches",
        )

    # Independently of traffic: replay is what is supposed to have closed the gaps, and a cg-ON trace still sitting
    # idle says it did not, which leaves the launch half of the opportunity open. This is the one signal that comes
    # only from the second trace -- the estimator above reads the launch-bound share, which replay does not change.
    if gap_on is not None and gap_on > max_residual_gap:
        return _report(
            VERDICT_HEADROOM,
            f"only {headroom:.3f} predicted gain survives on traffic, but {gap_on:.3f} of wall is still idle with "
            f"graphs on (bar {max_residual_gap:.3f}): replay did not close these gaps, so the launches are still "
            f"there to remove",
        )

    taken = (
        f"replay took {taken_fraction * 100:.0f}% of the idle time the cgoff trace showed and "
        if taken_fraction is not None
        else ""
    )
    return _report(
        VERDICT_INTERCEPTED,
        f"{taken}left {gap_on if gap_on is not None else 0.0:.3f} of wall idle; only {headroom:.3f} predicted gain "
        f"survives on traffic, below the {min_gain:.3f} bar. Both channels are spent, so authoring this chain buys "
        f"what replay already delivered",
    )


def _pct(value: Optional[float]) -> str:
    """Percent for humans; ``n/a`` when the trace could not supply the number."""
    return "n/a" if value is None else f"{value * 100:5.1f}%"


def format_report(report: InterceptReport) -> str:
    """The report as an operator-readable block."""
    off, on = report.cgoff, report.cgon
    lines = [
        f"verdict: {report.verdict}",
        f"  {report.reason}",
        "",
        f"{'':22}{'cuda-graph OFF':>16}{'cuda-graph ON':>16}",
        f"{'launch-bound share':22}{_pct(off.launch_bound_share):>16}{_pct(on.launch_bound_share):>16}",
        f"{'GPU busy of wall':22}{_pct(off.busy_fraction_of_wall):>16}{_pct(on.busy_fraction_of_wall):>16}",
        f"{'idle gap':22}{_pct(off.gap_fraction):>16}{_pct(on.gap_fraction):>16}",
        f"{'launch-bound traffic':22}{_pct(off.launch_bound_mem_share):>16}{_pct(on.launch_bound_mem_share):>16}",
        f"{'kernels':22}{int(off.kernels):>16d}{int(on.kernels):>16d}",
        "",
        f"replay reclaimed           {_pct(report.intercepted_gap)} of wall "
        f"({_pct(report.gap_intercepted_fraction)} of the idle time cgoff showed)",
        f"gap still open with graphs {_pct(report.residual_gap)} of wall",
        "",
        f"gain predicted from the cgoff trace alone  {_pct(report.cgoff_predicted_gain)}",
        f"gain still available with graphs on        {_pct(report.headroom_gain)}",
    ]
    return "\n".join(lines)


@click.command("fusion-intercept")
@click.option(
    "--cgoff-trace",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Decode kineto trace captured with CUDA graphs DISABLED (the one discovery reads).",
)
@click.option(
    "--cgon-trace",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Decode kineto trace of the SAME workload captured with CUDA graphs ENABLED.",
)
@click.option("--decode-batch", default=16, type=int, help="Decode batch the traces were captured at.")
@click.option(
    "--min-gain",
    default=DEFAULT_MIN_PREDICTED_GAIN,
    type=float,
    show_default=True,
    help="Acceptance bar for surviving gain, as a fraction.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the report as JSON instead of a table.")
def fusion_intercept(
    cgoff_trace: str,
    cgon_trace: str,
    decode_batch: int,
    min_gain: float,
    as_json: bool,
) -> None:
    """Cross-check a cgoff fusion opportunity against a CUDA-graph-ON trace of the same workload."""
    report = cross_check_intercept(
        cgoff_trace,
        cgon_trace,
        decode_batch=decode_batch,
        min_gain=min_gain,
    )
    click.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True) if as_json else format_report(report))
    # A non-comparable pair is an input problem the caller has to fix, so it must not read as "nothing to fuse".
    if not report.comparable or report.verdict == VERDICT_GRAPHS_NOT_ACTIVE:
        raise SystemExit(2)
