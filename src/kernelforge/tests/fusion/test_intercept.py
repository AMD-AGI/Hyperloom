# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the cg-OFF/cg-ON cross-check that says whether graph replay already took a fusion opportunity."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from kernelforge.fusion.intercept import (
    VERDICT_GRAPHS_NOT_ACTIVE,
    VERDICT_HEADROOM,
    VERDICT_INTERCEPTED,
    VERDICT_NOT_COMPARABLE,
    VERDICT_UNREADABLE,
    cross_check_intercept,
    fusion_intercept,
    trace_facts,
)

# Names chosen so ``categorize_kernel_name`` files them where the test intends: ``rmsnorm`` and ``cudafunctor_add``
# are launch-bound categories, ``gemm`` and ``paged_attention`` are the compute anchors.
_TAIL = ("rmsnorm_kernel", "cudafunctor_add")
_COMPUTE = ("gemm_kernel", "paged_attention_kernel")


def _trace(
    path: Path,
    *,
    tail_kernels: int,
    tail_us: float,
    compute_us: float,
    gap_us: float,
    tail_bytes: tuple[int, ...] = (),
    compute_bytes: tuple[int, ...] = (),
) -> str:
    """Write a kineto trace with a controllable idle gap between kernels.

    ``gap_us`` is inserted after every kernel, so the busy-of-wall fraction the diagnosis derives is driven by it:
    zero gap means back-to-back kernels, which is what graph replay produces.
    """
    events: list[dict] = []
    ts = 100.0
    for i in range(tail_kernels):
        events.append({"cat": "kernel", "name": _TAIL[i % len(_TAIL)], "ts": ts, "dur": tail_us})
        ts += tail_us + gap_us
    for i, name in enumerate(_COMPUTE):
        events.append({"cat": "kernel", "name": name, "ts": ts, "dur": compute_us})
        ts += compute_us + gap_us
    # ``cpu_op`` events are where the memory channel comes from; without them the estimator falls back to the flat
    # launch-share discount.
    for i, nbytes in enumerate(tail_bytes):
        events.append(
            {
                "cat": "cpu_op",
                "name": "aten::rms_norm" if i % 2 == 0 else "aten::add",
                "ts": 100.0,
                "dur": 1.0,
                "args": {"Input Dims": [[nbytes // 2]], "Input type": ["bfloat16"]},
            }
        )
    for nbytes in compute_bytes:
        events.append(
            {
                "cat": "cpu_op",
                "name": "aten::matmul",
                "ts": 100.0,
                "dur": 1.0,
                "args": {"Input Dims": [[nbytes // 2]], "Input type": ["bfloat16"]},
            }
        )
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return str(path)


def _pair(tmp_path: Path, *, off_gap: float, on_gap: float, **kw) -> tuple[str, str]:
    """The same workload captured twice, differing only in the idle gap replay would close."""
    off = _trace(tmp_path / "cgoff.json", gap_us=off_gap, **kw)
    on = _trace(tmp_path / "cgon.json", gap_us=on_gap, **kw)
    return off, on


class TestTraceFacts:
    def test_the_idle_gap_is_what_separates_the_two_captures(self, tmp_path):
        off, on = _pair(tmp_path, off_gap=90.0, on_gap=0.0, tail_kernels=8, tail_us=10.0, compute_us=10.0)
        facts_off, facts_on = trace_facts(off), trace_facts(on)
        # Same kernels in both, so the busy share cannot move -- only the wall does.
        assert facts_off.launch_bound_share == facts_on.launch_bound_share
        assert facts_off.gap_fraction > 0.8
        assert facts_on.gap_fraction < 0.05

    def test_a_trace_without_op_shapes_leaves_the_memory_channel_unmeasured(self, tmp_path):
        path = _trace(tmp_path / "t.json", tail_kernels=4, tail_us=10.0, compute_us=10.0, gap_us=0.0)
        # ``None`` rather than 0.0: nothing was measured, which is what makes the estimator fall back.
        assert trace_facts(path).launch_bound_mem_share is None

    def test_op_shapes_yield_a_measured_launch_bound_traffic_share(self, tmp_path):
        path = _trace(
            tmp_path / "t.json",
            tail_kernels=4,
            tail_us=10.0,
            compute_us=10.0,
            gap_us=0.0,
            tail_bytes=(8000, 8000),
            compute_bytes=(2000,),
        )
        share = trace_facts(path).launch_bound_mem_share
        assert share is not None
        assert 0.8 < share < 0.9


class TestVerdict:
    def test_replay_closing_the_gaps_with_little_traffic_left_reads_as_intercepted(self, tmp_path):
        # A small launch-bound tail: most GPU time is the compute anchors, and the tail moves little traffic.
        off, on = _pair(
            tmp_path,
            off_gap=90.0,
            on_gap=0.0,
            tail_kernels=2,
            tail_us=5.0,
            compute_us=200.0,
            tail_bytes=(1000,),
            compute_bytes=(200000,),
        )
        report = cross_check_intercept(off, on)
        assert report.verdict == VERDICT_INTERCEPTED
        assert report.worth_authoring is False
        # Smaller than the small-kernel case above: the 90us gap is a lesser share of a wall dominated by two 200us
        # compute kernels. What matters is that replay took all of it and left nothing idle.
        assert report.intercepted_gap > 0.3
        assert report.residual_gap < 0.05

    def test_surviving_memory_traffic_keeps_the_opportunity_open(self, tmp_path):
        # Replay closed every gap, but the tail still moves most of the traffic, and fusion saves that regardless.
        off, on = _pair(
            tmp_path,
            off_gap=90.0,
            on_gap=0.0,
            tail_kernels=8,
            tail_us=20.0,
            compute_us=20.0,
            tail_bytes=(40000, 40000, 40000),
            compute_bytes=(1000,),
        )
        report = cross_check_intercept(off, on)
        assert report.verdict == VERDICT_HEADROOM
        assert report.worth_authoring is True
        # The gap half was fully taken; the verdict rests on traffic alone.
        assert report.residual_gap < 0.05
        assert report.headroom_gain >= 0.03

    def test_a_gap_replay_never_closed_keeps_the_launch_half_open(self, tmp_path):
        # Graphs on and still idle: whatever the traffic says, those launches are still there to remove. Traffic is
        # deliberately negligible so only the residual gap can carry this verdict.
        off, on = _pair(
            tmp_path,
            off_gap=90.0,
            on_gap=40.0,
            tail_kernels=2,
            tail_us=5.0,
            compute_us=200.0,
            tail_bytes=(1000,),
            compute_bytes=(200000,),
        )
        report = cross_check_intercept(off, on)
        assert report.verdict == VERDICT_HEADROOM
        assert report.headroom_gain < 0.03
        assert report.residual_gap > 0.10

    def test_a_cg_on_trace_no_busier_than_cgoff_is_a_capture_mistake_not_a_result(self, tmp_path):
        off, on = _pair(tmp_path, off_gap=90.0, on_gap=90.0, tail_kernels=4, tail_us=10.0, compute_us=10.0)
        report = cross_check_intercept(off, on)
        assert report.verdict == VERDICT_GRAPHS_NOT_ACTIVE
        # It must not read as "nothing was intercepted", which is the answer the numbers alone would give.
        assert report.verdict != VERDICT_INTERCEPTED
        assert "graphs were enabled" in report.reason

    def test_two_different_code_paths_are_refused_rather_than_compared(self, tmp_path):
        # Replay changes gaps, never which kernels run, so a collapsed launch-bound share means these are not the
        # same workload.
        off = _trace(tmp_path / "cgoff.json", tail_kernels=10, tail_us=20.0, compute_us=10.0, gap_us=90.0)
        on = _trace(tmp_path / "cgon.json", tail_kernels=1, tail_us=1.0, compute_us=400.0, gap_us=0.0)
        report = cross_check_intercept(off, on)
        assert report.verdict == VERDICT_NOT_COMPARABLE
        assert report.comparable is False
        assert report.worth_authoring is False

    def test_an_unreadable_trace_is_not_a_no_opportunity_answer(self, tmp_path):
        good = _trace(tmp_path / "cgoff.json", tail_kernels=4, tail_us=10.0, compute_us=10.0, gap_us=90.0)
        empty = tmp_path / "cgon.json"
        empty.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
        report = cross_check_intercept(good, str(empty))
        assert report.verdict == VERDICT_UNREADABLE
        assert report.comparable is False


class TestGapAccounting:
    def test_the_fraction_replay_took_is_reported_against_the_cgoff_idle_time(self, tmp_path):
        off, on = _pair(
            tmp_path,
            off_gap=90.0,
            on_gap=0.0,
            tail_kernels=4,
            tail_us=10.0,
            compute_us=100.0,
            tail_bytes=(2000,),
            compute_bytes=(400000,),
        )
        report = cross_check_intercept(off, on)
        # Every gap closed, so replay took all of the idle time the cgoff trace showed.
        assert report.gap_intercepted_fraction == 1.0
        assert report.intercepted_gap == report.cgoff.gap_fraction

    def test_a_partly_closed_gap_is_reported_as_a_partial_fraction(self, tmp_path):
        off, on = _pair(tmp_path, off_gap=90.0, on_gap=45.0, tail_kernels=4, tail_us=10.0, compute_us=10.0)
        report = cross_check_intercept(off, on)
        assert 0.0 < report.gap_intercepted_fraction < 1.0

    def test_both_estimates_are_reported_rather_than_differenced(self, tmp_path):
        # The two predictions read the same op shapes, so they agree; the report must show both instead of a
        # derived delta that would always be zero for one workload captured twice.
        off, on = _pair(
            tmp_path,
            off_gap=90.0,
            on_gap=0.0,
            tail_kernels=4,
            tail_us=10.0,
            compute_us=100.0,
            tail_bytes=(2000,),
            compute_bytes=(400000,),
        )
        report = cross_check_intercept(off, on)
        assert report.cgoff_predicted_gain == report.headroom_gain
        assert not hasattr(report, "overstatement")

    def test_the_bar_is_adjustable(self, tmp_path):
        off, on = _pair(
            tmp_path,
            off_gap=90.0,
            on_gap=0.0,
            tail_kernels=8,
            tail_us=20.0,
            compute_us=20.0,
            tail_bytes=(40000, 40000, 40000),
            compute_bytes=(1000,),
        )
        assert cross_check_intercept(off, on, min_gain=0.03).verdict == VERDICT_HEADROOM
        # Raise the bar above the surviving gain and the same pair reads as intercepted.
        assert cross_check_intercept(off, on, min_gain=0.99).verdict == VERDICT_INTERCEPTED


class TestCli:
    def test_the_command_prints_both_columns_and_a_verdict(self, tmp_path):
        off, on = _pair(
            tmp_path,
            off_gap=90.0,
            on_gap=0.0,
            tail_kernels=8,
            tail_us=20.0,
            compute_us=20.0,
            tail_bytes=(40000, 40000, 40000),
            compute_bytes=(1000,),
        )
        result = CliRunner().invoke(fusion_intercept, ["--cgoff-trace", off, "--cgon-trace", on])
        assert result.exit_code == 0, result.output
        assert "cuda-graph OFF" in result.output
        assert "cuda-graph ON" in result.output
        assert VERDICT_HEADROOM in result.output

    def test_json_mode_emits_the_whole_report(self, tmp_path):
        off, on = _pair(tmp_path, off_gap=90.0, on_gap=0.0, tail_kernels=4, tail_us=10.0, compute_us=10.0)
        result = CliRunner().invoke(fusion_intercept, ["--cgoff-trace", off, "--cgon-trace", on, "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["verdict"]
        assert payload["cgoff"]["gap_fraction"] > 0.8
        assert payload["cgon"]["gap_fraction"] < 0.05

    def test_a_capture_mistake_exits_nonzero_so_a_script_cannot_read_it_as_a_result(self, tmp_path):
        off, on = _pair(tmp_path, off_gap=90.0, on_gap=90.0, tail_kernels=4, tail_us=10.0, compute_us=10.0)
        result = CliRunner().invoke(fusion_intercept, ["--cgoff-trace", off, "--cgon-trace", on])
        assert result.exit_code == 2
        assert VERDICT_GRAPHS_NOT_ACTIVE in result.output
