###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the semantic trace-validity probe (``_trace_probe``).

Every fixture below is a miniature of a capture observed in production, and the
assertions pin the measurement rather than the prose: the numbers in the
docstrings are the ones taken off the reference traces, so a threshold change
that would have let one of those captures through fails here.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _trace_probe as probe  # noqa: E402


# --------------------------------------------------------------------------- #
# Trace builders
# --------------------------------------------------------------------------- #


def _kernel(name: str, ts: float, dur: float, corr: int | None = None) -> dict:
    """A device kernel event."""
    ev = {"cat": "kernel", "ph": "X", "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": 7}
    if corr is not None:
        ev["args"] = {"correlation": corr}
    return ev


def _launch(name: str, ts: float, corr: int, ext_id: int = 1) -> dict:
    """A host-side runtime launch event."""
    return {
        "cat": "cuda_runtime",
        "ph": "X",
        "name": name,
        "ts": ts,
        "dur": 2.0,
        "args": {"correlation": corr, "External id": ext_id},
    }


def _step(ts: float, dur: float, idx: int) -> dict:
    """A per-iteration step annotation."""
    return {
        "cat": "gpu_user_annotation",
        "ph": "X",
        "name": f"step[DECODE bs=64] #{idx}",
        "ts": ts,
        "dur": dur,
    }


def _write(tmp_path: Path, events: list[dict], name: str = "rank_0.trace.json.gz") -> Path:
    """Write ``events`` as a gzipped Kineto trace and return the path."""
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"traceEvents": events}, fh)
    return path


def healthy_graph_trace(steps: int = 8, kernels_per_step: int = 4) -> list[dict]:
    """A graph-replay capture where every replay recorded its kernels.

    Mirrors the shape of the good ranks in the reference incident: one
    ``hipGraphLaunch`` per step, and kernels carrying that launch's correlation.
    """
    events: list[dict] = []
    corr = 1000
    for i in range(steps):
        base = i * 1000.0
        events.append(_step(base, 900.0, i))
        events.append(_launch("hipGraphLaunch", base + 1.0, corr))
        for k in range(kernels_per_step):
            events.append(_kernel(f"gemm_{k}", base + 10.0 + k * 100, 90.0, corr))
        corr += 1
    return events


def under_recorded_graph_trace(steps: int = 8) -> list[dict]:
    """A capture where only the final replay recorded kernels (1/N coverage).

    This is the Kimi-K3 / Qwen3.8 shape: 128 launches, 1 with kernels.
    """
    events: list[dict] = []
    corr = 1000
    for i in range(steps):
        base = i * 1000.0
        events.append(_step(base, 900.0, i))
        events.append(_launch("hipGraphLaunch", base + 1.0, corr + i))
    last = corr + steps - 1
    for k in range(64):
        events.append(_kernel(f"gemm_{k}", (steps - 1) * 1000.0 + 10.0 + k, 0.5, last))
    return events


# --------------------------------------------------------------------------- #
# GPU-record completeness
# --------------------------------------------------------------------------- #


def test_healthy_graph_trace_is_usable(tmp_path):
    """A fully recorded graph capture produces no findings."""
    result = probe.probe_file(_write(tmp_path, healthy_graph_trace()))
    assert result.verdict == probe.VERDICT_USABLE
    assert result.findings == []
    assert result.metrics["graph_replay_coverage"] == 1.0
    assert result.metrics["annotated_step_gpu_coverage"] == 1.0


def test_graph_replay_under_recording_is_detected(tmp_path):
    """1-of-N recorded replays fires the blocking under-recording finding.

    The reference incident measured 1/128 = 0.008 against 128/128 = 1.0 on
    sibling ranks of the same profile, with no values in between.
    """
    result = probe.probe_file(_write(tmp_path, under_recorded_graph_trace(steps=8)))
    codes = {f.code for f in result.findings}
    assert probe.GRAPH_REPLAY_UNDER_RECORDED in codes
    assert result.verdict == probe.VERDICT_UNUSABLE
    assert result.metrics["graph_replay_coverage"] == pytest.approx(0.125)
    finding = next(f for f in result.findings if f.code == probe.GRAPH_REPLAY_UNDER_RECORDED)
    assert finding.evidence["graph_replays_with_kernels"] == 1
    assert finding.evidence["graph_launch_count"] == 8


def test_annotated_steps_without_gpu_work(tmp_path):
    """Step annotations that enclose no kernel are reported.

    The Kimi-K3 capture declared 128 ``step[DECODE]`` annotations of which 0
    contained a kernel; MiniMax's steady-state chunk had 32 annotations, 640
    runtime launches, and 0 kernels.
    """
    events = [_step(i * 1000.0, 900.0, i) for i in range(8)]
    events += [_launch("hipLaunchKernel", i * 1000.0 + 1, 500 + i) for i in range(8)]
    # One kernel, far outside every annotated step.
    events.append(_kernel("stray", 90_000.0, 5.0, corr=999))
    result = probe.probe_file(_write(tmp_path, events))
    codes = {f.code for f in result.findings}
    assert probe.ANNOTATED_STEPS_WITHOUT_GPU_WORK in codes
    assert result.metrics["annotated_steps"] == 8
    assert result.metrics["annotated_steps_with_gpu_work"] == 0


def test_kernel_launch_ratio_collapse(tmp_path):
    """Runtime launches with no kernels at all fire the ratio check."""
    events = [_launch("hipLaunchKernel", float(i), 100 + i) for i in range(200)]
    events.append({"cat": "cpu_op", "ph": "X", "name": "aten::mm", "ts": 0.0, "dur": 1.0, "args": {}})
    result = probe.probe_file(_write(tmp_path, events))
    codes = {f.code for f in result.findings}
    assert probe.KERNEL_LAUNCH_RATIO_COLLAPSED in codes
    assert result.metrics["kernel_per_launch_ratio"] == 0.0


def test_eager_trace_ratio_near_one_is_clean(tmp_path):
    """An eager capture sits near 1.0 kernels per launch and must not fire.

    The ratio check is deliberately pinned near zero so it separates "device
    side missing" from the normal eager and graph-replay regimes, which differ
    from each other by orders of magnitude.
    """
    events: list[dict] = []
    for i in range(50):
        events.append(_launch("hipLaunchKernel", i * 10.0, 200 + i))
        events.append(_kernel(f"k{i}", i * 10.0 + 1, 5.0, corr=200 + i))
    result = probe.probe_file(_write(tmp_path, events))
    assert probe.KERNEL_LAUNCH_RATIO_COLLAPSED not in {f.code for f in result.findings}
    assert result.metrics["kernel_per_launch_ratio"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Timeline exclusivity
# --------------------------------------------------------------------------- #


def test_rank_skew_barrier_dominates_window(tmp_path):
    """One collective covering most of the window is reported.

    Measured at 73.9-97.2% of the window on six of six reference runs; the
    barrier's duration equalled the rank's profiler-start skew, so it carried
    no communication content at all.
    """
    events = [
        _kernel("_vocab_parallel_embedding_kernel", 0.0, 3.0, corr=1),
        _kernel("cross_device_reduce_2stage", 4.0, 8_000.0, corr=2),
    ]
    events += [_kernel(f"gemm_{i}", 8_100.0 + i * 10, 8.0, corr=10 + i) for i in range(60)]
    result = probe.probe_file(_write(tmp_path, events))
    codes = {f.code for f in result.findings}
    assert probe.SINGLE_EVENT_DOMINATES_WINDOW in codes
    finding = next(f for f in result.findings if f.code == probe.SINGLE_EVENT_DOMINATES_WINDOW)
    assert "cross_device_reduce_2stage" in finding.evidence["single_event_name"]
    assert finding.evidence["single_event_window_share"] > 0.9


def test_clean_window_has_no_dominant_event(tmp_path):
    """An evenly-filled decode window keeps its largest kernel well under the gate."""
    events = [_kernel(f"gemm_{i}", i * 100.0, 90.0, corr=i) for i in range(100)]
    result = probe.probe_file(_write(tmp_path, events))
    assert probe.SINGLE_EVENT_DOMINATES_WINDOW not in {f.code for f in result.findings}
    assert result.metrics["single_event_window_share"] < 0.02


def test_single_event_threshold_is_env_tunable(tmp_path, monkeypatch):
    """The share ceiling honours its env override.

    The defaults separate a seven-session sample on one framework and platform;
    an operator meeting a different regime has to be able to move them without
    a code change.
    """
    # 200 us of collective inside a 1000 us window: 20%, under the 30% default.
    # The remaining 800 us is spread over eight kernels so the collective stays
    # the single largest event.
    events = [_kernel("long_collective", 0.0, 200.0, corr=1)]
    events += [_kernel(f"work_{i}", 200.0 + i * 100.0, 100.0, corr=2 + i) for i in range(8)]
    path = _write(tmp_path, events)
    assert probe.probe_file(path).metrics["single_event_window_share"] == pytest.approx(0.2)
    assert probe.SINGLE_EVENT_DOMINATES_WINDOW not in {f.code for f in probe.probe_file(path).findings}
    monkeypatch.setenv(probe.SINGLE_EVENT_WINDOW_SHARE_MAX_ENV, "0.15")
    assert probe.SINGLE_EVENT_DOMINATES_WINDOW in {f.code for f in probe.probe_file(path).findings}


# --------------------------------------------------------------------------- #
# Cross-rank consistency
# --------------------------------------------------------------------------- #


def test_rank_kernel_count_imbalance(tmp_path):
    """An 87x kernel-count spread across lockstep ranks is reported.

    Reference: Qwen3.8 run1 recorded 3,594 kernels on TP-6 and 315,008 on
    TP-7, from the same profile of the same lockstep workload.
    """
    # 8 steps x 32 kernels = 256 recorded, against the starved rank's 64.
    good = _write(tmp_path, healthy_graph_trace(steps=8, kernels_per_step=32), "rank_1.trace.json.gz")
    starved = _write(tmp_path, under_recorded_graph_trace(steps=8), "rank_0.trace.json.gz")
    result = probe.probe_paths([starved, good])
    codes = {f.code for f in result.findings}
    assert probe.RANK_KERNEL_COUNT_IMBALANCE in codes
    finding = next(f for f in result.findings if f.code == probe.RANK_KERNEL_COUNT_IMBALANCE)
    assert finding.evidence["rank_kernel_spread"] > 2.0


def test_matched_ranks_produce_no_cross_rank_finding(tmp_path):
    """Two identically-recorded ranks are consistent."""
    a = _write(tmp_path, healthy_graph_trace(), "rank_0.trace.json.gz")
    b = _write(tmp_path, healthy_graph_trace(), "rank_1.trace.json.gz")
    result = probe.probe_paths([a, b])
    assert result.verdict == probe.VERDICT_USABLE
    assert result.metrics["rank_kernel_spread"] == pytest.approx(1.0)


def test_profiler_start_skew(tmp_path):
    """Ranks opening far apart are reported, keyed on the skew/span ratio.

    Reference: 30.88 s of start skew against a ~21 s median window and 0.5 s of
    stop skew -- the ranks decided to start together and did not.
    """
    early = _write(tmp_path, [_kernel(f"k{i}", i * 10.0, 5.0, corr=i) for i in range(50)], "rank_0.trace.json.gz")
    late_events = [_kernel(f"k{i}", 100_000.0 + i * 10.0, 5.0, corr=i) for i in range(50)]
    late = _write(tmp_path, late_events, "rank_1.trace.json.gz")
    result = probe.probe_paths([early, late])
    codes = {f.code for f in result.findings}
    assert probe.RANK_PROFILER_START_SKEW in codes
    finding = next(f for f in result.findings if f.code == probe.RANK_PROFILER_START_SKEW)
    assert finding.evidence["rank_start_skew_share"] > 0.10


def test_single_rank_skips_cross_rank_checks(tmp_path):
    """One rank cannot disagree with itself."""
    result = probe.probe_rank_set({"rank_0": probe.probe_file(_write(tmp_path, healthy_graph_trace()))})
    assert result.findings == []
    assert result.metrics["rank_count"] == 1


# --------------------------------------------------------------------------- #
# Structural checks have no discriminating power (the premise of this module)
# --------------------------------------------------------------------------- #


def test_under_recorded_trace_passes_every_structural_check(tmp_path):
    """The broken capture is readable, parseable, and holds GPU kernels.

    This is the finding the module exists for: on the reference sample the
    structural checks -- file opens, ``stream_errors`` empty, kernel events
    present, ``cpu_op`` counts matching across ranks -- passed on every trace
    that went on to break the KERNEL phase.
    """
    events = under_recorded_graph_trace(steps=8)
    events += [{"cat": "cpu_op", "ph": "X", "name": "aten::mm", "ts": float(i), "dur": 1.0} for i in range(500)]
    result = probe.probe_file(_write(tmp_path, events))
    assert result.metrics["stream_errors"] == 0
    assert result.metrics["kernel_events"] > 0
    assert result.metrics["cpu_op_events"] == 500
    # ... and yet:
    assert result.verdict == probe.VERDICT_UNUSABLE


# --------------------------------------------------------------------------- #
# Contract: never raise, never escalate
# --------------------------------------------------------------------------- #


def test_missing_file_reports_rather_than_raises(tmp_path):
    """An unreadable trace comes back as a finding, not an exception."""
    result = probe.probe_file(tmp_path / "nope.trace.json.gz")
    assert [f.code for f in result.findings] == [probe.TRACE_PROBE_UNREADABLE]
    assert result.verdict == probe.VERDICT_DEGRADED


def test_truncated_trace_reports_rather_than_raises(tmp_path):
    """A half-written trace is recoverable up to the truncation point."""
    path = tmp_path / "cut.trace.json.gz"
    body = json.dumps({"traceEvents": healthy_graph_trace()})
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(body[: len(body) // 2])
    result = probe.probe_file(path)
    assert probe.TRACE_PROBE_UNREADABLE not in {f.code for f in result.findings}
    assert result.metrics["stream_errors"] >= 1


def test_observation_only_caps_blocking_at_warning(tmp_path):
    """Blocking findings are emitted as warnings until the thresholds are proven."""
    result = probe.probe_file(_write(tmp_path, under_recorded_graph_trace()))
    assert result.verdict == probe.VERDICT_UNUSABLE
    rows = result.to_health_warnings()
    assert rows, "expected at least one warning row"
    assert all(r["severity"] != probe.SEVERITY_BLOCKING for r in rows)
    assert any(r.get("probe_severity") == probe.SEVERITY_BLOCKING for r in rows)
    raw = result.to_health_warnings(observation_only=False)
    assert any(r["severity"] == probe.SEVERITY_BLOCKING for r in raw)


def test_health_warning_rows_carry_known_codes(tmp_path):
    """Every emitted code is in the module's allow-list."""
    result = probe.probe_file(_write(tmp_path, under_recorded_graph_trace()))
    for row in result.to_health_warnings():
        assert row["code"] in probe.KNOWN_CODES
        assert row["message"]


def test_max_events_cap_marks_result_truncated(tmp_path):
    """A capped probe still answers, and says that it was capped."""
    result = probe.probe_file(_write(tmp_path, healthy_graph_trace(steps=8)), max_events=5)
    assert result.metrics["truncated"] is True


def test_probe_enabled_env_switch(monkeypatch):
    """The master switch defaults on and honours falsey values."""
    monkeypatch.delenv(probe.ENABLED_ENV, raising=False)
    assert probe.probe_enabled() is True
    monkeypatch.setenv(probe.ENABLED_ENV, "0")
    assert probe.probe_enabled() is False
    monkeypatch.setenv(probe.ENABLED_ENV, "1")
    assert probe.probe_enabled() is True


def test_summary_line_names_verdict_and_findings(tmp_path):
    """The log line carries the verdict, the codes, and the measurements."""
    line = probe.probe_file(_write(tmp_path, under_recorded_graph_trace())).summary_line()
    assert probe.VERDICT_UNUSABLE in line
    assert probe.GRAPH_REPLAY_UNDER_RECORDED in line
    assert "graph_replay_coverage=" in line
