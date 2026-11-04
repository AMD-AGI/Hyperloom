# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kernel shares of a report window come from its raw GPU events: events that break a stream's order are left out.

``data/*_gpu_events.json.gz`` are synthetic windows that ``data/_make_traces.py`` writes, with the structure the ROCm
profiler leaves: a branch's decode step (one graph replay of 64 kernels, and six eager kernels queued behind it that
the profiler stamps as starting inside the replay and ending at the window's end) with the rank trace it was cut from,
a sibling whose replay lost its kernels, the root's prefill step, and two eager-mode windows, one of them TP2. The
numbers asserted are those the generator's durations give.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools.kernel_shares import (
    COLLECTIVE,
    JITTER_US,
    MAX_COLLECTIVE_STALL,
    MAX_EXCLUDED_SHARE,
    MIN_COLLECTIVE_CALLS,
    MIN_GPU_BUSY_PERCENT,
    GpuEvent,
    KernelTraceError,
    kernel_label,
    kernel_window,
    read_gpu_events,
    split_in_order,
    stalled_collective,
    write_kernel_table,
)

DATA = Path(__file__).parent / "data"
GEMM = "kernel_gemm_0"
QUANT = "aiter::dynamic_per_token_scaled_quant_kernel"
PAGED_ATTENTION = "paged_attention_ll4mi_QKV_mfma16_kernel"
CK_GEMM = "ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle"
TENSILE_GEMM = "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs"
CROSS_REDUCE = "aiter::cross_device_reduce_1stage"


def _trace(name: str) -> list[GpuEvent]:
    return read_gpu_events(DATA / f"{name}_gpu_events.json.gz")


def _event(start: float, duration: float, name: str = "k", stream: str = "3") -> GpuEvent:
    return GpuEvent(("4", stream), start, start + duration, name)


def test_the_decode_step_ranks_the_graph_kernels_and_leaves_the_six_misreported_eager_kernels_out() -> None:
    """The six eager kernels claim 2.59 ms of a step whose kernels run 0.565 ms: taken at their word, index_put, copy_,
    greedy_sample and add would lead the shares. Left out, the replay's kernels lead and the eager ones are a remainder;
    what the six really cost (each at the median of its undisturbed instances in the rank trace) is far under the cap."""
    window = kernel_window(_trace("decode_window"), lambda: _trace("decode_rank_trace"))
    assert window.degraded_reason == ""
    assert (window.excluded_events, round(window.excluded_claimed_ms, 2)) == (6, 2.59)
    assert round(window.kernel_ms, 3) == 0.565 and round(window.busy_percent, 1) == 77.4
    assert window.excluded_estimated_ms == pytest.approx(0.018) and window.unbounded == ()  # 6 + 3 + 2 + 2 + 2 + 3 us
    assert window.excluded_estimated_ms < MAX_EXCLUDED_SHARE * window.kernel_ms
    assert sum(window.shares.values()) == pytest.approx(100.0)
    assert list(window.shares)[:3] == [GEMM, QUANT, PAGED_ATTENTION]
    assert window.shares[GEMM] == pytest.approx(35.32, abs=0.01)
    assert window.shares[PAGED_ATTENTION] == pytest.approx(8.15, abs=0.01)
    eager = {
        name: share for name, share in window.shares.items() if name.startswith("at::native::") or "greedy" in name
    }
    assert "aiter::greedy_sample_kernel" not in window.shares and sum(eager.values()) < 2
    assert window.rows[0].count == 16  # 4 FlyDSL GEMMs x 4 layers, one name


def test_without_a_clean_instance_of_an_excluded_kernel_in_the_trace_the_window_is_degraded_not_guessed() -> None:
    """The window alone holds no undisturbed ``greedy_sample``: what it costs is unknown, so any ranking is a guess."""
    window = kernel_window(_trace("decode_window"), lambda: _trace("decode_window"))
    assert "aiter::greedy_sample_kernel" in window.unbounded
    reason = window.degraded_reason
    assert "6 GPU events start before the previous event of their stream ended (they claim 2.59 ms)" in reason
    assert "no undisturbed instance of " in reason and "aiter::greedy_sample_kernel" in reason


def test_a_window_whose_graph_replay_kernels_are_missing_is_degraded_with_the_span_it_fills() -> None:
    """The decode step's sibling: its one replay recorded no kernels, so 10 eager events (27 us) sit in a 5 ms span."""
    window = kernel_window(_trace("lost_replay_window"), list)
    assert round(window.busy_percent, 2) == 0.54 and window.excluded_events == 0
    assert "fill 0.5% of its 5.00 ms GPU span (under 10%)" in window.degraded_reason
    assert "CUDA-graph replays are missing" in window.degraded_reason


def test_a_gemm_that_starts_under_a_microsecond_before_its_predecessor_ended_stays_in_the_shares() -> None:
    """The root's prefill window: a GEMM starts 0.4 us before its predecessor ends, as real clocks do; the GEMMs are 80%
    of its kernel time."""
    events = _trace("root_window")
    assert sum(1 for a, b in zip(events, events[1:]) if b.start < a.end) >= 1
    window = kernel_window(events, list)
    assert window.excluded_events == 0 and window.degraded_reason == ""
    assert list(window.shares)[0] == CK_GEMM and window.shares[CK_GEMM] == pytest.approx(80.0)
    assert sum(window.shares.values()) == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("report", "top", "share", "second", "busy"),
    [
        # Eager-mode windows: every hipBLASLt tile is a kernel of its own name, so the GEMM family is the sum of them.
        ("eager", "at::native::vectorized_elementwise_kernel", 38.9, TENSILE_GEMM, 34.8),
        ("eager_tp2", TENSILE_GEMM, 34.8, CROSS_REDUCE, 27.2),
    ],
)
def test_the_eager_windows_keep_their_collectives_and_one_gemm_family_in_the_shares(
    report: str, top: str, share: float, second: str, busy: float
) -> None:
    window = kernel_window(_trace(f"{report}_window"), list)
    assert window.excluded_events == 0 and window.degraded_reason == ""
    assert list(window.shares)[:2] == [top, second]
    assert window.shares[top] == pytest.approx(share, abs=0.05) and window.busy_percent == pytest.approx(busy, abs=0.05)
    assert sum(window.shares.values()) == pytest.approx(100.0)


def test_the_clocks_of_clean_windows_overlap_kernels_far_less_than_the_jitter_tolerance_and_the_defect_far_more() -> (
    None
):
    """The largest overlap of neighbours on a stream: under a microsecond in the clean windows (the clocks' jitter), the
    six eager events' hundreds of microseconds in the decode step (the profiler's defect)."""

    def overlap(events: list[GpuEvent]) -> float:
        by_stream: dict[tuple[str, str], list[GpuEvent]] = {}
        for event in sorted(events, key=lambda e: e.start):
            by_stream.setdefault(event.stream, []).append(event)
        return max(a.end - b.start for run in by_stream.values() for a, b in zip(run, run[1:]))

    clean = [overlap(_trace(name)) for name in ("root_window", "eager_window", "eager_tp2_window")]
    assert 0 < max(clean) < 1.0 < JITTER_US < 100.0 < overlap(_trace("decode_window"))
    assert (JITTER_US, MIN_GPU_BUSY_PERCENT, MAX_EXCLUDED_SHARE) == (5.0, 10.0, 0.05)


def test_an_event_that_overlaps_hundreds_is_the_one_excluded_never_the_hundreds() -> None:
    """Kept in start order the long event would be kept and the 300 kernels under it dropped; by earliest end it is the one."""
    kernels = [_event(10.0 * i, 10.0, f"k{i % 3}") for i in range(300)]
    long = _event(105.0, 9000.0, "long")
    ordered = split_in_order([*kernels, long])
    assert ordered.excluded == [long] and len(ordered.kept) == 300


@pytest.mark.parametrize("short_ends_first", [True, False])
def test_of_two_events_that_overlap_by_a_few_microseconds_the_shorter_stays_whichever_ends_first(
    short_ends_first: bool,
) -> None:
    """A long event that ends a microsecond before a short real one: kept by earliest end it would claim 1000 us for itself."""
    long = _event(0.0, 1000.0, "long")
    short = _event(990.0, 11.0, "short") if short_ends_first else _event(-1.0, 12.0, "short")
    ordered = split_in_order([long, short])
    assert ordered.excluded == [long] and ordered.kept == [short]


@pytest.mark.parametrize(("overlap", "excluded"), [(JITTER_US, 0), (JITTER_US + 0.01, 1)])
def test_an_event_starting_up_to_the_jitter_before_its_predecessor_ended_is_in_order(
    overlap: float, excluded: int
) -> None:
    ordered = split_in_order([_event(0.0, 100.0, "a"), _event(100.0 - overlap, 50.0, "b")])
    assert len(ordered.excluded) == excluded and len(ordered.kept) == 2 - excluded


def test_streams_and_devices_are_in_order_each_on_their_own() -> None:
    events = [_event(0.0, 100.0, "a", "3"), _event(50.0, 100.0, "b", "2"), GpuEvent(("5", "3"), 50.0, 150.0, "c")]
    assert split_in_order(events).excluded == []


def _stream() -> list[GpuEvent]:
    """Ten 100 us kernels one after another: 1 ms of kernel time."""
    return [_event(100.0 * i, 100.0, "work") for i in range(10)]


def _defect() -> GpuEvent:
    """An eager kernel the profiler stamps from inside the stream's work to 100 us after its end."""
    return _event(200.0, 900.0, "eager")


def _elsewhere(*durations: float) -> list[GpuEvent]:
    return [_event(5000.0 + 100.0 * i, duration, "eager") for i, duration in enumerate(durations)]


def test_the_excluded_events_are_priced_at_the_median_of_their_undisturbed_instances() -> None:
    window = kernel_window([*_stream(), _defect()], lambda: [*_stream(), _defect(), *_elsewhere(20.0, 30.0, 25.0)])
    assert window.excluded_events == 1 and window.excluded_claimed_ms == pytest.approx(0.9)
    assert window.excluded_estimated_ms == pytest.approx(0.025) and window.degraded_reason == ""
    assert set(window.shares) == {"work"}


def test_excluded_events_that_cost_over_five_percent_of_the_kernels_degrade_the_window_with_the_estimate() -> None:
    window = kernel_window([*_stream(), _defect()], lambda: [*_stream(), *_elsewhere(80.0)])
    assert "they are estimated at 0.08 ms, over 5% of the other kernels' 1.00 ms" in window.degraded_reason


@pytest.mark.parametrize(("price_us", "degraded"), [(MAX_EXCLUDED_SHARE * 1000.0, False), (51.0, True)])
def test_the_excluded_cost_cap_is_inclusive_of_five_percent_of_the_kept_kernel_time(
    price_us: float, degraded: bool
) -> None:
    window = kernel_window([*_stream(), _defect()], lambda: [*_stream(), *_elsewhere(price_us)])
    assert bool(window.excluded_reason) is degraded


@pytest.mark.parametrize(("fill", "degraded"), [(MIN_GPU_BUSY_PERCENT - 0.01, True), (MIN_GPU_BUSY_PERCENT, False)])
def test_the_busy_threshold_is_inclusive_of_ten_percent_of_the_span(fill: float, degraded: bool) -> None:
    span = 1000.0
    window = kernel_window([_event(0.0, span * fill / 100.0), _event(span, 0.0, "mark", "2")], list)
    assert window.busy_percent == pytest.approx(fill) and bool(window.starved_reason) is degraded


def test_a_window_with_no_gpu_events_is_degraded_not_empty_shares() -> None:
    window = kernel_window([], list)
    assert window.rows == () and "no GPU kernel events" in window.degraded_reason


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        (
            "void aiter::greedy_sample_kernel<float, 1024, 16>(float const*, int*, int, int)",
            "aiter::greedy_sample_kernel",
        ),
        ("_ZN5aiter37dynamic_per_token_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPT_PKfiPKii", QUANT),
        (
            "_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_ES3_S5_EEEEvPS3_PS4_i",
            "sgl_hip::activation::act_and_mul_kernel",
        ),
        (
            "void paged_attention_ll4mi_QKV_mfma16_kernel<0, __hip_bfloat16, (vllm::Fp8KVCacheDataType)1>(int)",
            PAGED_ATTENTION,
        ),
        ("void (anonymous namespace)::kernel<float>(float*)", "(anonymous namespace)::kernel"),
        ("Memcpy DtoD (Device -> Device)", "Memcpy DtoD"),
        ("kernel_gemm_0", GEMM),
        ("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x256_MI16x16x1_SN_LDSB0_AFC0", TENSILE_GEMM),
        ("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x128x128_MI16x16x1_SN_LDSB1", TENSILE_GEMM),
    ],
)
def test_a_kernel_is_named_by_its_function_without_templates_parameters_or_gemm_tile(raw: str, label: str) -> None:
    assert kernel_label(raw) == label


def test_every_kernel_of_the_fixture_windows_gets_a_short_label() -> None:
    names = {
        event.name for window in ("decode_rank_trace", "root_window", "eager_tp2_window") for event in _trace(window)
    }
    labels = {name: kernel_label(name) for name in names}
    assert all(
        0 < len(label) <= 120 and "<" not in label and "(" not in label.replace("(anonymous namespace)", "")
        for label in labels.values()
    )
    assert len(set(labels.values())) < len(names)  # template variants share a function


def test_the_table_lists_every_kernel_largest_first_with_its_share_time_and_count(tmp_path: Path) -> None:
    window = kernel_window(_trace("decode_window"), lambda: _trace("decode_rank_trace"))
    path = tmp_path / "kernel_shares.csv"
    write_kernel_table(window, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "name,Percentage (%),total_time_ms,count" and lines[1].startswith(f"{GEMM},35.31")
    assert len(lines) == 1 + len(window.rows)


@pytest.mark.parametrize(
    "content",
    [b"not gzip", gzip.compress(b"{", mtime=0), gzip.compress(b'{"traceEvents": 3}', mtime=0)],
    ids=["not-gzip", "truncated-json", "events-not-a-list"],
)
def test_a_trace_that_cannot_be_read_as_events_is_an_error_not_an_empty_window(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "w.json.gz"
    path.write_bytes(content)
    with pytest.raises(KernelTraceError):
        read_gpu_events(path)


def test_only_complete_gpu_events_with_finite_times_are_read(tmp_path: Path) -> None:
    good = {"ph": "X", "cat": "kernel", "name": "k", "pid": 4, "ts": 1.0, "dur": 2.0, "args": {"stream": 3}}
    events = [
        good,
        {**good, "cat": "cpu_op"},
        {**good, "ph": "B"},
        {**good, "dur": -1.0},
        {**good, "ts": "x"},
        {**good, "name": None},
        {**good, "dur": True},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD (Device -> Device)", "pid": 4, "ts": 4.0, "dur": 1.0},
        "not an event",
    ]
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    assert [event.name for event in read_gpu_events(path)] == ["k", "Memcpy DtoD (Device -> Device)"]


AR = "aiter::ar_ll128_unroll2"


def _decode_steps(stall_us: float, steps: int = 32, calls: int = 96) -> list[GpuEvent]:
    """``steps`` decode steps on one stream: a GEMM and ``calls`` all-reduces of 10 us each, the first all-reduce of
    each step lasting ``stall_us`` instead (what a rank held up by its peer records)."""
    events, at = [], 0.0
    for _ in range(steps):
        for call in range(calls):
            events.append(_event(at, 20.0, GEMM))
            duration = stall_us if call == 0 else 10.0
            events.append(_event(at + 20.0, duration, f"void {AR}<float>(int)"))
            at += 20.0 + duration
    return events


def test_a_window_whose_all_reduce_waits_on_a_held_up_rank_is_degraded_naming_the_collective() -> None:
    window = kernel_window(_decode_steps(stall_us=1_100_000.0), list)

    assert window.degraded_reason == window.stalled_reason
    assert window.stalled_reason.startswith(f"the collective {AR} lasts ")
    assert f"over {MAX_COLLECTIVE_STALL:g}x" in window.stalled_reason
    assert window.shares[AR] > 99.0, "the shares alone would have read as a 99% all-reduce ceiling"


def test_the_same_window_with_its_all_reduces_at_their_usual_length_is_not_degraded() -> None:
    assert kernel_window(_decode_steps(stall_us=10.0), list).degraded_reason == ""


@pytest.mark.parametrize(
    ("ratio", "stalled"), [(MAX_COLLECTIVE_STALL * 0.99, False), (MAX_COLLECTIVE_STALL * 1.01, True)]
)
def test_the_stall_threshold_is_the_mean_over_the_median_of_one_collectives_calls(ratio: float, stalled: bool) -> None:
    calls = 100
    long = 10.0 * (ratio * calls - (calls - 1))
    events = [_event(30.0 * i, long if i == 0 else 10.0, AR) for i in range(calls)]

    assert bool(stalled_collective(events)) is stalled


def test_a_collective_with_too_few_calls_to_have_a_median_and_a_slow_gemm_are_no_stall() -> None:
    few = [_event(30.0 * i, 10_000.0 if i == 0 else 10.0, AR) for i in range(MIN_COLLECTIVE_CALLS - 1)]
    gemm = [_event(30.0 * i, 10_000.0 if i == 0 else 10.0, GEMM) for i in range(100)]

    assert stalled_collective(few) == "" and stalled_collective(gemm) == ""


@pytest.mark.parametrize(
    "name", ["lost_replay_window", "decode_window", "eager_window", "eager_tp2_window", "root_window"]
)
def test_no_window_of_the_suite_is_a_stall(name: str) -> None:
    assert stalled_collective(split_in_order(_trace(name)).kept) == ""


def test_the_tp2_windows_one_shot_all_reduce_is_a_collective_whose_calls_say_it_is_not_a_stall() -> None:
    """The no-stall check above is not empty for the TP2 window: its attention all-reduce has a call per layer, enough
    for a median, each about as long as the others."""
    calls = [e for e in split_in_order(_trace("eager_tp2_window")).kept if COLLECTIVE.search(kernel_label(e.name))]
    assert len(calls) >= MIN_COLLECTIVE_CALLS and max(e.duration for e in calls) < 2 * min(e.duration for e in calls)
