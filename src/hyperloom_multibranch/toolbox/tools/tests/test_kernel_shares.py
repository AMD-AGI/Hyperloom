# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kernel shares of a report window come from its raw GPU events: events that break a stream's order are left out.

``data/*_gpu_events.json.gz`` are the GPU events (kernels and copies, unaltered) of real windows: the Qwen3-14B-FP8 branch
b40 decode step (one graph replay of 604 kernels, and six eager kernels the ROCm profiler stamps as 9 ms each) with the rank
trace it was cut from, its sibling whose replay lost its kernels, a Qwen3-14B-FP8 root prefill step, and slices of the
Qwen3-8B and Qwen3-8B TP2 windows.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools.kernel_shares import (
    JITTER_US,
    MAX_EXCLUDED_SHARE,
    MIN_GPU_BUSY_PERCENT,
    GpuEvent,
    KernelTraceError,
    kernel_label,
    kernel_window,
    read_gpu_events,
    split_in_order,
    write_kernel_table,
)

DATA = Path(__file__).parent / "data"
GEMM = "kernel_gemm_0"
QUANT = "aiter::dynamic_per_token_scaled_quant_kernel"
PAGED_ATTENTION = "paged_attention_ll4mi_QKV_mfma16_kernel"
CK_GEMM = "ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle"
TENSILE_GEMM = "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs"


def _real(name: str) -> list[GpuEvent]:
    return read_gpu_events(DATA / f"{name}_gpu_events.json.gz")


def _event(start: float, duration: float, name: str = "k", stream: str = "3") -> GpuEvent:
    return GpuEvent(("4", stream), start, start + duration, name)


def test_the_b40_decode_step_ranks_the_graph_kernels_and_leaves_the_six_misreported_eager_kernels_out() -> None:
    """Before: index_put 29%, copy_ 29%, greedy_sample 15%, add 15% (the brief of every head profiled after fp8 KV)."""
    window = kernel_window(_real("b40_window"), lambda: _real("b40_rank_trace"))
    assert window.degraded_reason == ""
    assert (window.excluded_events, round(window.excluded_claimed_ms, 2)) == (6, 54.84)
    assert round(window.kernel_ms, 2) == 7.93 and round(window.busy_percent, 1) == 74.4
    assert 0 < window.excluded_estimated_ms < 0.03 and window.unbounded == ()  # 23 us of 7.93 ms
    assert sum(window.shares.values()) == pytest.approx(100.0)
    assert list(window.shares)[:3] == [GEMM, QUANT, PAGED_ATTENTION]
    assert window.shares[GEMM] == pytest.approx(44.1, abs=0.05)
    assert window.shares[PAGED_ATTENTION] == pytest.approx(10.6, abs=0.05)
    eager = {
        name: share for name, share in window.shares.items() if name.startswith("at::native::") or "greedy" in name
    }
    assert "aiter::greedy_sample_kernel" not in window.shares and sum(eager.values()) < 1.5  # was 87% with the six
    assert window.rows[0].count == 160  # 4 FlyDSL GEMMs x 40 layers, one name


def test_without_a_clean_instance_of_an_excluded_kernel_in_the_trace_the_window_is_degraded_not_guessed() -> None:
    """The window alone holds no undisturbed ``greedy_sample``: what it costs is unknown, so any ranking is a guess."""
    window = kernel_window(_real("b40_window"), lambda: _real("b40_window"))
    assert "aiter::greedy_sample_kernel" in window.unbounded
    reason = window.degraded_reason
    assert "6 GPU events start before the previous event of their stream ended (they claim 54.84 ms)" in reason
    assert "no undisturbed instance of " in reason and "aiter::greedy_sample_kernel" in reason


def test_a_window_whose_graph_replay_kernels_are_missing_is_degraded_with_the_span_it_fills() -> None:
    """The sibling of b40: its one replay recorded no kernels, so 11 eager kernels (0.05 ms) sit in an 8.2 ms span."""
    window = kernel_window(_real("b06_window"), list)
    assert round(window.busy_percent, 2) == 0.61 and window.excluded_events == 0
    assert "fill 0.6% of its 8.22 ms GPU span (under 10%)" in window.degraded_reason
    assert "CUDA-graph replays are missing" in window.degraded_reason


def test_a_gemm_that_starts_under_a_microsecond_before_its_predecessor_ended_stays_in_the_shares() -> None:
    """A Qwen3-14B-FP8 root prefill window: a 5.4 ms GEMM starts 0.63 us early, as real clocks do; GEMM leads at 87%."""
    events = _real("root_window")
    assert sum(1 for a, b in zip(events, events[1:]) if b.start < a.end) >= 1
    window = kernel_window(events, list)
    assert window.excluded_events == 0 and window.degraded_reason == ""
    assert list(window.shares)[0] == CK_GEMM and window.shares[CK_GEMM] == pytest.approx(87.1, abs=0.05)
    assert sum(window.shares.values()) == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("report", "top", "share", "second", "busy"),
    [
        # Eager-mode Qwen3-8B windows: every hipBLASLt tile is a kernel of its own name, so the GEMM family is the sum of them.
        ("qwen3_8b", "at::native::vectorized_elementwise_kernel", 37.3, TENSILE_GEMM, 31.3),
        ("qwen3_8b_tp2", TENSILE_GEMM, 28.6, "aiter::cross_device_reduce_1stage", 26.6),
    ],
)
def test_the_qwen3_8b_windows_keep_their_collectives_and_one_gemm_family_in_the_shares(
    report: str, top: str, share: float, second: str, busy: float
) -> None:
    window = kernel_window(_real(f"{report}_window"), list)
    assert window.excluded_events == 0 and window.degraded_reason == ""
    assert list(window.shares)[:2] == [top, second]
    assert window.shares[top] == pytest.approx(share, abs=0.05) and window.busy_percent == pytest.approx(busy, abs=0.05)
    assert sum(window.shares.values()) == pytest.approx(100.0)


def test_the_clocks_of_real_windows_overlap_kernels_far_less_than_the_jitter_tolerance_and_the_defect_far_more() -> (
    None
):
    """The largest overlap of neighbours on a stream: 0.63 us in the clean windows, the six eager events' 8 ms in b40."""

    def overlap(events: list[GpuEvent]) -> float:
        by_stream: dict[tuple[str, str], list[GpuEvent]] = {}
        for event in sorted(events, key=lambda e: e.start):
            by_stream.setdefault(event.stream, []).append(event)
        return max(a.end - b.start for run in by_stream.values() for a, b in zip(run, run[1:]))

    clean = [overlap(_real(name)) for name in ("root_window", "qwen3_8b_window", "qwen3_8b_tp2_window")]
    assert 0 < max(clean) < 1.0 < JITTER_US < 100.0 < overlap(_real("b40_window"))
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


def test_every_kernel_of_the_real_windows_gets_a_short_label() -> None:
    names = {
        event.name for window in ("b40_rank_trace", "root_window", "qwen3_8b_tp2_window") for event in _real(window)
    }
    labels = {name: kernel_label(name) for name in names}
    assert all(
        0 < len(label) <= 120 and "<" not in label and "(" not in label.replace("(anonymous namespace)", "")
        for label in labels.values()
    )
    assert len(set(labels.values())) < len(names)  # template variants share a function


def test_the_table_lists_every_kernel_largest_first_with_its_share_time_and_count(tmp_path: Path) -> None:
    window = kernel_window(_real("b40_window"), lambda: _real("b40_rank_trace"))
    path = tmp_path / "kernel_shares.csv"
    write_kernel_table(window, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "name,Percentage (%),total_time_ms,count" and lines[1].startswith(f"{GEMM},44.1")
    assert len(lines) == 1 + len(window.rows)


@pytest.mark.parametrize("content", [b"not gzip", gzip.compress(b"{"), gzip.compress(b'{"traceEvents": 3}')])
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
