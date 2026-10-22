# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The report's op shares are those of a graded wave: its decode window and its prefill window, weighted by the time each takes.

The real excerpts are those of ``final_probe.py``: a Qwen3-14B-FP8 profile at ISL 1024, OSL 1024, concurrency 64, whose
``mixed`` window is 52% prefill by time while a wave of that workload is 11% prefill.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools.kernel_launchers import Relabelled
from hyperloom_multibranch.toolbox.tools.kernel_shares import GpuEvent, kernel_window
from hyperloom_multibranch.toolbox.tools.op_shares import read_op_shares
from hyperloom_multibranch.toolbox.tools.phase_mix import (
    DECODE,
    EXECUTION_DETAILS,
    PREFILL,
    PhaseWindow,
    StepCounts,
    Workload,
    graded_mix,
    mix_fields,
    rank_trace_reader,
    read_phase_window,
    split_steps,
    wave_ms,
    write_mix_table,
)
from hyperloom_multibranch.toolbox.tools.tests import final_probe as probe

WORKLOAD = Workload(conc=64, isl=1024, osl=1024)
BOTH = (PREFILL, DECODE)


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    """What the splitter leaves for the real excerpts: the two window files and its step list."""
    folder = tmp_path / "trace_split"
    folder.mkdir()
    shutil.copy(probe.DECODE_WINDOW, folder / probe.DECODE_CHUNK)
    shutil.copy(probe.PREFILL_WINDOW, folder / probe.PREFILL_CHUNK)
    shutil.copy(probe.DATA / "final_probe_execution_details.json", folder / EXECUTION_DETAILS)
    return folder


@pytest.fixture
def real(split_dir: Path, tmp_path: Path) -> dict[str, PhaseWindow]:
    capture = tmp_path / "graph_capture_profile"
    probe.install_capture(capture)
    steps = split_steps(split_dir)
    return {
        PREFILL: read_phase_window(PREFILL, split_dir / probe.PREFILL_CHUNK, steps, capture, list),
        DECODE: read_phase_window(DECODE, split_dir / probe.DECODE_CHUNK, steps, capture, list),
    }


def _synthetic(phase: str, steps: int, tokens: int, step_ms: float, kernel: str) -> PhaseWindow:
    """A window of ``steps`` steps that each run one ``kernel`` for ``step_ms``, one after another."""
    events = [GpuEvent(("0", "1"), i * step_ms * 1000, (i + 1) * step_ms * 1000, kernel) for i in range(steps)]
    return PhaseWindow(
        phase, f"{phase}.json.gz", kernel_window(events, list), Relabelled(events, 0, 0, ()), steps, tokens, ""
    )


def test_a_wave_is_conc_times_isl_over_the_prefill_step_size_prefill_steps_and_osl_decode_steps() -> None:
    prefill = _synthetic(PREFILL, steps=5, tokens=65536, step_ms=200.0, kernel="B")
    decode = _synthetic(DECODE, steps=32, tokens=0, step_ms=10.0, kernel="A")

    assert wave_ms(prefill, WORKLOAD) == pytest.approx(5 * 200.0)  # 64 x 1024 / 13107.2 tokens per step = 5 steps
    assert wave_ms(decode, WORKLOAD) == pytest.approx(1024 * 10.0)
    mix = graded_mix([prefill, decode], WORKLOAD)
    assert mix.fractions[PREFILL] == pytest.approx(1000 / (1000 + 10240)) and sum(mix.fractions.values()) == 1.0
    assert {row.name: row.percent for row in mix.rows} == pytest.approx(
        {"A": 100 * 10240 / 11240, "B": 100 * 1000 / 11240}
    )


@pytest.mark.parametrize(
    ("workload", "prefill_ms"),
    [
        (Workload(conc=32, isl=1024, osl=1024), 500.0),  # half the requests, half the prompt tokens
        (Workload(conc=64, isl=512, osl=1024), 500.0),
        (Workload(conc=64, isl=1024, osl=2048), 1000.0),  # the prefill does not depend on the output length
    ],
)
def test_the_fractions_follow_the_workload_not_a_constant(workload: Workload, prefill_ms: float) -> None:
    prefill = _synthetic(PREFILL, steps=5, tokens=65536, step_ms=200.0, kernel="B")
    decode = _synthetic(DECODE, steps=32, tokens=0, step_ms=10.0, kernel="A")

    assert wave_ms(prefill, workload) == pytest.approx(prefill_ms)
    assert wave_ms(decode, workload) == pytest.approx(workload.osl * 10.0)
    shares = graded_mix([prefill, decode], workload).fractions
    assert shares[PREFILL] == pytest.approx(prefill_ms / (prefill_ms + workload.osl * 10.0))


def test_the_real_probes_wave_is_eleven_percent_prefill_and_its_shares_are_near_the_decode_windows(
    real: dict[str, PhaseWindow],
) -> None:
    """Paged attention reads 14.1% of the mixed window and 28.4% of the decode window; a wave has it at 25.2%."""
    mix = graded_mix([real[PREFILL], real[DECODE]], WORKLOAD)

    assert mix.degraded_reason == "" and mix.note == ""
    assert wave_ms(real[PREFILL], WORKLOAD) == pytest.approx(1790.6, abs=0.1)  # 65536 prompt tokens at 27.3 us each
    assert wave_ms(real[DECODE], WORKLOAD) == pytest.approx(13844.1, abs=0.1)  # 1024 steps of 13.52 ms
    assert mix.fractions[PREFILL] == pytest.approx(0.1145, abs=0.0005) and sum(mix.fractions.values()) == 1.0
    top = {row.name: row for row in mix.rows[:5]}
    assert next(iter(top)) == probe.PAGED_ATTENTION
    assert top[probe.PAGED_ATTENTION].percent == pytest.approx(25.18, abs=0.01)
    assert top[probe.PAGED_ATTENTION].percent > 14.13 + 10  # not the mixed window's
    assert all(abs(row.percent - row.by_phase[DECODE]) < 4 for row in top.values())
    merged = f"{probe.CK_GEMM} [Qwen2MLP.MergedColumnParallelLinear]"
    assert (top[merged].by_phase[DECODE], top[merged].by_phase[PREFILL]) == pytest.approx((14.65, 47.72), abs=0.01)
    assert top[merged].percent == pytest.approx(18.44, abs=0.01)  # the prefill's GEMM-heavy 47.7% counts 11%, not 52%
    assert sum(row.percent for row in mix.rows) == pytest.approx(100.0)
    for row in mix.rows:
        assert row.percent == pytest.approx(
            mix.fractions[DECODE] * row.by_phase[DECODE] + mix.fractions[PREFILL] * row.by_phase[PREFILL]
        )


def test_the_table_the_op_shares_are_read_from_holds_the_weighted_shares_and_each_phases_own(
    real: dict[str, PhaseWindow], tmp_path: Path
) -> None:
    mix = graded_mix([real[PREFILL], real[DECODE]], WORKLOAD)
    table = tmp_path / "kernel_shares.csv"
    write_mix_table(mix, table)

    shares = read_op_shares(mix_fields(list(real.values()), mix, table))

    assert next(iter(shares)) == probe.PAGED_ATTENTION and 0 < len(shares) <= 50
    assert shares[probe.PAGED_ATTENTION] == pytest.approx(25.18, abs=0.01)
    assert sum(shares.values()) == pytest.approx(100.0, abs=0.01)
    assert table.read_text(encoding="utf-8").splitlines()[0] == "name,Percentage (%),decode_percent,prefill_percent"


def test_the_result_says_the_weights_and_the_windows_it_used(real: dict[str, PhaseWindow], tmp_path: Path) -> None:
    mix = graded_mix([real[PREFILL], real[DECODE]], WORKLOAD)

    fields = mix_fields([real[PREFILL], real[DECODE]], mix, tmp_path / "t.csv")

    assert fields["phase_fractions"] == {"prefill": 0.1145, "decode": 0.8855}
    prefill, decode = fields["phase_windows"][PREFILL], fields["phase_windows"][DECODE]
    assert (prefill["chunk"], prefill["steps"], prefill["tokens"], prefill["used"]) == (
        probe.PREFILL_CHUNK,
        2,
        32768,
        True,
    )
    assert (decode["chunk"], decode["steps"], decode["tokens"], decode["used"]) == (probe.DECODE_CHUNK, 4, 0, True)
    assert (decode["step_ms"], prefill["step_ms"]) == (13.52, 447.661)
    assert fields["kernel_shares_path"] == str(tmp_path / "t.csv") and fields["window_degraded_reason"] == ""
    assert (fields["excluded_events"], fields["phase_note"]) == (0, "")


def test_a_window_the_splitter_wrote_none_for_leaves_the_other_alone_and_the_result_says_so(
    real: dict[str, PhaseWindow], split_dir: Path
) -> None:
    gone = read_phase_window(PREFILL, None, {}, None, list)
    mix = graded_mix([gone, real[DECODE]], WORKLOAD)

    assert dict(mix.fractions) == {PREFILL: 0.0, DECODE: 1.0}
    assert {row.name: row.percent for row in mix.rows} == pytest.approx(real[DECODE].kernels.shares)  # type: ignore[union-attr]
    assert "the decode window stands for the whole wave" in mix.note
    assert "the splitter wrote no prefilldecode steady-state window" in mix.note
    assert mix_fields([gone, real[DECODE]], mix, split_dir / "t.csv")["phase_windows"][PREFILL]["used"] is False


def test_a_window_whose_kernels_are_missing_from_the_trace_leaves_the_other_alone(
    real: dict[str, PhaseWindow], split_dir: Path
) -> None:
    starved = split_dir / probe.DECODE_CHUNK
    shutil.copy(probe.DATA / "b06_window_gpu_events.json.gz", starved)  # the replay lost its kernels: 0.6% of the span
    steps = split_steps(split_dir)
    decode = read_phase_window(DECODE, starved, steps, None, list)

    mix = graded_mix([real[PREFILL], decode], WORKLOAD)

    assert not decode.usable and "0.6% of its 8.22 ms GPU span" in decode.reason
    assert dict(mix.fractions) == {PREFILL: 1.0, DECODE: 0.0}
    assert "the prefill window stands for the whole wave" in mix.note and "0.6% of its 8.22 ms" in mix.note
    assert {row.name: row.percent for row in mix.rows} == pytest.approx(real[PREFILL].kernels.shares)  # type: ignore[union-attr]


def test_with_neither_window_the_shares_are_degraded_and_the_reason_names_both(split_dir: Path) -> None:
    gone = read_phase_window(PREFILL, None, {}, None, list)
    starved = split_dir / probe.DECODE_CHUNK
    shutil.copy(probe.DATA / "b06_window_gpu_events.json.gz", starved)
    decode = read_phase_window(DECODE, starved, split_steps(split_dir), None, list)

    mix = graded_mix([gone, decode], WORKLOAD)

    assert mix.rows == () and dict(mix.fractions) == {}
    assert "the prefill window: the splitter wrote no prefilldecode" in mix.degraded_reason
    assert "the decode window: the window's recorded GPU kernels fill 0.6%" in mix.degraded_reason
    fields = mix_fields([gone, decode], mix, split_dir / "t.csv")
    assert fields["kernel_shares_path"] == "" and fields["window_degraded_reason"] == mix.degraded_reason
    assert fields["phase_fractions"] == {} and fields["excluded_events"] == 0


@pytest.mark.parametrize(
    ("steps", "reason"),
    [
        ({}, "lists no steps for"),
        ({probe.PREFILL_CHUNK: StepCounts(2, 0)}, "lists no steps for " + probe.DECODE_CHUNK),
    ],
)
def test_a_window_without_steps_in_the_splitters_list_cannot_be_weighted_so_it_is_not_used(
    split_dir: Path, steps: dict[str, StepCounts], reason: str
) -> None:
    decode = read_phase_window(DECODE, split_dir / probe.DECODE_CHUNK, steps, None, list)

    assert not decode.usable and reason in decode.reason


def test_a_prefill_window_with_no_prompt_tokens_in_its_steps_is_not_used(split_dir: Path) -> None:
    prefill = read_phase_window(
        PREFILL, split_dir / probe.PREFILL_CHUNK, {probe.PREFILL_CHUNK: StepCounts(2, 0)}, None, list
    )

    assert not prefill.usable and "lists no prompt tokens in the steps of" in prefill.reason


def test_two_windows_cannot_be_weighted_without_the_workload_one_window_needs_none(
    real: dict[str, PhaseWindow],
) -> None:
    both = graded_mix([real[PREFILL], real[DECODE]], None)
    gone = read_phase_window(PREFILL, None, {}, None, list)
    alone = graded_mix([gone, real[DECODE]], None)

    assert both.rows == () and "split_conc, split_isl and split_osl" in both.degraded_reason
    assert alone.degraded_reason == "" and alone.rows


def test_a_workload_whose_wave_time_overflows_is_degraded_not_nan(real: dict[str, PhaseWindow]) -> None:
    mix = graded_mix([real[PREFILL], real[DECODE]], Workload(conc=1e308, isl=1e308, osl=1e308))

    assert mix.rows == () and dict(mix.fractions) == {} and "not a finite number" in mix.degraded_reason


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        ({"split_conc": "64", "split_isl": "1024", "split_osl": "1024"}, Workload(64, 1024, 1024)),
        ({"split_conc": "64", "split_isl": "1024", "split_osl": "128.5", "split_r": "1"}, Workload(64, 1024, 128.5)),
        ({"split_conc": "64", "split_osl": "1024"}, None),
        ({}, None),
        ({"split_conc": "64", "split_isl": "0", "split_osl": "1024"}, None),
        ({"split_conc": "many", "split_isl": "1024", "split_osl": "1024"}, None),
        ({"split_conc": "64", "split_isl": "nan", "split_osl": "1024"}, None),
    ],
)
def test_the_workload_is_the_requests_conc_isl_and_osl_or_nothing(
    hints: dict[str, str], expected: Workload | None
) -> None:
    assert Workload.from_hints(hints) == expected


@pytest.mark.parametrize(
    "content", ["", "{", "{}", '[{"output_path": 7}]', '[{"output_path": "a", "steps": "x"}]', "[1]"]
)
def test_a_step_list_the_splitter_did_not_write_gives_no_steps(tmp_path: Path, content: str) -> None:
    (tmp_path / EXECUTION_DETAILS).write_text(content, encoding="utf-8")

    assert split_steps(tmp_path) == {}


def test_the_steps_of_a_window_are_its_listed_steps_and_the_prompt_tokens_of_them(split_dir: Path) -> None:
    assert split_steps(split_dir) == {
        probe.DECODE_CHUNK: StepCounts(4, 0),
        probe.PREFILL_CHUNK: StepCounts(2, 32768),
    }
    assert split_steps(split_dir / "none") == {}


def test_the_rank_trace_is_read_once_however_many_windows_price_excluded_events_from_it(tmp_path: Path) -> None:
    trace = tmp_path / "t.trace.json.gz"
    event = {"ph": "X", "cat": "kernel", "ts": 1.0, "dur": 2.0, "name": "k", "pid": 1, "args": {"stream": 3}}
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": [event]}, handle)
    read = rank_trace_reader(trace)

    first = read()
    trace.unlink()

    assert len(first) == 1 and read() is first
