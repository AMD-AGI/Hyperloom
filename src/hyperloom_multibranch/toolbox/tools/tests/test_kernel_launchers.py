# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A kernel name that several modules launch is split by the module, as the trace records it; otherwise the name stays.

``data/*_launch_events.json.gz`` are synthetic windows (``data/_make_traces.py``) with what launched their kernels: a
branch's decode step (one graph replay of 64 kernels, four layers of a made-up decoder, and the eager sampling after it)
and the root's prefill step (eager kernels, two layers). ``decode_capture_bs64_launches.json.gz`` is the graph capture of
batch size 64 the replay replays: its kernel launches and the ``nn.Module`` frames that made them, which the replay's
kernels do not carry.
"""

from __future__ import annotations

import gzip
import json
import shutil
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools.kernel_launchers import (
    CAPTURE_MANIFEST,
    Replay,
    TraceWindow,
    read_launched_window,
    read_launches,
    read_window,
    relabel_replays,
)
from hyperloom_multibranch.toolbox.tools.kernel_shares import GpuEvent, kernel_window, read_gpu_events

DATA = Path(__file__).parent / "data"
GEMM = "kernel_gemm_0"
CK_GEMM = "ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle"
GATE_UP, DOWN = "DecoderMLP.MergedColumnParallelLinear", "DecoderMLP.RowParallelLinear"
QKV, OUT = "DecoderAttention.QKVParallelLinear", "DecoderAttention.RowParallelLinear"
REPLAY_KERNELS, LAYERS = 64, 4
PAGED_ATTENTION = "paged_attention_ll4mi_QKV_mfma16_kernel"


def _entry(name: str, launcher: str) -> str:
    return f"{name} [{launcher}]"


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    """The graph-capture folder of the decode profile: the manifest TraceLens writes and the trace of batch size 64."""
    folder = tmp_path / "graph_capture_profile"
    folder.mkdir()
    shutil.copy(DATA / "decode_capture_execution_details.json", folder / CAPTURE_MANIFEST)
    shutil.copy(DATA / "decode_capture_bs64_launches.json.gz", folder / "DecodeCudaGraphRunner_bs_64_rank0.json.gz")
    return folder


def _decode(folder: Path | None):
    return read_launched_window(DATA / "decode_window_launch_events.json.gz", folder)


#: The correlation id of the decode window's one graph replay.
REPLAY = read_window(DATA / "decode_window_launch_events.json.gz").replays[0].correlation


def _decode_shares(events: list[GpuEvent]) -> dict[str, float]:
    return kernel_window(events, lambda: read_gpu_events(DATA / "decode_rank_trace_gpu_events.json.gz")).shares


def test_the_decode_replays_one_gemm_name_is_four_entries_one_per_projection_of_the_layer(capture: Path) -> None:
    """Without the capture ``kernel_gemm_0`` is one entry (35.3%), the four FlyDSL GEMMs of a layer under one name and
    partly under the names the window recorded them by. With it, each projection is its own entry, the largest first."""
    launched = _decode(capture)
    assert launched.notes == () and launched.renamed == 30
    shares = _decode_shares(launched.events)
    assert GEMM not in shares and sum(shares.values()) == pytest.approx(100.0)
    gemms = {name: share for name, share in shares.items() if name.startswith(f"{GEMM} [")}
    assert list(gemms) == [_entry(GEMM, launcher) for launcher in (GATE_UP, DOWN, QKV, OUT)]
    assert [round(share, 1) for share in gemms.values()] == [28.5, 17.1, 8.5, 7.1]  # 61.2% of the step, not 35.3%


def test_a_name_with_one_launcher_keeps_its_plain_name_and_a_name_with_several_is_split_everywhere(
    capture: Path,
) -> None:
    shares = _decode_shares(_decode(capture).events)
    assert PAGED_ATTENTION in shares and "reshape_and_cache_flash" in shares  # one launcher: RadixAttention
    quant = [name for name in shares if name.startswith("aiter::dynamic_per_token_scaled_quant_kernel")]
    assert len(quant) == 4 and all(name.endswith("]") for name in quant)  # one before each GEMM
    assert sum(1 for name in shares if "[" in name) == 10 and len(shares) == 24


def test_without_the_capture_the_names_stay_as_the_window_records_them(tmp_path: Path) -> None:
    for folder in (None, tmp_path):
        launched = _decode(folder)
        assert [e for e in launched.events if e.name == GEMM and e.launcher] == []
        assert len(launched.notes) == 1 and "left as recorded" in launched.notes[0]
        assert _decode_shares(launched.events)[GEMM] == pytest.approx(35.3, abs=0.05)  # one entry, as recorded


def test_the_capture_names_the_replays_slots_because_the_slots_times_follow_it(capture: Path) -> None:
    """The window records 30 of the replay's 64 kernels (its last two layers) under the name of the slot two places on.

    Taken by its recorded names, ``kernel_gemm_0`` spreads over an order of magnitude (some of its names sit on a norm's
    or an activation's slot); taken by the capture's, each projection takes one time: its layers within 35% of each
    other, each projection at its own median.
    """
    recorded = [e.duration for e in read_window(DATA / "decode_window_launch_events.json.gz").events if e.name == GEMM]
    assert max(recorded) / min(recorded) > 10
    by_launcher: dict[str, list[float]] = defaultdict(list)
    for event in _decode(capture).events:
        if event.name == GEMM:
            by_launcher[event.launcher].append(event.duration)
    assert {launcher: len(times) for launcher, times in by_launcher.items()} == dict.fromkeys(
        (GATE_UP, DOWN, QKV, OUT), LAYERS
    )
    medians = {launcher: statistics.median(times) for launcher, times in by_launcher.items()}
    assert {launcher: round(median, 1) for launcher, median in medians.items()} == {
        GATE_UP: 40.4,
        DOWN: 24.2,
        QKV: 12.1,
        OUT: 10.1,
    }
    for times in by_launcher.values():
        assert max(times) / min(times) < 1.35


def test_a_replay_in_the_order_its_capture_launched_it_keeps_its_names_and_gets_the_launchers(tmp_path: Path) -> None:
    """A replay that runs in capture order (as the node-by-node windows' do) has the capture's names already."""
    launches = read_launches(DATA / "decode_capture_bs64_launches.json.gz")
    events = [GpuEvent(("4", "3"), 10.0 * i, 10.0 * i + 9.0, launch.kernel, 7) for i, launch in enumerate(launches)]
    window = TraceWindow(events, (Replay(7, 64),))
    folder = tmp_path / "capture"
    folder.mkdir()
    shutil.copy(DATA / "decode_capture_execution_details.json", folder / CAPTURE_MANIFEST)
    shutil.copy(DATA / "decode_capture_bs64_launches.json.gz", folder / "DecodeCudaGraphRunner_bs_64_rank0.json.gz")
    result = relabel_replays(window, folder)
    assert (result.renamed, result.notes) == (0, ())
    assert [e.launcher for e in result.events] == [launch.launcher for launch in launches]


def test_a_launcher_is_the_innermost_module_and_the_one_around_it_without_instance_numbers() -> None:
    launches = read_launches(DATA / "decode_capture_bs64_launches.json.gz")
    assert len(launches) == REPLAY_KERNELS
    assert {launch.launcher for launch in launches if launch.kernel == GEMM} == {GATE_UP, DOWN, QKV, OUT}
    assert launches[-1].launcher == "LogitsProcessor" and launches[0].launcher == "DecoderModel.VocabParallelEmbedding"
    assert all(launch.launcher and "_" not in launch.launcher.replace("Linear", "") for launch in launches)


@pytest.mark.parametrize(
    ("batch_size", "why"),
    [
        (None, "it has no capture folder, or its step names no batch size"),
        (80, "graph_capture_profile lists no FULL capture of 80 requests or more with a trace file"),
        (50, "graph_capture_profile lists no FULL capture of 50 requests or more with a trace file"),  # bs 56: no file
    ],
)
def test_a_replay_without_a_capture_to_read_is_left_as_recorded_with_the_reason(
    capture: Path, batch_size: int | None, why: str
) -> None:
    window = read_window(DATA / "decode_window_launch_events.json.gz")
    result = relabel_replays(replace(window, replays=(Replay(REPLAY, batch_size),)), capture)
    assert result.events == window.events and result.renamed == 0
    assert result.notes == (f"the replay of correlation {REPLAY} is left as recorded: {why}",)


def test_a_replay_that_lost_a_kernel_is_not_paired_by_position(capture: Path) -> None:
    """The window's profiler drops kernels (it lost every kernel of another branch's replay): a shorter replay is no
    capture's."""
    window = read_window(DATA / "decode_window_launch_events.json.gz")
    lost = next(i for i, e in enumerate(window.events) if e.correlation == REPLAY and e.name == GEMM)
    short = replace(window, events=[e for i, e in enumerate(window.events) if i != lost])
    result = relabel_replays(short, capture)
    assert f"its {REPLAY_KERNELS - 1} kernels are not the {REPLAY_KERNELS} its capture launched" in result.notes[0]
    assert [e for e in result.events if e.name == GEMM and e.launcher] == []


def test_a_replay_on_two_streams_is_not_paired_by_position(capture: Path) -> None:
    window = read_window(DATA / "decode_window_launch_events.json.gz")
    first = next(i for i, e in enumerate(window.events) if e.correlation == REPLAY)
    events = list(window.events)
    events[first] = replace(events[first], stream=("4", "9"))
    result = relabel_replays(replace(window, events=events), capture)
    assert "more than one stream" in result.notes[0] and result.launched == 1  # the eager sampler's own


def test_a_capture_that_cannot_be_read_is_a_note_not_a_failure(capture: Path) -> None:
    (capture / "DecodeCudaGraphRunner_bs_64_rank0.json.gz").write_bytes(b"not gzip")
    result = _decode(capture)
    assert "is not a readable trace" in result.notes[0] and result.renamed == 0


def test_the_kernels_of_a_prefill_step_have_their_launch_calls_in_the_window_and_split_the_same_way() -> None:
    """Eager kernels: the launching thread's ``nn.Module`` frames around the call give the launcher, no capture needed."""
    launched = read_launched_window(DATA / "root_extend_window_launch_events.json.gz", None)
    shares = kernel_window(launched.events, list).shares
    gemms = [name for name in shares if name.startswith(f"{CK_GEMM} [")]
    assert gemms == [_entry(CK_GEMM, launcher) for launcher in (GATE_UP, DOWN, QKV, OUT)] and CK_GEMM not in shares
    assert [round(shares[name], 1) for name in gemms] == [44.2, 22.1, 9.8, 7.4]


def test_a_kernel_whose_launch_call_the_trace_lacks_keeps_its_plain_name(tmp_path: Path) -> None:
    payload = json.loads(gzip.open(DATA / "root_extend_window_launch_events.json.gz", "rt").read())
    payload["traceEvents"] = [e for e in payload["traceEvents"] if e.get("cat") != "cuda_runtime"]
    path = tmp_path / "no_calls.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    launched = read_launched_window(path, None)
    assert launched.launched == 0 and CK_GEMM in kernel_window(launched.events, list).shares


def _event(launcher: str, name: str = "k", start: float = 0.0) -> GpuEvent:
    return GpuEvent(("4", "3"), start, start + 10.0, name, launcher=launcher)


def test_a_name_is_split_only_when_two_launchers_are_known_unknown_instances_do_not_count() -> None:
    one_known = [_event("A.B", start=0.0), _event("", start=10.0), _event("A.B", start=20.0)]
    assert list(kernel_window(one_known, list).shares) == ["k"]
    two_known = [*one_known, _event("C.D", start=30.0)]
    assert set(kernel_window(two_known, list).shares) == {"k", "k [A.B]", "k [C.D]"}
    assert kernel_window(two_known, list).shares["k [A.B]"] == pytest.approx(50.0)


def test_the_launcher_suffix_is_cut_to_sixty_characters_and_the_name_stays_a_key() -> None:
    long = "M" * 90
    shares = kernel_window([_event(long), _event("N", start=10.0)], list).shares
    assert f"k [{'M' * 60}]" in shares and "k [N]" in shares
