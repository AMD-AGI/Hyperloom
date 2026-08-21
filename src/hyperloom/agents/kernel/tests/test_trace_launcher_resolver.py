###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Tests for recovering a kernel's Python launcher frame from a Kineto trace.

The resolver exists because TraceLens reports ``launcher_path = "Not found"``
for hand-written Triton kernels (no ``cpu_op`` parent), even though the trace
still carries the full ``python_function`` chain down to the launch.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from _trace_launcher_resolver import (  # noqa: E402
    LauncherFrame,
    _match_kernel,
    _open_trace_binary,
    resolve_launchers_from_trace,
)


_USER_FRAME = "/repo/pkg/kernels/moe.py(124): _grouped_gemm"
_TID = 7


def _kernel(name: str, corr: int) -> dict:
    return {"cat": "kernel", "name": name, "ts": 200.0, "dur": 10.0, "args": {"correlation": corr}}


def _runtime(corr: int, api: str = "hipModuleLaunchKernel", ts: float = 150.0) -> dict:
    return {
        "cat": "cuda_runtime",
        "name": api,
        "tid": _TID,
        "ts": ts,
        "dur": 5.0,
        "args": {"correlation": corr},
    }


def _frame(name: str, ts: float = 100.0, dur: float = 200.0, tid: int = _TID) -> dict:
    return {"cat": "python_function", "name": name, "tid": tid, "ts": ts, "dur": dur}


def _write(path: Path, events: list[dict], *, gzipped: bool = False) -> Path:
    blob = json.dumps({"traceEvents": events})
    if gzipped:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(blob)
    else:
        path.write_text(blob, encoding="utf-8")
    return path


def _stack(*frames: str, base_ts: float = 100.0) -> list[dict]:
    """Nested frames, outermost first, all covering the launch at ts=150."""
    out = []
    for depth, name in enumerate(frames):
        out.append(_frame(name, ts=base_ts + depth, dur=200.0 - 2 * depth))
    return out


def test_the_most_specific_overlapping_symbol_wins():
    """``wanted`` is a set, so a first-hit-wins scan raced on PYTHONHASHSEED.

    Both names are substrings of the event, and picking the shorter one binds
    the kernel to a different symbol -- and then to a different source file.
    """
    for _ in range(50):
        assert _match_kernel("moe_gemm1_0.kd", {"moe_gemm1", "moe_gemm1_0"}) == "moe_gemm1_0"


def test_an_exact_hit_outranks_a_longer_substring():
    """Equality is the strongest evidence available, whatever the lengths."""
    assert _match_kernel("moe_gemm1", {"moe_gemm1", "moe_gemm1_0_extra"}) == "moe_gemm1"


def test_kernel_matching_respects_symbol_boundaries_and_decorations():
    """Decorated symbols match, but identifier suffixes do not."""
    assert _match_kernel("foo_epilogue", {"foo"}) is None
    assert _match_kernel("foo<int>", {"foo"}) == "foo"
    assert _match_kernel("foo.kd", {"foo"}) == "foo"
    assert _match_kernel("void foo(float*)", {"foo"}) == "foo"


def test_equally_specific_symbols_resolve_to_nothing():
    """A real tie must not be broken by iteration order."""
    assert _match_kernel("ab_cd", {"ab_", "_cd"}) is None


def test_a_substring_tie_does_not_fall_through_to_an_elided_prefix():
    """The weaker bucket must not rescue an ambiguity in the stronger one."""
    wanted = {"ab_", "_cd", "ab_cd_long_enough_prefix..."}
    assert _match_kernel("ab_cd", wanted) is None


def test_non_exact_match_across_multiple_event_symbols_is_refused(tmp_path):
    """One decorated name must not merge distinct complete event symbols."""
    trace = _write(
        tmp_path / "ambiguous.json",
        [
            *_stack(_USER_FRAME),
            _runtime(1),
            _kernel("foo<int>", 1),
            _runtime(2, ts=160.0),
            _kernel("foo<float>", 2),
        ],
    )
    assert resolve_launchers_from_trace([trace], {"foo"}) == {}


def test_exact_match_survives_ambiguous_decorated_symbols(tmp_path):
    """Ambiguous fallback matches must not discard an available exact event."""
    trace = _write(
        tmp_path / "exact.json",
        [
            *_stack(_USER_FRAME),
            _runtime(1),
            _kernel("foo", 1),
            _runtime(2, ts=160.0),
            _kernel("foo<int>", 2),
            _runtime(3, ts=170.0),
            _kernel("foo<float>", 3),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"foo"})
    assert got["foo"].source_file == "/repo/pkg/kernels/moe.py"
    assert got["foo"].sample_count == 1


def test_a_correlation_id_reused_across_ranks_resolves_nothing(tmp_path):
    """A merged trace restarts correlation ids per rank.

    The id then names two launches, so it names neither, and both are dropped.
    Pairing on ``(pid, correlation)`` cannot rescue this: Kineto records the
    kernel against the device and the launch against the host, so those two
    pids never match on a real trace (see the GPU/host test below). Refusing
    the ambiguous id is the only answer that cannot attribute a kernel to
    another rank's source file.
    """

    def ev(base, pid):
        out = dict(base)
        out["pid"] = pid
        return out

    events = [
        # Rank 0: correlation 1 on tid 7, launched from a.py
        ev(_frame("/repo/rank0/a.py(10): launch_a"), 100),
        ev(_runtime(1), 100),
        ev(_kernel("shared_kernel", 1), 100),
        # Rank 1: same correlation and tid, different launcher
        ev(_frame("/repo/rank1/b.py(20): launch_b"), 200),
        ev(_runtime(1), 200),
        ev(_kernel("other_kernel", 1), 200),
    ]
    trace = _write(tmp_path / "merged.json", events)
    got = resolve_launchers_from_trace([trace], {"shared_kernel", "other_kernel"})
    assert got == {}


def test_a_device_side_kernel_pairs_with_its_host_side_launch(tmp_path):
    """The shape every real Kineto trace has: GPU pid != host pid.

    The kernel event is attributed to the device and the runtime call to the
    host process, so correlation is the only field spanning the pair. Keying on
    pid too made this -- the ordinary case -- resolve to nothing.
    """
    host_pid, gpu_pid, tid = 12345, 0, 777
    events = [
        {
            "cat": "kernel",
            "name": "device_kernel",
            "pid": gpu_pid,
            "tid": 1,
            "ts": 200.0,
            "dur": 10.0,
            "args": {"correlation": 42},
        },
        {
            "cat": "cuda_runtime",
            "name": "hipLaunchKernel",
            "pid": host_pid,
            "tid": tid,
            "ts": 150.0,
            "dur": 5.0,
            "args": {"correlation": 42},
        },
        {
            "cat": "python_function",
            "name": "/repo/moe.py(120): run_moe",
            "pid": host_pid,
            "tid": tid,
            "ts": 100.0,
            "dur": 200.0,
        },
    ]
    got = resolve_launchers_from_trace([_write(tmp_path / "device.json", events)], {"device_kernel"})
    assert got["device_kernel"].frame == "/repo/moe.py(120): run_moe"


def test_one_host_process_driving_several_devices(tmp_path):
    """Correlation stays unique per process, so multi-GPU needs no pid help.

    Each device stream carries its own pid, none of which is the host's; the
    ids do not collide because they are allocated by the one profiled process.
    """
    host_pid, tid = 999, 5
    events = [
        {
            "cat": "python_function",
            "name": "/repo/a.py(1): launch_a",
            "pid": host_pid,
            "tid": tid,
            "ts": 100.0,
            "dur": 100.0,
        },
        {
            "cat": "cuda_runtime",
            "name": "hipLaunchKernel",
            "pid": host_pid,
            "tid": tid,
            "ts": 120.0,
            "dur": 5.0,
            "args": {"correlation": 1},
        },
        {
            "cat": "kernel",
            "name": "kernel_on_gpu0",
            "pid": 0,
            "tid": 1,
            "ts": 130.0,
            "dur": 5.0,
            "args": {"correlation": 1},
        },
        {
            "cat": "python_function",
            "name": "/repo/b.py(2): launch_b",
            "pid": host_pid,
            "tid": tid,
            "ts": 300.0,
            "dur": 100.0,
        },
        {
            "cat": "cuda_runtime",
            "name": "hipLaunchKernel",
            "pid": host_pid,
            "tid": tid,
            "ts": 320.0,
            "dur": 5.0,
            "args": {"correlation": 2},
        },
        {
            "cat": "kernel",
            "name": "kernel_on_gpu1",
            "pid": 1,
            "tid": 1,
            "ts": 330.0,
            "dur": 5.0,
            "args": {"correlation": 2},
        },
    ]
    got = resolve_launchers_from_trace(
        [_write(tmp_path / "multigpu.json", events)],
        {"kernel_on_gpu0", "kernel_on_gpu1"},
    )
    assert got["kernel_on_gpu0"].source_file == "/repo/a.py"
    assert got["kernel_on_gpu1"].source_file == "/repo/b.py"


def test_split_probes_leave_the_kernel_unresolved(tmp_path):
    """A plurality is not agreement.

    Three probes pointing at three different frames used to resolve to
    whichever Counter saw first -- trace order, not evidence.
    """
    events = []
    for i, path in enumerate(["/repo/x.py(1): fa", "/repo/y.py(2): fb", "/repo/z.py(3): fc"]):
        base = 100.0 + i * 1000
        events.append(_frame(path, ts=base, dur=500.0, tid=7 + i))
        events.append(
            {
                "cat": "cuda_runtime",
                "name": "hipModuleLaunchKernel",
                "tid": 7 + i,
                "ts": base + 10,
                "dur": 5.0,
                "args": {"correlation": i + 1},
            }
        )
        events.append(_kernel("split_kernel", i + 1))
    got = resolve_launchers_from_trace([_write(tmp_path / "split.json", events)], {"split_kernel"})
    assert got == {}


def test_per_file_error_is_reported_to_the_caller(tmp_path, monkeypatch):
    """An unreadable trace must not look like "nothing needed resolving"."""
    trace = _write(tmp_path / "t.json", [_kernel("k", 1)])

    def _boom(_fh, **_kwargs):
        """Simulate an I/O error from the streaming parser."""
        raise OSError("truncated gzip")

    monkeypatch.setattr("_trace_launcher_resolver.stream_events", _boom)
    errors: list[str] = []
    got = resolve_launchers_from_trace([trace], {"k"}, file_errors=errors)
    assert got == {}
    assert errors and "t.json" in errors[0] and "OSError" in errors[0]


def test_truncated_trace_reports_an_error_but_keeps_complete_events(tmp_path):
    """Recovered leading events remain usable while corruption is observable."""
    complete = [
        *_stack(_USER_FRAME),
        _runtime(1),
        _kernel("_grouped_gemm_kernel", 1),
    ]
    text = '{"traceEvents": [' + ",".join(json.dumps(event) for event in complete)
    text += ',{"cat":"kernel","name":"cut'
    trace = tmp_path / "truncated.json"
    trace.write_text(text, encoding="utf-8")
    errors: list[str] = []
    got = resolve_launchers_from_trace(
        [trace],
        {"_grouped_gemm_kernel"},
        file_errors=errors,
    )
    assert got["_grouped_gemm_kernel"].source_file == "/repo/pkg/kernels/moe.py"
    assert len(errors) == 1
    assert "truncated.json" in errors[0]
    assert "truncated" in errors[0]


def test_truncated_gzip_fails_soft_and_next_file_resolves(tmp_path):
    """A gzip read failure must not prevent scanning a healthy later trace."""
    broken = _write(
        tmp_path / "broken.json.gz",
        [_kernel("_grouped_gemm_kernel", 1)],
        gzipped=True,
    )
    broken.write_bytes(broken.read_bytes()[:-8])
    healthy = _write(
        tmp_path / "healthy.json",
        [*_stack(_USER_FRAME), _runtime(2), _kernel("_grouped_gemm_kernel", 2)],
    )
    errors: list[str] = []
    got = resolve_launchers_from_trace(
        [broken, healthy],
        {"_grouped_gemm_kernel"},
        max_trace_files=2,
        file_errors=errors,
    )
    assert got["_grouped_gemm_kernel"].source_file == "/repo/pkg/kernels/moe.py"
    assert any("broken.json.gz" in error and "EOFError" in error for error in errors)


def test_missing_trace_events_is_reported_as_a_file_error(tmp_path):
    """A valid non-Kineto JSON document is not a healthy empty trace."""
    trace = tmp_path / "not-a-trace.json"
    trace.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    errors: list[str] = []
    assert resolve_launchers_from_trace([trace], {"kernel"}, file_errors=errors) == {}
    assert errors == ["not-a-trace.json: stream error: traceEvents array not found"]


def test_single_trailing_dot_is_not_treated_as_elided(tmp_path):
    """Prefix matching must be gated by the same predicate as the ambiguity check.

    _match_kernel used to accept any name that rstrip changed, so a single
    trailing dot enabled prefix matching while _is_elided still said False --
    the name then skipped the ambiguity check and could bind to any sibling.
    """
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack("/repo/pkg/k.py(7): launch_it"),
            _runtime(1),
            _kernel("_ZN5aiter12some_kernelEv", 1),
        ],
    )
    # "_ZN5aiter1." is a prefix of the event symbol but is not ellipsis-elided.
    assert resolve_launchers_from_trace([trace], {"_ZN5aiter1."}) == {}


def test_scan_is_bounded_when_kernels_never_resolve(tmp_path, monkeypatch):
    """A capture folder must not be read end to end chasing the unresolvable.

    Kernels launched from C++ or only replayed from a graph never resolve, so
    "stop once everything is resolved" alone would read every file in the
    folder -- dozens of them, two streaming passes each.
    """
    files = []
    for i in range(10):
        files.append(_write(tmp_path / f"rank{i}.json", [_kernel("_never_launched_kernel", i + 1)]))
    opened: list[str] = []

    def _counting_open(path):
        opened.append(Path(path).name)
        return _open_trace_binary(path)

    monkeypatch.setattr("_trace_launcher_resolver._open_trace_binary", _counting_open)
    got = resolve_launchers_from_trace(files, {"_never_launched_kernel"})
    assert got == {}
    # Distinct files touched, not passes: bounded well below the 10 supplied.
    assert len(set(opened)) <= 2, f"scanned {sorted(set(opened))}"


def test_scan_stops_when_barren_limit_is_reached(tmp_path, monkeypatch):
    """The first healthy barren file reaches the one-file stop threshold."""
    files = [_write(tmp_path / f"rank{i}.json", [_kernel("_never_launched_kernel", i + 1)]) for i in range(3)]
    opened: list[str] = []

    def _counting_open(path):
        """Record each trace opened by the resolver."""
        opened.append(Path(path).name)
        return _open_trace_binary(path)

    monkeypatch.setattr("_trace_launcher_resolver._open_trace_binary", _counting_open)
    assert (
        resolve_launchers_from_trace(
            files,
            {"_never_launched_kernel"},
            max_trace_files=3,
        )
        == {}
    )
    assert opened == ["rank0.json"]


def test_scan_stops_after_a_file_adds_nothing(tmp_path):
    """A file that contributes no new resolution ends the scan."""
    good = _write(
        tmp_path / "a.json",
        [*_stack("/repo/pkg/k.py(7): launch_it"), _runtime(1), _kernel("k_one", 1)],
    )
    barren = _write(tmp_path / "b.json", [_kernel("k_two", 2)])
    got = resolve_launchers_from_trace([good, barren], {"k_one", "k_two"})
    assert set(got) == {"k_one"}


def test_non_launch_runtime_events_are_not_retained(tmp_path):
    """Correlated memcpy/synchronize events must not enter the probe map."""
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack("/repo/pkg/k.py(7): launch_it"),
            {
                "cat": "cuda_runtime",
                "name": "hipMemcpyAsync",
                "tid": _TID,
                "ts": 140.0,
                "dur": 1.0,
                "args": {"correlation": 1},
            },
            _runtime(2),
            _kernel("k_one", 2),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"k_one"})
    assert got["k_one"].launch_api == "hipModuleLaunchKernel"


def _tied_frames_trace(tmp_path, *, outer_first: bool) -> Path:
    """Two frames sharing a ``ts``; only ``dur`` reveals which one is nested."""
    outer = {
        "cat": "python_function",
        "name": "/opt/aiter/aiter/fused_moe.py(1169): fused_moe",
        "tid": _TID,
        "ts": 100.0,
        "dur": 50.0,
    }
    inner = {
        "cat": "python_function",
        "name": "/opt/aiter/aiter/ops/flydsl/moe_kernels.py(859): _moe_kernel",
        "tid": _TID,
        "ts": 100.0,
        "dur": 10.0,
    }
    frames = [outer, inner] if outer_first else [inner, outer]
    name = f"tied_{'outer' if outer_first else 'inner'}.json"
    return _write(
        tmp_path / name,
        [*frames, _runtime(1, ts=105.0), _kernel("_moe_gemm_kernel", 1)],
    )


def test_same_timestamp_frames_resolve_by_nesting_not_write_order(tmp_path):
    """Microsecond ties must not let trace write order pick the frame.

    Profiler timestamps are microsecond-granular, so adjacent frames on a fast
    call chain land on the same ``ts``. Ordering by start alone left the tie to
    Python's stable sort, i.e. to the order events happen to appear in the file,
    which collapsed different kernels onto whichever frame was written first.
    """
    expected = ("/opt/aiter/aiter/ops/flydsl/moe_kernels.py", 859, "_moe_kernel")
    picks = []
    for outer_first in (True, False):
        got = resolve_launchers_from_trace(
            [_tied_frames_trace(tmp_path, outer_first=outer_first)], {"_moe_gemm_kernel"}
        )
        frame = got["_moe_gemm_kernel"]
        picks.append((frame.source_file, frame.line, frame.function))
    assert picks[0] == picks[1], "write order changed the resolved source"
    assert picks[0] == expected


def test_resolves_user_frame_behind_triton_plumbing(tmp_path):
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack(
                "/repo/serve/runner.py(10): forward",
                _USER_FRAME,
                "/venv/triton/runtime/jit.py(708): run",
                "/venv/triton/backends/amd/driver.py(831): __call__",
                "<built-in function launch>",
            ),
            _runtime(1),
            _kernel("_grouped_gemm_kernel", 1),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"})
    assert set(got) == {"_grouped_gemm_kernel"}
    frame = got["_grouped_gemm_kernel"]
    assert frame.source_file == "/repo/pkg/kernels/moe.py"
    assert frame.line == 124
    assert frame.function == "_grouped_gemm"
    assert frame.launch_api == "hipModuleLaunchKernel"
    assert frame.frame == _USER_FRAME


def test_reads_gzipped_trace(tmp_path):
    trace = _write(
        tmp_path / "t.json.gz",
        [*_stack(_USER_FRAME), _runtime(1), _kernel("_grouped_gemm_kernel", 1)],
        gzipped=True,
    )
    got = resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"})
    assert got["_grouped_gemm_kernel"].line == 124


def test_pure_graph_replay_kernel_is_omitted(tmp_path):
    """A replay stack ends at torch/cuda/graphs.py, so nothing is guessed."""
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack(
                "/repo/serve/runner.py(10): forward",
                "/venv/torch/cuda/graphs.py(139): replay",
            ),
            _runtime(1, api="hipGraphLaunch"),
            _kernel("_grouped_gemm_kernel", 1),
        ],
    )
    assert resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"}) == {}


def test_graph_launch_api_without_a_torch_marker_is_omitted(tmp_path):
    """API identity rejects inductor and framework-native graph replay stacks."""
    for index, api in enumerate(("hipGraphLaunch", "cudaGraphLaunch", "cuGraphLaunch"), start=1):
        trace = _write(
            tmp_path / f"graph-{index}.json",
            [
                *_stack(
                    "/repo/serve/runner.py(10): forward",
                    "/venv/torch/_inductor/cudagraph_trees.py(2100): run",
                ),
                _runtime(index, api=api),
                _kernel("_grouped_gemm_kernel", index),
            ],
        )
        assert resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"}) == {}


def test_eager_probe_preferred_over_graph_replay(tmp_path):
    """With both launch paths present, the eager one supplies the answer."""
    trace = _write(
        tmp_path / "t.json",
        [
            # Graph replay on one thread.
            _frame("/venv/torch/cuda/graphs.py(139): replay", ts=100.0, dur=100.0, tid=9),
            {
                "cat": "cuda_runtime",
                "name": "hipGraphLaunch",
                "tid": 9,
                "ts": 150.0,
                "dur": 5.0,
                "args": {"correlation": 1},
            },
            _kernel("_grouped_gemm_kernel", 1),
            # Eager launch on another thread.
            *_stack(_USER_FRAME),
            _runtime(2),
            _kernel("_grouped_gemm_kernel", 2),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"})
    assert got["_grouped_gemm_kernel"].source_file == "/repo/pkg/kernels/moe.py"
    assert got["_grouped_gemm_kernel"].launch_api == "hipModuleLaunchKernel"


def test_non_python_frames_are_ignored(tmp_path):
    """nn.Module markers carry no path(line) and must not be mistaken for source."""
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack(
                _USER_FRAME,
                "nn.Module: FusedMoE_0",
                "/venv/torch/nn/modules/module.py(1779): _call_impl",
            ),
            _runtime(1),
            _kernel("_grouped_gemm_kernel", 1),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"})
    assert got["_grouped_gemm_kernel"].source_file == "/repo/pkg/kernels/moe.py"


def test_kernel_without_matching_runtime_is_omitted(tmp_path):
    trace = _write(tmp_path / "t.json", [*_stack(_USER_FRAME), _kernel("_grouped_gemm_kernel", 99)])
    assert resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"}) == {}


def test_truncated_mangled_symbol_still_matches(tmp_path):
    """TraceLens elides a long mangled symbol; the trace keeps the full name.

    Real case: the Kernel Name cell reads
    ``_ZN5aiter33reduce_scatter_cross_device_storeIDF16bLi8EEEvPNS_8RankDataENS_1...``
    while the trace event is the untruncated symbol.
    """
    full = "_ZN5aiter33reduce_scatter_cross_device_storeIDF16bLi8EEEvPNS_8RankDataENS_11RankSignalsEiiii"
    truncated = "_ZN5aiter33reduce_scatter_cross_device_storeIDF16bLi8EEEvPNS_8RankDataENS_1..."
    trace = _write(
        tmp_path / "t.json",
        [*_stack(_USER_FRAME), _runtime(1), _kernel(full, 1)],
    )
    got = resolve_launchers_from_trace([trace], {truncated})
    assert set(got) == {truncated}
    assert got[truncated].line == 124


# --- JIT toolchain plumbing (real stacks from a Kimi-K3 / vLLM run) ----------


def test_flydsl_jit_plumbing_is_skipped_for_the_real_kernel(tmp_path):
    """Verbatim eager stack of aiter's FlyDSL MoE GEMM.

    Every FlyDSL-compiled kernel bottoms out in the same
    ``flydsl/compiler/jit_executor.py: __call__``; resolving there aims a backend
    at the compiler and collapses distinct kernels onto one source.
    """
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack(
                "aiter/fused_moe.py(538): fused_moe_",
                "aiter/fused_moe.py(2478): fused_moe_2stages",
                "aiter/fused_moe.py(1169): _flydsl_stage1_wrapper",
                "aiter/ops/flydsl/moe_kernels.py(1261): flydsl_moe_stage1",
                "aiter/ops/flydsl/moe_kernels.py(859): _run_compiled",
                "flydsl/compiler/jit_function.py(1635): __call__",
                "flydsl/compiler/jit_executor.py(210): __call__",
                "<flydsl-dispatch>(34): dispatch",
            ),
            _runtime(1),
            _kernel("moe_gemm1_0", 1),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"moe_gemm1_0"})
    frame = got["moe_gemm1_0"]
    assert frame.source_file == "aiter/ops/flydsl/moe_kernels.py"
    assert frame.line == 859
    assert "jit_executor" not in frame.source_file
    assert "jit_function" not in frame.source_file


def test_two_flydsl_kernels_do_not_collapse_onto_the_compiler(tmp_path):
    """The concrete damage: both MoE GEMMs used to resolve to jit_executor.py."""

    def _flydsl_stack(kernel_line: int, base_ts: float) -> list[dict]:
        return _stack(
            f"aiter/ops/flydsl/moe_kernels.py({kernel_line}): flydsl_moe",
            "flydsl/compiler/jit_executor.py(210): __call__",
            base_ts=base_ts,
        )

    trace = _write(
        tmp_path / "t.json",
        [
            *_flydsl_stack(1261, 100.0),
            _runtime(1, ts=150.0),
            _kernel("moe_gemm1_0", 1),
            *_flydsl_stack(1480, 400.0),
            _runtime(2, ts=450.0),
            _kernel("moe_gemm2_0", 2),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"moe_gemm1_0", "moe_gemm2_0"})
    assert got["moe_gemm1_0"].line == 1261
    assert got["moe_gemm2_0"].line == 1480


def test_flydsl_authored_kernel_stays_visible(tmp_path):
    """Only flydsl/compiler/ is plumbing; a kernel written in FlyDSL is a target."""
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack(
                "/repo/kernels/my_flydsl_kernel.py(42): my_kernel",
                "flydsl/compiler/jit_executor.py(210): __call__",
            ),
            _runtime(1),
            _kernel("my_flydsl_kernel_0", 1),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"my_flydsl_kernel_0"})
    assert got["my_flydsl_kernel_0"].source_file == "/repo/kernels/my_flydsl_kernel.py"


def test_aiter_jit_and_flydsl_are_treated_alike(tmp_path):
    """Both JIT toolchains must yield the caller, not their dispatch wrapper."""
    cases = {
        "aiter_kernel_0": "aiter/jit/core.py(1504): wrapper",
        "flydsl_kernel_0": "flydsl/compiler/jit_executor.py(210): __call__",
    }
    for index, (kernel, plumbing) in enumerate(cases.items(), start=1):
        trace = _write(
            tmp_path / f"t{index}.json",
            [
                *_stack("/repo/ops/real_kernel.py(77): launch_it", plumbing),
                _runtime(index),
                _kernel(kernel, index),
            ],
        )
        got = resolve_launchers_from_trace([trace], {kernel})
        assert got[kernel].source_file == "/repo/ops/real_kernel.py", kernel
        assert got[kernel].line == 77, kernel


def test_triton_stack_unaffected_by_the_flydsl_rule(tmp_path):
    """Regression guard: the vLLM Triton path from the same run still resolves."""
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack(
                "vllm/models/kimi_k3/amd/linear.py(571): forward",
                "vllm/models/kimi_k3/amd/ops/attn_res.py(88): attn_res",
                "triton/runtime/jit.py(720): run",
                "triton/backends/amd/driver.py(343): __call__",
                "<built-in function launch>",
            ),
            _runtime(1),
            _kernel("_attn_res_kernel", 1),
        ],
    )
    got = resolve_launchers_from_trace([trace], {"_attn_res_kernel"})
    assert got["_attn_res_kernel"].source_file == "vllm/models/kimi_k3/amd/ops/attn_res.py"
    assert got["_attn_res_kernel"].line == 88


def test_ambiguous_elided_prefix_is_refused(tmp_path):
    """Two distinct symbols behind one elided name cannot be told apart.

    Attributing either launcher to the shared prefix would hand a backend the
    wrong source file, so the resolver declines and leaves it to grep.
    """
    shared = "_ZN5aiter40some_quite_long_collective_symbol_nameI"
    trace = _write(
        tmp_path / "t.json",
        [
            *_stack(_USER_FRAME),
            _runtime(1),
            _kernel(shared + "Lb0EEEvPf", 1),
            _runtime(2, ts=160.0),
            _kernel(shared + "Lb1EEEvPd", 2),
        ],
    )
    assert resolve_launchers_from_trace([trace], {shared + "..."}) == {}


def test_unambiguous_elided_prefix_still_resolves(tmp_path):
    """One symbol behind the elision is fine -- that is the normal case."""
    full = "_ZN5aiter40some_quite_long_collective_symbol_nameILb0EEEvPf"
    trace = _write(
        tmp_path / "t.json",
        [*_stack(_USER_FRAME), _runtime(1), _kernel(full, 1)],
    )
    elided = "_ZN5aiter40some_quite_long_collective_symbol_nameI..."
    assert set(resolve_launchers_from_trace([trace], {elided})) == {elided}


def test_degenerate_elided_prefix_is_refused(tmp_path):
    """``_ZN5...`` prefixes an entire namespace and must not bind a kernel."""
    trace = _write(
        tmp_path / "t.json",
        [*_stack(_USER_FRAME), _runtime(1), _kernel("_ZN5aiter9unrelatedE", 1)],
    )
    assert resolve_launchers_from_trace([trace], {"_ZN5..."}) == {}


def test_unwanted_kernels_are_not_resolved(tmp_path):
    trace = _write(
        tmp_path / "t.json",
        [*_stack(_USER_FRAME), _runtime(1), _kernel("_some_other_kernel", 1)],
    )
    assert resolve_launchers_from_trace([trace], {"_grouped_gemm_kernel"}) == {}


def test_empty_inputs_short_circuit(tmp_path):
    trace = _write(tmp_path / "t.json", [_kernel("_k", 1)])
    assert resolve_launchers_from_trace([], {"_k"}) == {}
    assert resolve_launchers_from_trace([trace], set()) == {}
    assert resolve_launchers_from_trace([trace], {"  "}) == {}


def test_missing_or_corrupt_trace_fails_soft(tmp_path):
    """Missing and malformed inputs remain non-fatal."""
    missing = tmp_path / "nope.json"
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("this is not json", encoding="utf-8")
    assert resolve_launchers_from_trace([missing], {"_k"}) == {}
    assert resolve_launchers_from_trace([corrupt], {"_k"}) == {}


def test_stops_after_all_kernels_resolved(tmp_path):
    """The second file must not be opened once the first answered everything."""
    first = _write(
        tmp_path / "a.json",
        [*_stack(_USER_FRAME), _runtime(1), _kernel("_grouped_gemm_kernel", 1)],
    )
    absent = tmp_path / "b-does-not-exist.json"
    got = resolve_launchers_from_trace([first, absent], {"_grouped_gemm_kernel"})
    assert set(got) == {"_grouped_gemm_kernel"}


def test_launcher_frame_render():
    frame = LauncherFrame(
        source_file="/repo/a.py",
        line=7,
        function="fn",
        sample_count=2,
        launch_api="hipLaunchKernel",
    )
    assert frame.frame == "/repo/a.py(7): fn"
