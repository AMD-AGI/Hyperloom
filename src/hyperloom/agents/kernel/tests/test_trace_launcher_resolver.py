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
