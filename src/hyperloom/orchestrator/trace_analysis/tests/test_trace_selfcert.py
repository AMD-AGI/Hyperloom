###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""What the self-certification record concludes from a trace directory: the steps it finds and who may use it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis.trace_selfcert import certify_trace_dir


def _step(name: str, ts: int, dur: int) -> dict:
    """An iteration root on the host timeline."""
    return {"cat": "user_annotation", "ph": "X", "name": name, "ts": ts, "dur": dur}


def _launch(corr: int, ts: int, kernels: list[tuple[int, int]], *, api: str = "hipLaunchKernel") -> list[dict]:
    """An op, its launch at ``ts``, and the (start, duration) of every kernel that launch produced."""
    return [
        {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": corr, "Input Dims": [[4, 4]]}},
        {
            "cat": "cuda_runtime",
            "ph": "X",
            "name": api,
            "ts": ts,
            "dur": 5,
            "args": {"correlation": corr, "External id": corr},
        },
        *(
            {"cat": "kernel", "ph": "X", "name": "gemm", "ts": start, "dur": dur, "args": {"correlation": corr}}
            for start, dur in kernels
        ),
    ]


#: An SGLang extend step whose graph replay ran two kernels beside one plain launch, then a decode step.
_STEP_EVENTS = [
    _step("step[EXTEND bs=2 toks=64]", 1000, 400),
    *_launch(1, 1010, [(1020, 50)]),
    *_launch(2, 1100, [(1110, 40), (1160, 40)], api="hipGraphLaunch"),
    _step("step[DECODE bs=2]", 1500, 200),
    *_launch(3, 1510, [(1520, 30)]),
]


def _write(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({"traceEvents": events}).encode("utf-8"))


def test_each_step_root_is_credited_with_the_kernels_its_own_launches_produced(tmp_path):
    _write(tmp_path / "rank0.pt.trace.json", _STEP_EVENTS)

    annotations = certify_trace_dir(tmp_path, tmp_path, framework="sglang")["rank_level"][0]["annotations"]

    assert annotations["step_root_pattern_family"] == "backup"
    steps = [
        (s["kind"], s["is_prefill"], s["runtime_launch_count"], s["graph_launch_count"], s["kernel_count"])
        for s in annotations["steps"]
    ]
    assert steps == [("EXTEND", True, 2, 1, 3), ("DECODE", False, 1, 0, 1)]
    assert [s["gpu_busy_us"] for s in annotations["steps"]] == [130.0, 30.0]


@pytest.mark.parametrize(
    ("trace_input", "certified"),
    [
        pytest.param(".", "rank0.pt.trace.json", id="directory"),
        pytest.param("trace_split/chunk.pt.trace.json", "chunk.pt.trace.json", id="file"),
    ],
)
def test_the_certificate_reads_the_file_the_analysis_reads(tmp_path, trace_input, certified):
    _write(tmp_path / "rank0.pt.trace.json", _STEP_EVENTS)
    _write(tmp_path / "trace_split" / "chunk.pt.trace.json", _STEP_EVENTS)

    record = certify_trace_dir(tmp_path, tmp_path / trace_input, framework="sglang")

    assert Path(record["trace_dir_level"]["selected_path"]).name == certified


def test_kernels_no_op_claims_leave_the_trace_unusable(tmp_path):
    _write(tmp_path / "rank0.pt.trace.json", [ev for ev in _STEP_EVENTS if ev["cat"] != "cpu_op"])

    verdict = certify_trace_dir(tmp_path, tmp_path, framework="sglang")["verdict"]

    assert verdict["usable_by"] == []
    assert verdict["blocking_reasons"] == ["correlation chain resolves no kernel to an op"]
