# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A PD profile is analysed once over every node, and each node's top kernels reach the orchestration prompt."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors import profile
from hyperloom.orchestrator.actions.executors import roofline as rf
from hyperloom.orchestrator.actions.executors import trace_analyze
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state._shared_state import variant_line
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute

_PD_INPUT = {
    "role_node_traces": {"decode": ["/t/decode/1-TP-0.trace.json.gz"], "prefill": ["/t/prefill/1-TP-0.trace.json.gz"]},
    "trace_dir": "/t",
}


def test_capture_health_check_finds_a_role_capture_folder(tmp_path: Path) -> None:
    capture = tmp_path / "decode" / "graph_capture_profile"
    capture.mkdir(parents=True)
    (capture / "DecodeCudaGraphRunner_bs_1_rank0.json.gz").write_bytes(b"x")

    found, files = profile._capture_trace_files(tmp_path)

    assert found == capture
    assert [f.name for f in files] == ["DecodeCudaGraphRunner_bs_1_rank0.json.gz"]


def _run(route: AnalysisRoute) -> trace_analyze._AnalysisRun:
    return trace_analyze._AnalysisRun(
        trace_input="/t/decode/1-TP-0.trace.json.gz",
        workspace_path="/ws",
        tracelens_root=Path("/tl"),
        route=route,
        scriptable=False,
        model_name="m",
        framework="sglang",
        target_platform="MI300X",
    )


def test_agent_route_passes_the_pd_node_traces_to_the_tool() -> None:
    payload = {"pd_node_traces": _PD_INPUT, "steady_state_mode": "decode_only_or_generic", "split_llm_inference": False}

    cmd, mode = trace_analyze._build_trace_analyze_cmd(
        payload, _run(AnalysisRoute.AGENT), session_dir=Path("/s"), workload={}
    )

    assert json.loads(cmd[cmd.index("--pd-node-traces") + 1]) == _PD_INPUT
    assert mode == "decode_only_or_generic" and "--split-llm-inference" not in cmd


def test_non_llm_route_analyses_only_the_main_trace() -> None:
    payload = {"pd_node_traces": _PD_INPUT, "steady_state_mode": "decode_only_or_generic", "split_llm_inference": False}

    cmd, _mode = trace_analyze._build_trace_analyze_cmd(
        payload, _run(AnalysisRoute.NON_LLM), session_dir=Path("/s"), workload={}
    )

    assert "--pd-node-traces" not in cmd


@pytest.mark.parametrize(("clamp", "expected"), [(True, True), (False, False), (None, False)])
def test_stream_overlap_clamp_flag_follows_the_payload(clamp, expected) -> None:
    payload = {} if clamp is None else {"clamp_stream_overlaps": clamp}

    cmd, _mode = trace_analyze._build_trace_analyze_cmd(
        payload, _run(AnalysisRoute.AGENT), session_dir=Path("/s"), workload={}
    )

    assert ("--clamp-stream-overlaps" in cmd) is expected


def test_roofline_no_longer_runs_the_per_role_or_csv_reports() -> None:
    assert not hasattr(rf, "report_pd_profile") and not hasattr(rf, "analyze_role_traces")
    assert not hasattr(rf.RooflineExecutor, "_record_role_analyses")


def _analysis_json(path: Path) -> None:
    def task(node: str, name: str) -> dict:
        return {
            "operation": name,
            "node": node,
            "members": [{"kernel_name": [f"{name}_k"], "pct_e2e": 10.0, "category": "GEMM", "bound": "compute"}],
        }

    tasks = [task("decode_0", f"d{i}") for i in range(3)] + [task("prefill_0", "p0")]
    tasks += [task("decode_0", f"d{i}") for i in range(3, 6)] + [task("prefill_0", "p1")]
    path.write_text(json.dumps({"compute_optimizations": tasks}), encoding="utf-8")


@pytest.mark.asyncio
async def test_pd_profile_is_analysed_once_and_summarised_per_node(tmp_path: Path) -> None:
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    _analysis_json(tmp_path / "analysis.json")
    profile_result = {
        "status": "succeeded",
        "main_trace_path": _PD_INPUT["role_node_traces"]["decode"][0],
        "trace_dir": "/t",
        "role_traces": {role: paths[0] for role, paths in _PD_INPUT["role_node_traces"].items()},
        "role_node_traces": _PD_INPUT["role_node_traces"],
        "workspace": "/tmp/workspace",
    }
    payloads: list[dict] = []

    async def fake_profile(ctx):
        return profile_result

    async def fake_ta(payload, *, session_dir):
        payloads.append(dict(payload))
        return {
            "status": "ok",
            "trace_report_path": str(md),
            "hot_kernels": [{"name": "d0", "tracelens_node": "decode_0"}],
            "trace_health_warnings": [
                {"code": "high_gpu_idle_pct", "severity": "warning", "node": "prefill_0"},
                {"code": "pd_node_shares", "severity": "info", "nodes": {}},
            ],
        }

    state = SharedState()
    state.baseline_tput = 100.0
    task = Task(
        task_id="t-pd",
        kind="roofline",
        state="running",
        params={"base_extra_args": ""},
        idempotency_key="roofline:pd",
        requires_lanes=["profile_lane"],
    )
    ctx = RunnerContext(task=task, lease=None, extra={"session_dir": str(tmp_path)})
    with (
        patch("hyperloom.orchestrator.actions.executors.profile.profile_executor", new=fake_profile),
        patch("hyperloom.orchestrator.actions.executors.trace_analyze.trace_analyze_handler", new=fake_ta),
    ):
        await rf.RooflineExecutor(shared_state=state)(ctx)

    assert len(payloads) == 1
    assert payloads[0]["pd_node_traces"] == _PD_INPUT
    assert payloads[0]["steady_state_mode"] == "decode_only_or_generic"
    assert payloads[0]["clamp_stream_overlaps"] is True
    by_node = state.last_trace_analyze_by_role
    assert list(by_node) == ["decode_0", "prefill_0"]
    assert [k["name"] for k in by_node["decode_0"]["top"]] == ["d0", "d1", "d2", "d3", "d4"]
    assert [k["name"] for k in by_node["prefill_0"]["top"]] == ["p0", "p1"]
    assert by_node["prefill_0"]["warnings"] == ["high_gpu_idle_pct"] and by_node["decode_0"]["warnings"] == []
    text = variant_line.role_trace_text(by_node)
    assert text.startswith("prefill_0: p0/GEMM/compute 10.0%") and "decode_0: d0" in text


def test_a_single_role_profile_keeps_the_main_trace_payload() -> None:
    assert rf._pd_multi_trace_input({"role_node_traces": {"decode": ["/t/d.gz"]}, "trace_dir": "/t"}) is None
    assert rf._pd_multi_trace_input({"role_node_traces": {"prefill": ["/a", "/b"]}, "trace_dir": "/t"}) is None
    assert rf._pd_multi_trace_input({"role_node_traces": _PD_INPUT["role_node_traces"], "trace_dir": "/t"}) == (
        _PD_INPUT
    )
