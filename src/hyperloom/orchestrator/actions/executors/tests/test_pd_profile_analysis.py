# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A PD profile is analysed once over every node with one multi-trace TraceLens run."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors import profile
from hyperloom.orchestrator.actions.executors import roofline as rf
from hyperloom.orchestrator.actions.executors import trace_analyze
from hyperloom.orchestrator.actions.executors._pd_role_trace import pd_multi_trace_input
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute

_PD_INPUT = {
    "role_node_traces": {
        "decode": ["/t/decode/0/1-TP-0.trace.json.gz"],
        "prefill": ["/t/prefill/0/1-TP-0.trace.json.gz"],
    },
}
_PD_PAYLOAD = {
    "trace_input": "/t/decode/0/1-TP-0.trace.json.gz",
    "pd_node_traces": _PD_INPUT,
    "steady_state_mode": "decode_only_or_generic",
    "split_llm_inference": False,
    "clamp_stream_overlaps": True,
}


def _run(route: AnalysisRoute) -> trace_analyze._AnalysisRun:
    return trace_analyze._AnalysisRun(
        trace_input="/t/decode/0/1-TP-0.trace.json.gz",
        workspace_path="/ws",
        tracelens_root=Path("/tl"),
        route=route,
        scriptable=False,
        model_name="m",
        framework="sglang",
        target_platform="MI300X",
    )


def test_agent_route_passes_the_pd_node_traces_to_the_tool() -> None:
    cmd, mode = trace_analyze._build_trace_analyze_cmd(
        _PD_PAYLOAD, _run(AnalysisRoute.AGENT), session_dir=Path("/s"), workload={"conc": 64}
    )

    assert json.loads(cmd[cmd.index("--pd-node-traces") + 1]) == _PD_INPUT
    assert mode == "decode_only_or_generic"
    assert "--split-llm-inference" not in cmd and "--split-conc" not in cmd
    assert "--clamp-stream-overlaps" in cmd


def test_non_llm_route_analyses_only_the_main_trace() -> None:
    cmd, _mode = trace_analyze._build_trace_analyze_cmd(
        _PD_PAYLOAD, _run(AnalysisRoute.NON_LLM), session_dir=Path("/s"), workload={}
    )

    assert "--pd-node-traces" not in cmd


def test_a_plain_payload_keeps_the_stream_overlaps() -> None:
    cmd, _mode = trace_analyze._build_trace_analyze_cmd(
        {"trace_input": "/t/x.trace.json.gz"}, _run(AnalysisRoute.AGENT), session_dir=Path("/s"), workload={}
    )

    assert "--clamp-stream-overlaps" not in cmd and "--pd-node-traces" not in cmd


def test_capture_health_check_finds_a_role_capture_folder(tmp_path: Path) -> None:
    capture = tmp_path / "decode" / "graph_capture_profile"
    capture.mkdir(parents=True)
    (capture / "DecodeCudaGraphRunner_bs_1_rank0.json.gz").write_bytes(b"x")

    found, files = profile._capture_trace_files(tmp_path)

    assert found == capture
    assert [f.name for f in files] == ["DecodeCudaGraphRunner_bs_1_rank0.json.gz"]


async def _pd_roofline_payloads(tmp_path: Path, hot_kernels: list[dict]) -> list[dict]:
    """Run the roofline executor on a two-role PD profile and return every trace_analyze payload it sent."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
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
        return {"status": "ok", "trace_report_path": str(md), "hot_kernels": hot_kernels}

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
    return payloads


@pytest.mark.asyncio
async def test_pd_profile_is_analysed_once_on_every_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", raising=False)
    payloads = await _pd_roofline_payloads(tmp_path, [{"name": "d0"}])

    assert len(payloads) == 1
    assert payloads[0]["trace_input"] == _PD_INPUT["role_node_traces"]["decode"][0]
    assert payloads[0]["pd_node_traces"] == _PD_INPUT
    assert payloads[0]["steady_state_mode"] == "decode_only_or_generic"
    assert payloads[0]["split_llm_inference"] is False
    assert payloads[0]["clamp_stream_overlaps"] is True


@pytest.mark.asyncio
async def test_compute_bound_reanalysis_keeps_the_pd_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", raising=False)
    monkeypatch.setattr(rf, "is_multi_node", lambda: True)
    monkeypatch.setattr(rf, "_trace_is_high_idle", lambda result: True)
    payloads = await _pd_roofline_payloads(tmp_path, [])

    assert len(payloads) == 2
    for key in ("trace_input", "pd_node_traces", "steady_state_mode", "split_llm_inference", "clamp_stream_overlaps"):
        assert payloads[1][key] == payloads[0][key]


def test_a_single_role_or_single_node_profile_keeps_the_main_trace_payload() -> None:
    assert pd_multi_trace_input({"role_node_traces": {"decode": ["/t/d.gz"]}}) is None
    assert pd_multi_trace_input({"role_node_traces": {"prefill": ["/a", "/b"]}}) is None
    assert pd_multi_trace_input({}) is None
    assert pd_multi_trace_input({"role_node_traces": {"decode": ["/a", "/b"]}}) == {
        "role_node_traces": {"decode": ["/a", "/b"]}
    }
    assert pd_multi_trace_input({"role_node_traces": _PD_INPUT["role_node_traces"]}) == _PD_INPUT
