# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``tracelens`` tool's default route runs TraceLens' split and perf report with no agent; ``agent`` still runs one."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.contracts.notebook.records import ToolResultPointer
from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolRequest, roofline, tracelens_tool
from hyperloom_multibranch.toolbox.tools import tracelens_report as report
from hyperloom_multibranch.toolbox.tools.op_shares import DEGRADED_NOTICE
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink

SPLITTER = "TraceLens.TraceUtils.split_trace.main"
CHUNK = "mixed_steady_state_prefill_1_prefilldecode_0_decode_31_bs11_conc3_t-TP-0.trace.json.gz"
OPS_SUMMARY = (
    "name,parent_module,Percentage (%)\naten::mm,A,20.0\naten::mm,B,4.5\naten::copy_,C,19.0\naten::mul,D,10.0\n"
)
SHARES = {"aten::mm": 24.5, "aten::copy_": 19.0, "aten::mul": 10.0}
ANALYSIS_JSON = {"report_info": {"mode": "deterministic-fallback", "trace_quality": "poor", "warnings": ["collapsed"]}}


class Shell:
    """Stands in for ``run_with_session_kill``: writes what TraceLens' splitter and perf report would, records the argv."""

    def __init__(self, *, chunk: str | None = CHUNK, report_rc: int = 0) -> None:
        self.commands: list[list[str]] = []
        self._chunk, self._report_rc = chunk, report_rc

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.commands.append(cmd)
        assert Path(kwargs["cwd"]).is_dir() and kwargs["timeout"] == 3600
        if SPLITTER in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.mkdir(parents=True)
            if self._chunk:
                (out / self._chunk).write_bytes(b"chunk")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        csvs = Path(cmd[cmd.index("--output_csvs_dir") + 1])
        csvs.mkdir(parents=True)
        (csvs / "ops_summary.csv").write_text(OPS_SUMMARY, encoding="utf-8")
        (csvs / "unified_perf_summary.csv").write_text("name\n", encoding="utf-8")
        Path(cmd[cmd.index("--output_xlsx_path") + 1]).write_bytes(b"xlsx")
        return subprocess.CompletedProcess(cmd, self._report_rc, "", "perf report blew up" if self._report_rc else "")


@dataclass
class Coverage:
    graph_under_recorded: bool
    graph_replay_fraction: float


def _module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: Any) -> None:
    module = types.ModuleType(name)
    vars(module).update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


@dataclass
class Rig:
    trace_input: Path
    shell: Shell
    agent_calls: list[dict]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """A profile directory beside its capture folder, TraceLens' checkout, and fakes for what needs a real TraceLens."""
    profile = tmp_path / "profile"
    (profile / "torch_trace" / "graph_capture_profile").mkdir(parents=True)
    trace = profile / "torch_trace" / "1.-TP-0.trace.json.gz"
    trace.write_bytes(b"trace")
    (tmp_path / "TraceLens").mkdir()
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "TraceLens"))
    for name in (
        "HYPERLOOM_TRACE_ANALYSIS_ROUTE",
        "INFERENCE_OPTIMIZER_STEADY_STATE_MODE",
        "CONC",
        "RANDOM_RANGE_RATIO",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OSL", "64")
    shell = Shell()
    monkeypatch.setattr(report, "run_with_session_kill", shell)
    monkeypatch.setattr(report, "resolve_arch_json_path", lambda platform: tmp_path / f"{platform}.json")
    _module(
        monkeypatch,
        "hyperloom.orchestrator.trace_analysis.tracelens_analysis",
        discover_trace_inputs=lambda path: ("capture_dir", [trace]),
        count_gpu_kernel_events=lambda path: 15940,
    )
    _module(monkeypatch, "TraceLens.Agent.Analysis.utils", check_graph_replay_coverage=lambda csv: Coverage(False, 0.0))
    agent_calls: list[dict] = []

    async def agent(payload: dict, *, session_dir: Path) -> dict:
        agent_calls.append(payload)
        return {"status": "ok", "tool": "tracelens_analysis"}

    monkeypatch.setattr(roofline, "trace_analyze_handler", agent)
    return Rig(profile, shell, agent_calls)


async def _run(tmp_path: Path, params: dict[str, Any], rig: Rig) -> tuple[JobStatus, ToolResultPointer, dict]:
    sink = RecordingSink()
    jobs = ToolJobs([tracelens_tool()], sink, tmp_path / "jobs", push_delay_seconds=0)
    request = {"trace_input": str(rig.trace_input), "commit_id": "c9", "profile_request_id": "r1", **params}
    job = jobs.submit(ToolRequest("tracelens", "b1", request), job_id="j1")
    status = await jobs.wait(job)
    record = json.loads((tmp_path / "jobs" / "j1" / "result.json").read_text(encoding="utf-8"))
    return status, sink.pointers[0], record


async def test_by_default_no_agent_runs_and_the_pointer_carries_the_ops_table_shares(rig: Rig, tmp_path: Path) -> None:
    status, pointer, record = await _run(tmp_path, {}, rig)
    assert status is JobStatus.SUCCEEDED and rig.agent_calls == []
    assert dict(pointer.op_shares) == SHARES
    assert (pointer.commit_id, pointer.profile_request_id) == ("c9", "r1")
    split, perf = rig.shell.commands
    assert split[split.index("-o") + 1].endswith("/tracelens/trace_split")
    assert {"--find-steady-state", "--llm-inference"} <= set(split) and split[split.index("--OSL") + 1] == "64"
    assert perf[perf.index("--profile_json_path") + 1].endswith(CHUNK)
    assert perf[perf.index("--capture_folder") + 1].endswith("torch_trace/graph_capture_profile")
    assert perf[perf.index("--gpu_arch_json_path") + 1].endswith("MI355X.json")
    assert {"--group_by_parent_module", "--enable_pseudo_ops", "--include_call_stack"} <= set(perf)
    result = record["result"]
    assert (result["route"], result["steady_state_mode"], result["graph_under_recorded"]) == ("report", "mixed", False)
    assert Path(result["ops_summary_path"]).read_text(encoding="utf-8") == OPS_SUMMARY  # copied next to result.json


@pytest.mark.parametrize(
    ("params", "env"),
    [
        ({"analysis_route": "agent"}, None),
        ({"analysis_route": "bypass"}, None),
        ({"analysis_route": "AGENT"}, None),
        ({}, "agent"),
        ({"framework": "xdit"}, None),
    ],
)
async def test_the_orchestrators_own_routes_still_run_the_orchestrators_handler(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict, env: str | None
) -> None:
    if env:
        monkeypatch.setenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", env)
    status, _, record = await _run(tmp_path, params, rig)
    assert status is JobStatus.SUCCEEDED and record["result"]["tool"] == "tracelens_analysis"
    assert [call.get("analysis_route") for call in rig.agent_calls] == [params.get("analysis_route")]
    assert rig.shell.commands == []


@pytest.mark.parametrize(
    ("params", "env"),
    [({"analysis_route": "report"}, "agent"), ({"analysis_route": "report", "framework": "xdit"}, None)],
)
async def test_a_request_naming_the_report_route_gets_it_whatever_else_would_have_chosen(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict, env: str | None
) -> None:
    if env:
        monkeypatch.setenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", env)
    status, pointer, _ = await _run(tmp_path, params, rig)
    assert status is JobStatus.SUCCEEDED and rig.agent_calls == [] and dict(pointer.op_shares) == SHARES


async def test_a_graph_collapsed_report_gets_the_degraded_fallback_and_no_shares(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fallback(csv: Path, fraction: float) -> str:
        return f"# Deterministic Fallback Analysis {fraction}\n"

    def analysis_json(md: Path) -> Path:
        out = md.with_name("analysis.json")
        out.write_text(json.dumps(ANALYSIS_JSON), encoding="utf-8")
        return out

    _module(
        monkeypatch, "TraceLens.Agent.Analysis.utils", check_graph_replay_coverage=lambda csv: Coverage(True, 0.362)
    )
    _module(monkeypatch, "TraceLens.Agent.Analysis.utils.deterministic_fallback", render_fallback_report=fallback)
    _module(monkeypatch, "TraceLens.Agent.Analysis.post_processing", render_analysis_json=analysis_json)
    status, pointer, record = await _run(tmp_path, {}, rig)
    assert status is JobStatus.SUCCEEDED and not pointer.op_shares and DEGRADED_NOTICE in pointer.summary
    assert record["result"]["graph_under_recorded"] is True
    assert Path(record["result"]["analysis_md_path"]).read_text(encoding="utf-8").startswith("# Deterministic Fallback")


@pytest.mark.parametrize(
    ("params", "shell", "error_class"),
    [
        ({"steady_state_mode": "everything"}, Shell(), "invalid_steady_state_mode"),
        ({"split_conc": "many"}, Shell(), "invalid_split_hint"),
        ({}, Shell(chunk=None), "steady_state_chunk_missing"),
        ({}, Shell(report_rc=2), "tracelens_command_failed"),
    ],
)
async def test_a_step_that_fails_fails_the_job_with_its_reason(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict, shell: Shell, error_class: str
) -> None:
    monkeypatch.setattr(report, "run_with_session_kill", shell)
    status, pointer, record = await _run(tmp_path, params, rig)
    assert status is JobStatus.FAILED and not pointer.op_shares and rig.agent_calls == []
    assert record["result"]["error_class"] == error_class and record["result"]["error"] in pointer.summary
