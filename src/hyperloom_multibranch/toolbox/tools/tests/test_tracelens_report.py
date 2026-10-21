# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``tracelens`` tool's default route runs TraceLens' split and perf report with no agent; ``agent`` still runs one.

Its op shares are kernel shares of the window's raw GPU events (``data/*_gpu_events.json.gz``, real windows: see
``test_kernel_shares.py``), not TraceLens' aten-op table, which the fake perf report here still writes.
"""

from __future__ import annotations

import json
import shutil
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

DATA = Path(__file__).parent / "data"
SPLITTER = "TraceLens.TraceUtils.split_trace.main"
CHUNK = "mixed_steady_state_prefill_1_prefilldecode_0_decode_31_bs11_conc3_t-TP-0.trace.json.gz"
OPS_SUMMARY = (
    "name,parent_module,total_direct_kernel_time_ms,Percentage (%)\n"
    "aten::mm,A,0.2,20.0\naten::mm,B,0.045,4.5\naten::copy_,C,0.19,19.0\naten::mul,D,0.1,10.0\n"
)
HINTS = {"split_conc": "4", "split_osl": "64", "split_r": "1"}
ANALYSIS_JSON = {"report_info": {"mode": "deterministic-fallback", "trace_quality": "poor", "warnings": ["collapsed"]}}
#: The window the fake splitter writes unless a test names another: a Qwen3-14B-FP8 root prefill step, GEMM-led.
DEFAULT_WINDOW = "root_window"
CK_GEMM = "ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle"


class Shell:
    """Stands in for ``run_with_session_kill``: writes what TraceLens' splitter and perf report would, records the argv."""

    def __init__(self, *, chunk: str | None = CHUNK, report_rc: int = 0, window: str = DEFAULT_WINDOW) -> None:
        self.commands: list[list[str]] = []
        self._chunk, self._report_rc, self._window = chunk, report_rc, window

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.commands.append(cmd)
        assert Path(kwargs["cwd"]).is_dir() and kwargs["timeout"] == 3600
        if SPLITTER in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            out.mkdir(parents=True)
            if self._chunk:
                shutil.copy(DATA / f"{self._window}_gpu_events.json.gz", out / self._chunk)
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
    trace: Path


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """A profile directory beside its capture folder, TraceLens' checkout, and fakes for what needs a real TraceLens."""
    profile = tmp_path / "profile"
    (profile / "torch_trace" / "graph_capture_profile").mkdir(parents=True)
    trace = profile / "torch_trace" / "1.-TP-0.trace.json.gz"
    trace.write_bytes(b"trace")
    (tmp_path / "TraceLens").mkdir()
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "TraceLens"))
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
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
    return Rig(profile, shell, agent_calls, trace)


async def _run(tmp_path: Path, params: dict[str, Any], rig: Rig) -> tuple[JobStatus, ToolResultPointer, dict]:
    sink = RecordingSink()
    jobs = ToolJobs([tracelens_tool()], sink, tmp_path / "jobs", push_delay_seconds=0)
    request = {"trace_input": str(rig.trace_input), "commit_id": "c9", "profile_request_id": "r1", **params}
    job = jobs.submit(ToolRequest("tracelens", "b1", request), job_id="j1")
    status = await jobs.wait(job)
    record = json.loads((tmp_path / "jobs" / "j1" / "result.json").read_text(encoding="utf-8"))
    return status, sink.pointers[0], record


async def test_by_default_no_agent_runs_and_the_pointer_carries_the_windows_kernel_shares(
    rig: Rig, tmp_path: Path
) -> None:
    status, pointer, record = await _run(tmp_path, HINTS, rig)
    assert status is JobStatus.SUCCEEDED and rig.agent_calls == []
    assert max(pointer.op_shares, key=pointer.op_shares.get) == CK_GEMM and "aten::mm" not in pointer.op_shares
    assert sum(pointer.op_shares.values()) == pytest.approx(100.0)
    assert (pointer.commit_id, pointer.profile_request_id) == ("c9", "r1")
    split, perf = rig.shell.commands
    assert split[split.index("-o") + 1].endswith("/tracelens/trace_split")
    assert {"--find-steady-state", "--llm-inference"} <= set(split)
    assert [split[split.index(flag) + 1] for flag in ("--CONC", "--OSL", "--R")] == ["4", "64", "1"]
    assert perf[perf.index("--profile_json_path") + 1].endswith(CHUNK)
    assert perf[perf.index("--capture_folder") + 1].endswith("torch_trace/graph_capture_profile")
    assert perf[perf.index("--gpu_arch_json_path") + 1].endswith("MI355X.json")
    assert {"--group_by_parent_module", "--enable_pseudo_ops", "--include_call_stack"} <= set(perf)
    result = record["result"]
    assert (result["route"], result["steady_state_mode"], result["graph_under_recorded"]) == ("report", "mixed", False)
    assert result["split_hints"] == HINTS
    assert Path(result["ops_summary_path"]).read_text(encoding="utf-8") == OPS_SUMMARY  # copied next to result.json
    table = Path(result["kernel_shares_path"])
    assert table.parent == Path(result["ops_summary_path"]).parent.parent and table.is_file()  # copied back as well
    assert (result["excluded_events"], result["excluded_claimed_ms"], result["window_degraded_reason"]) == (0, 0.0, "")
    assert result["window_gpu_busy_percent"] == 100.0 and DEGRADED_NOTICE not in pointer.summary


async def test_the_splitter_hints_and_the_mode_come_from_the_request_never_from_the_toolbox_process(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real session's toolbox process exports no workload, and one that did would speak for every branch."""
    for name, value in (
        ("CONC", "9"),
        ("OSL", "99"),
        ("RANDOM_RANGE_RATIO", "0.5"),
        ("TRACELENS_SPLIT_CONC", "9"),
        ("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "decode_only"),
    ):
        monkeypatch.setenv(name, value)
    status, _, record = await _run(tmp_path, {}, rig)
    assert status is JobStatus.SUCCEEDED, record
    split, _ = rig.shell.commands
    assert not {"--CONC", "--OSL", "--R"} & set(split)
    assert record["result"]["split_hints"] == {} and record["result"]["steady_state_mode"] == "mixed"


@pytest.mark.parametrize(
    ("params", "hints"),
    [
        ({"split_conc": 8}, {"split_conc": "8"}),
        (
            {"split_conc": "8.0", "split_osl": "128.5", "split_r": "0"},
            {"split_conc": "8", "split_osl": "128.5", "split_r": "0"},
        ),
        ({"split_osl": "", "split_r": None}, {}),
    ],
)
async def test_a_hint_is_passed_as_the_splitter_takes_it(
    rig: Rig, tmp_path: Path, params: dict, hints: dict[str, str]
) -> None:
    status, _, record = await _run(tmp_path, params, rig)
    assert status is JobStatus.SUCCEEDED and record["result"]["split_hints"] == hints
    split, _ = rig.shell.commands
    assert sum(flag in split for flag in ("--CONC", "--OSL", "--R")) == len(hints)


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
    assert status is JobStatus.SUCCEEDED and rig.agent_calls == [] and CK_GEMM in pointer.op_shares


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


async def _with_window(
    tmp_path: Path, rig: Rig, monkeypatch: pytest.MonkeyPatch, window: str, *, trace: str = ""
) -> Any:
    """Run the report on the real ``window`` (and, when a test prices excluded kernels, the rank trace ``trace``)."""
    monkeypatch.setattr(report, "run_with_session_kill", Shell(window=window))
    if trace:
        shutil.copy(DATA / f"{trace}_gpu_events.json.gz", rig.trace)
    return await _run(tmp_path, HINTS, rig)


async def test_a_window_whose_graph_replay_kernels_are_missing_from_the_trace_gives_no_shares_and_says_why(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Qwen3-14B-FP8 branch profile: its one decode step is a replay with no kernels, TraceLens skips it with a warning.

    The window holds the four eager ops around it (11 kernels, 0.05 ms) in an 8.2 ms span, no GEMM, and TraceLens' own
    check passes it (0% of the time is in replay wrapper ops); only the window's own events show 0.6% of the span covered.
    """
    status, pointer, record = await _with_window(tmp_path, rig, monkeypatch, "b06_window")
    assert status is JobStatus.SUCCEEDED and not pointer.op_shares and not pointer.commit_id
    result = record["result"]
    assert result["graph_under_recorded"] is False and result["window_gpu_busy_percent"] == 0.61
    assert "0.6% of its 8.22 ms GPU span" in result["window_degraded_reason"] and result["kernel_shares_path"] == ""
    assert DEGRADED_NOTICE in pointer.summary and result["window_degraded_reason"] in pointer.summary


@pytest.mark.parametrize(
    ("window", "busy", "top", "top_share"),
    [
        ("root_window", 100.0, CK_GEMM, 87.08),
        ("qwen3_8b_window", 31.28, "at::native::vectorized_elementwise_kernel", 37.34),
        ("qwen3_8b_tp2_window", 26.57, "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs", 28.62),
    ],
)
async def test_a_real_window_whose_kernels_run_one_after_another_keeps_its_kernel_shares(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str, busy: float, top: str, top_share: float
) -> None:
    status, pointer, record = await _with_window(tmp_path, rig, monkeypatch, window)
    assert status is JobStatus.SUCCEEDED and DEGRADED_NOTICE not in pointer.summary
    result = record["result"]
    assert (result["window_gpu_busy_percent"], result["excluded_events"]) == (busy, 0)
    assert result["window_degraded_reason"] == "" and max(pointer.op_shares, key=pointer.op_shares.get) == top
    assert pointer.op_shares[top] == pytest.approx(top_share, abs=0.005)
    assert sum(pointer.op_shares.values()) == pytest.approx(100.0, abs=0.01)


async def test_a_decode_window_with_eager_kernels_stamped_inside_the_graph_replay_gets_kernel_shares_and_a_count(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Qwen3-14B-FP8 branch b40 (profile 2b7acc33): the brief's top four were copy_, index_put, greedy_sample and add.

    The profiler stamps six eager kernels after the one replay as lasting 9 ms each (54.84 ms in all, 50 to 80 us really),
    on the stream the replay's 604 kernels run on. They are left out and counted; the FlyDSL GEMM, the fp8 quantisation and
    paged attention lead a list that sums to 100% of the replay's 7.93 ms.
    """
    status, pointer, record = await _with_window(tmp_path, rig, monkeypatch, "b40_window", trace="b40_rank_trace")
    assert status is JobStatus.SUCCEEDED and DEGRADED_NOTICE not in pointer.summary
    result = record["result"]
    assert (result["excluded_events"], result["excluded_claimed_ms"], result["window_degraded_reason"]) == (
        6,
        54.84,
        "",
    )
    assert (result["window_kernel_ms"], result["excluded_estimated_ms"]) == (7.93, 0.027)
    assert list(pointer.op_shares)[:3] == [
        "kernel_gemm_0",
        "aiter::dynamic_per_token_scaled_quant_kernel",
        "paged_attention_ll4mi_QKV_mfma16_kernel",
    ]
    assert sum(pointer.op_shares.values()) == pytest.approx(100.0, abs=0.01)
    assert "6 GPU events the profiler misreported (claiming 54.84 ms) are left out of the shares" in pointer.summary
    assert (
        Path(result["kernel_shares_path"]).read_text(encoding="utf-8").splitlines()[1].startswith("kernel_gemm_0,44.1")
    )


async def test_excluded_events_of_a_kernel_the_trace_never_ran_clean_degrade_the_window_it_does_not_rank_a_guess(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same window, but the rank trace it was cut from holds no undisturbed ``greedy_sample``: no price, no shares."""
    status, pointer, record = await _with_window(tmp_path, rig, monkeypatch, "b40_window", trace="root_window")
    assert status is JobStatus.SUCCEEDED and not pointer.op_shares and not pointer.commit_id
    result = record["result"]
    assert "no undisturbed instance of " in result["window_degraded_reason"] and result["kernel_shares_path"] == ""
    assert result["excluded_events"] == 6 and DEGRADED_NOTICE in pointer.summary


@pytest.mark.parametrize("content", [b"chunk", b""])
async def test_a_window_that_cannot_be_read_fails_the_report_it_does_not_pass_it(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    class Unreadable(Shell):
        def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
            done = super().__call__(cmd, **kwargs)
            if SPLITTER in cmd:
                (Path(cmd[cmd.index("-o") + 1]) / CHUNK).write_bytes(content)
            return done

    monkeypatch.setattr(report, "run_with_session_kill", Unreadable())
    status, pointer, record = await _run(tmp_path, HINTS, rig)
    assert status is JobStatus.FAILED and not pointer.op_shares
    assert record["result"]["error_class"] == "kernel_trace_unreadable"


async def test_a_perf_report_without_the_tables_the_coverage_check_reads_fails_the_report(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoTables(Shell):
        def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
            done = super().__call__(cmd, **kwargs)
            if SPLITTER not in cmd:
                (Path(cmd[cmd.index("--output_csvs_dir") + 1]) / "ops_summary.csv").unlink()
            return done

    monkeypatch.setattr(report, "run_with_session_kill", NoTables())
    status, pointer, record = await _run(tmp_path, HINTS, rig)
    assert status is JobStatus.FAILED and not pointer.op_shares
    assert record["result"]["error_class"] == "perf_report_incomplete"


@pytest.mark.parametrize(
    ("params", "shell", "error_class"),
    [
        ({"steady_state_mode": "everything"}, Shell(), "invalid_steady_state_mode"),
        ({"split_conc": "many"}, Shell(), "invalid_split_hint"),
        ({"split_conc": "4.5"}, Shell(), "invalid_split_hint"),
        ({"split_osl": "0"}, Shell(), "invalid_split_hint"),
        ({"split_r": "1.5"}, Shell(), "invalid_split_hint"),
        ({"split_osl": "nan"}, Shell(), "invalid_split_hint"),
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
