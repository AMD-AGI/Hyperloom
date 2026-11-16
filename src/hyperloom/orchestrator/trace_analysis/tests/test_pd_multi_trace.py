# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A PD profile analysed as one combine_by_node multi-trace TraceLens run, gated per node."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.trace_analysis import _pd_gates as gates
from hyperloom.orchestrator.trace_analysis import _pd_jobs as jobs_mod
from hyperloom.orchestrator.trace_analysis import _trace_gates
from hyperloom.orchestrator.trace_analysis import tracelens_analysis as tla
from hyperloom.orchestrator.trace_analysis import tracelens_skill_runner as tlr
from hyperloom.orchestrator.trace_analysis._analysis_json import _member_to_candidate
from hyperloom.orchestrator.trace_analysis._analysis_run import start_run
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute
from TraceLens.Agent.Analysis.post_processing import render_analysis_json
from TraceLens.Agent.Analysis.utils.config_utils import check_config

_ANALYSIS_MD = """# Model - MI300X Standalone Analysis
<!-- report-begin kind=report_mode mode=agentic -->
<!-- report-end -->

## Executive Summary

Decode and prefill shares.

| Metric | Total | decode_0 | prefill_0 |
|--------|-------|----------|-----------|
| Total Time | 30.0 ms | 20.0 ms | 10.0 ms |
| Compute % | 60.0% | {decode_compute}% | {prefill_compute}% |
| Idle % | 35.0% | {decode_idle}% | {prefill_idle}% |
| Exposed Communication % | 5.0% | 5.0% | 5.0% |
| Top Bottleneck Category | GEMM (50%) | GEMM (60%) | Attention (40%) |

Total is the sum of node times, not wall time.
"""


def _report(tmp_path: Path, *, prefill_idle: float = 15.0, decode_idle: float = 15.0) -> Path:
    """Write a 1P1D ``analysis.md`` with the given idle shares and render ``analysis.json`` beside it."""
    md = tmp_path / "analysis.md"
    md.write_text(
        _ANALYSIS_MD.format(
            decode_idle=decode_idle,
            prefill_idle=prefill_idle,
            decode_compute=100.0 - decode_idle,
            prefill_compute=100.0 - prefill_idle,
        ),
        encoding="utf-8",
    )
    render_analysis_json(md)
    return md


def _candidates() -> list[dict]:
    return [
        {"name": "gemm_decode", "tracelens_node": "decode_0"},
        {"name": "attn_prefill", "tracelens_node": "prefill_0"},
    ]


def _gate(md: Path, policy: str, raw_traces: dict[str, Path] | None = None):
    verdicts = gates.evaluate_node_gates(gates.node_shares(md.with_name("analysis.json")), raw_traces or {}, md)
    return verdicts, gates.gate_pd_candidates(_candidates(), verdicts, policy)


def test_node_shares_ignore_the_total_column(tmp_path: Path) -> None:
    shares = gates.node_shares(_report(tmp_path, prefill_idle=85.0).with_name("analysis.json"))

    assert list(shares) == ["decode_0", "prefill_0"]
    assert (shares["decode_0"].idle_pct, shares["prefill_0"].idle_pct) == (15.0, 85.0)


def test_worst_policy_drops_every_candidate_when_only_prefill_is_idle(tmp_path: Path) -> None:
    _verdicts, (kept, warnings, all_dropped) = _gate(_report(tmp_path, prefill_idle=85.0), gates.PD_GATE_POLICY_WORST)

    assert kept == [] and all_dropped
    idle = [w for w in warnings if w["code"] == "high_gpu_idle_pct"]
    assert [(w["node"], w["idle_pct"]) for w in idle] == [("prefill_0", 85.0)]
    assert "prefill_0" in idle[0]["message"]


def test_worst_policy_keeps_every_candidate_when_both_nodes_are_busy(tmp_path: Path) -> None:
    _verdicts, (kept, warnings, all_dropped) = _gate(_report(tmp_path), gates.PD_GATE_POLICY_WORST)

    assert kept == _candidates() and not all_dropped and warnings == []


def test_an_under_recorded_idle_prefill_is_exempt_and_keeps_every_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefill_raw, decode_raw = tmp_path / "prefill.trace.json.gz", tmp_path / "decode.trace.json.gz"
    monkeypatch.setattr(
        _trace_gates,
        "_graph_coverage_from_raw_trace",
        lambda path: {"graph_under_recorded": True, "graph_launch_count": 7} if path == prefill_raw else {},
    )

    verdicts, (kept, warnings, all_dropped) = _gate(
        _report(tmp_path, prefill_idle=85.0),
        gates.PD_GATE_POLICY_WORST,
        {"prefill_0": prefill_raw, "decode_0": decode_raw},
    )

    assert kept == _candidates() and not all_dropped
    assert [(w["code"], w["node"]) for w in warnings] == [("graph_under_recorded", "prefill_0")]
    assert [v.node for v in verdicts if v.exempt] == ["prefill_0"]


def test_per_node_policy_drops_only_the_idle_nodes_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gates.PD_GATE_POLICY_ENV, "per_node")

    _verdicts, (kept, warnings, all_dropped) = _gate(_report(tmp_path, prefill_idle=85.0), gates.pd_gate_policy())

    assert [c["name"] for c in kept] == ["gemm_decode"] and not all_dropped
    assert [w["node"] for w in warnings] == ["prefill_0"]


def test_gate_policy_defaults_to_worst(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gates.PD_GATE_POLICY_ENV, raising=False)
    assert gates.pd_gate_policy() == gates.PD_GATE_POLICY_WORST
    monkeypatch.setenv(gates.PD_GATE_POLICY_ENV, "bogus")
    assert gates.pd_gate_policy() == gates.PD_GATE_POLICY_WORST


def test_pd_route_outcome_records_every_nodes_shares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gates.PD_GATE_POLICY_ENV, raising=False)
    md = _report(tmp_path, prefill_idle=85.0)
    warnings: list[dict] = []

    outcome = tla._pd_agent_route_outcome(
        tlr.TraceLensSkillRunResult(output_dir=tmp_path, report_path=md, runner="claude_agent_sdk"),
        {},
        warnings,
        tmp_path / "log.txt",
    )

    assert outcome.raw_candidates == []
    shares = next(w for w in warnings if w["code"] == "pd_node_shares")
    assert shares["policy"] == "worst"
    assert shares["nodes"]["prefill_0"]["idle_pct"] == 85.0 and shares["nodes"]["prefill_0"]["failing"]
    assert shares["nodes"]["decode_0"]["idle_pct"] == 15.0 and not shares["nodes"]["decode_0"]["failing"]


def _pd_layout(tmp_path: Path) -> tuple[Path, list[Path], Path, Path]:
    """A torch_trace root with two decode nodes, one prefill node and a decode graph-capture folder."""
    root = tmp_path / "torch_trace"
    decode_capture = root / "decode" / "graph_capture_profile"
    decode_capture.mkdir(parents=True)
    (decode_capture / "DecodeCudaGraphRunner_bs_1_rank0.json.gz").write_bytes(b"x")
    (root / "prefill").mkdir()
    decode_traces = [root / "decode" / f"1-TP-{rank}.trace.json.gz" for rank in (0, 8)]
    prefill_trace = root / "prefill" / "1-TP-0-EXTEND.trace.json.gz"
    for path in [*decode_traces, prefill_trace]:
        path.write_bytes(b"trace")
    return root, decode_traces, prefill_trace, decode_capture


def _fake_splitter(commands: list[list[str]]):
    """A ``run_command`` that writes a decode-only and a generic chunk per split."""

    def _run(cmd, *, cwd, log_path, timeout_s, env=None):
        commands.append(cmd)
        out = Path(cmd[cmd.index("-o") + 1])
        (out / "decode_only_steady_state_0.json.gz").write_bytes(b"decode")
        (out / "steady_state_0.json.gz").write_bytes(b"generic")
        return 0

    return _run


def test_jobs_split_and_clamp_decode_per_node_and_pass_prefill_raw(tmp_path: Path, monkeypatch) -> None:
    root, decode_traces, prefill_trace, decode_capture = _pd_layout(tmp_path)
    commands: list[list[str]] = []
    clamped: list[Path] = []
    monkeypatch.setattr(jobs_mod, "run_command", _fake_splitter(commands))
    monkeypatch.setattr(jobs_mod, "clamp_stream_overlaps", lambda p: clamped.append(p) or {"clamped_events": 0})
    out = tmp_path / "tracelens"

    jobs = jobs_mod.pd_report_jobs(
        {"prefill": [str(prefill_trace)], "decode": [str(p) for p in decode_traces]},
        trace_root=root,
        output_dir=out,
        platform="MI300X",
        tracelens_root=tmp_path,
        log_path=tmp_path / "log.txt",
    )

    decode_chunks = [out / "trace_split" / f"decode_{n}" / "decode_only_steady_state_0.json.gz" for n in (0, 1)]
    assert jobs == [
        {
            "phase": "decode",
            "node": n,
            "trace_path": str(chunk),
            "platform": "MI300X",
            "capture_folder": str(decode_capture),
        }
        for n, chunk in enumerate(decode_chunks)
    ] + [{"phase": "prefill", "node": 0, "trace_path": str(prefill_trace), "platform": "MI300X"}]
    assert clamped == decode_chunks
    split_inputs = [cmd[cmd.index("TraceLens.TraceUtils.split_trace.main") + 1] for cmd in commands]
    assert split_inputs == [str(p) for p in decode_traces]


def test_focus_mode_applies_only_to_the_first_decode_node(tmp_path: Path, monkeypatch) -> None:
    root, decode_traces, prefill_trace, _capture = _pd_layout(tmp_path)
    monkeypatch.setattr(jobs_mod, "run_command", _fake_splitter([]))
    monkeypatch.setattr(jobs_mod, "clamp_stream_overlaps", lambda p: {"clamped_events": 0})
    out = tmp_path / "tracelens"

    jobs = jobs_mod.pd_report_jobs(
        {"decode": [str(p) for p in decode_traces], "prefill": [str(prefill_trace)]},
        trace_root=root,
        output_dir=out,
        platform="MI300X",
        tracelens_root=tmp_path,
        log_path=tmp_path / "log.txt",
        focus_mode="generic",
        health_warnings=[],
    )

    assert Path(jobs[0]["trace_path"]).name == "steady_state_0.json.gz"
    assert Path(jobs[1]["trace_path"]).name == "decode_only_steady_state_0.json.gz"


def test_a_missing_focus_chunk_records_the_n26_retry_hint(tmp_path: Path, monkeypatch) -> None:
    root, decode_traces, prefill_trace, _capture = _pd_layout(tmp_path)
    monkeypatch.setattr(jobs_mod, "run_command", _fake_splitter([]))
    warnings: list[dict] = []

    with pytest.raises(RuntimeError, match="steady_state_chunk_missing"):
        jobs_mod.pd_report_jobs(
            {"decode": [str(decode_traces[0])], "prefill": [str(prefill_trace)]},
            trace_root=root,
            output_dir=tmp_path / "tracelens",
            platform="MI300X",
            tracelens_root=tmp_path,
            log_path=tmp_path / "log.txt",
            focus_mode="mixed",
            health_warnings=warnings,
        )

    assert warnings[0]["code"] == "steady_state_chunk_missing"
    assert "decode_only" in warnings[0]["available_modes"]


def test_multi_trace_run_config_is_combine_by_node_and_passes_check_config(tmp_path: Path) -> None:
    root, decode_traces, prefill_trace, decode_capture = _pd_layout(tmp_path)
    out = tmp_path / "tracelens"
    out.mkdir()
    jobs = [
        {"phase": "decode", "node": 0, "trace_path": str(decode_traces[0]), "platform": "MI300X"},
        {"phase": "prefill", "node": 0, "trace_path": str(prefill_trace), "platform": "MI300X"},
    ]
    jobs[0]["capture_folder"] = str(decode_capture)

    path = tlr.write_multi_trace_run_config(output_dir=out, jobs=jobs, analysis_mode="inference")

    config = yaml.safe_load(path.read_text())
    assert config["aggregation"] == {"policy": "combine_by_node"}
    assert "kernel_summary" not in config
    assert [(job["phase"], job["node"]) for job in config["jobs"]] == [("decode", 0), ("prefill", 0)]
    assert {job["platform"] for job in config["jobs"]} == {"MI300X"}
    assert config["jobs"][0]["capture_folder"] == str(decode_capture.resolve())
    assert "capture_folder" not in config["jobs"][1]
    assert check_config(config, "multi_trace").variables["focus_job"] == "decode_0"


def test_colocated_run_config_is_unchanged(tmp_path: Path) -> None:
    trace = tmp_path / "t.trace.json.gz"
    trace.write_bytes(b"trace")

    path = tlr.write_run_config(
        output_dir=tmp_path, trace_path=trace, platform="MI300X", analysis_mode="inference", capture_folder=None
    )

    assert path.read_text() == (
        f"analysis_mode: inference\noutput_dir: {tmp_path.resolve()}\njobs:\n"
        f"- trace_path: {trace.resolve()}\n  platform: MI300X\n"
    )


def _request(tmp_path: Path, **kwargs) -> tlr.TraceLensSkillRequest:
    return tlr.TraceLensSkillRequest(
        skill_path=tmp_path / "SKILL.md",
        output_dir=tmp_path / "out",
        tracelens_root=tmp_path / "tl",
        tracelens_internal_root=None,
        capture_folder=None,
        **kwargs,
    )


def test_colocated_prompt_is_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_config = (out / "run.yaml").resolve()
    expected = f"""You are running TraceLens standalone analysis for Hyperloom.

Read and follow the FULL instructions in this skill file:
{tmp_path / "SKILL.md"}

Run it from config {run_config}, standalone.
The execution context below answers the remaining Step 0 questions.
Do not ask the user any questions; proceed with the analysis.

Execution context:
- Environment: local
- TraceLens root: {tmp_path / "tl"}
- TraceLens-internal root: (not installed; OSS-only mode)
- TL_EXTENSION: (unset)
- Command prefix cache: {out / "cache" / "cmd_prefix.txt"}
- Run config: {run_config}
- Comparison scope: standalone
- Output directory: {out}

Important requirements:
1. Use the provided command prefix cache for all shell commands; do not rebuild it.
2. Run the analysis-orchestrator workflow through Step 12.
3. Take the trace, platform, analysis mode and capture folder from the run config
   only. If the Step 0 config check fails, stop and report its error.
4. Write all TraceLens outputs under the output directory above.
5. Ensure this file exists before you finish:
   - {out / "analysis.md"}  (TraceLens final report; REQUIRED)
6. Do not run GEAK, kernel optimization, or modify model/framework source.

When complete, respond with a short summary of the artifacts you wrote.
"""
    assert tlr.build_orchestrator_prompt(_request(tmp_path)) == expected


def test_multi_trace_prompt_names_the_scope(tmp_path: Path) -> None:
    prompt = tlr.build_orchestrator_prompt(_request(tmp_path, scope="multi_trace"))

    assert prompt.startswith("You are running TraceLens multi_trace analysis for Hyperloom.")
    assert f"Run it from config {(tmp_path / 'out' / 'run.yaml').resolve()}, multi_trace." in prompt


def test_task_node_is_copied_into_the_candidate() -> None:
    member = {"operation_cell": "gemm", "kernel_name": ["k"], "time_ms": 1.0, "count": 2, "pct_e2e": 3.0}

    assert _member_to_candidate({"operation": "GEMM", "node": "prefill_0"}, member)["tracelens_node"] == "prefill_0"
    assert "tracelens_node" not in _member_to_candidate({"operation": "GEMM", "node": None}, member)


def test_pd_entry_runs_the_agent_once_on_a_multi_trace_config(tmp_path: Path, monkeypatch) -> None:
    root, decode_traces, prefill_trace, _capture = _pd_layout(tmp_path)
    monkeypatch.setattr(jobs_mod, "run_command", _fake_splitter([]))
    monkeypatch.setattr(jobs_mod, "clamp_stream_overlaps", lambda p: {"clamped_events": 0})
    monkeypatch.setattr(tla, "_check_tracelens_checkout", lambda args, log: (tmp_path, None, tmp_path / "SKILL.md"))
    monkeypatch.setattr(tla, "require_claude_sdk", lambda: None)
    monkeypatch.setattr(tla, "_gpu_arch_json", lambda run, **kw: None)
    runs: list[tlr.TraceLensSkillRequest] = []

    def _agent(request, args, log_path):
        runs.append(request)
        md = request.output_dir / "analysis.md"
        md.write_text(_ANALYSIS_MD.format(decode_idle=10, prefill_idle=10, decode_compute=90, prefill_compute=90))
        return tlr.TraceLensSkillRunResult(output_dir=request.output_dir, report_path=md, runner="claude_agent_sdk")

    monkeypatch.setattr(tla, "_run_agent_skill", _agent)
    args = tla._build_parser().parse_args(
        [
            "--trace-input",
            str(decode_traces[0]),
            "--workspace-path",
            str(tmp_path / "ws"),
            "--target-platform",
            "MI300X",
            "--framework",
            "sglang",
            "--steady-state-mode",
            "decode_only_or_generic",
        ]
    )
    run = start_run(args, AnalysisRoute.AGENT)

    result = tla._analyse_pd_traces(
        run,
        json.loads(
            json.dumps(
                {
                    "role_node_traces": {"decode": [str(decode_traces[0])], "prefill": [str(prefill_trace)]},
                    "trace_dir": str(root),
                }
            )
        ),
    )

    assert len(runs) == 1 and runs[0].scope == "multi_trace"
    config = yaml.safe_load((runs[0].output_dir / "run.yaml").read_text())
    assert [(job["phase"], job["node"]) for job in config["jobs"]] == [("decode", 0), ("prefill", 0)]
    assert result.candidates == []
    assert any(w["code"] == "pd_node_shares" for w in run.trace_health_warnings)
