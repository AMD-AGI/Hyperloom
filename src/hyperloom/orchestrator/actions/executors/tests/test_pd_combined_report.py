# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD-disaggregated TraceLens inputs: per-role analysis requests and the combined per-node report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from hyperloom.orchestrator.actions.executors import _pd_combined_report as combined
from hyperloom.orchestrator.actions.executors import profile
from hyperloom.orchestrator.actions.executors import trace_analyze
from hyperloom.orchestrator.actions.executors._pd_role_trace import role_analysis_payload
from hyperloom.orchestrator.trace_analysis import _trace_split as split_module
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute
from hyperloom.orchestrator.trace_analysis.tests.test_stream_overlap_repair import _durations, _kernel, _write


def _overrunning_trace(path: Path) -> None:
    """A trace whose first kernel overruns the same-stream kernel after it."""
    _write(path, [_kernel("index_elementwise_kernel", 1000.0, 533_100.0), _kernel("gemm", 1004.5, 10.0)])


def test_role_requests_split_decode_without_llm_flags_and_leave_prefill_unsplit() -> None:
    decode = role_analysis_payload("decode", "/t/decode/1-TP-0.trace.json.gz", "sglang")
    prefill = role_analysis_payload("prefill", "/t/prefill/1-TP-0.trace.json.gz", "sglang")

    assert decode["steady_state_mode"] == "decode_only_or_generic"
    assert prefill["steady_state_mode"] == "unsplit"
    run = trace_analyze._AnalysisRun(
        trace_input=decode["trace_input"],
        workspace_path="/ws",
        tracelens_root=Path("/tl"),
        route=AnalysisRoute.NON_LLM,
        scriptable=False,
        model_name="m",
        framework="sglang",
        target_platform="MI300X",
    )
    cmd, mode = trace_analyze._build_trace_analyze_cmd(decode, run, session_dir=Path("/s"), workload={"conc": 64})
    assert mode == "decode_only_or_generic"
    assert "--split-llm-inference" not in cmd
    assert "--split-conc" not in cmd
    assert "--clamp-stream-overlaps" in cmd
    plain, _mode = trace_analyze._build_trace_analyze_cmd(
        {"trace_input": decode["trace_input"], "steady_state_mode": "unsplit"}, run, session_dir=Path("/s"), workload={}
    )
    assert "--clamp-stream-overlaps" not in plain


def test_capture_health_check_finds_a_role_capture_folder(tmp_path: Path) -> None:
    capture = tmp_path / "decode" / "graph_capture_profile"
    capture.mkdir(parents=True)
    (capture / "DecodeCudaGraphRunner_bs_1_rank0.json.gz").write_bytes(b"x")

    found, files = profile._capture_trace_files(tmp_path)

    assert found == capture
    assert [f.name for f in files] == ["DecodeCudaGraphRunner_bs_1_rank0.json.gz"]


def test_combined_report_splits_decode_per_node_copies_prefill_and_clamps_each_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "torch_trace"
    decode_capture = root / "decode" / "graph_capture_profile"
    decode_capture.mkdir(parents=True)
    (decode_capture / "DecodeCudaGraphRunner_bs_1_rank0.json.gz").write_bytes(b"x")
    (root / "prefill" / "graph_capture_profile").mkdir(parents=True)
    (root / "prefill" / "graph_capture_profile" / "capture.log").write_bytes(b"x")
    decode_traces = [root / "decode" / f"1-TP-{rank}.trace.json.gz" for rank in (0, 8)]
    prefill_trace = root / "prefill" / "1-TP-0-EXTEND.trace.json.gz"
    for path in [*decode_traces, prefill_trace]:
        _overrunning_trace(path)
    commands: list[list[str]] = []

    def _fake_run(cmd, *, cwd, log_path, timeout_s):
        commands.append(cmd)
        if "TraceLens.TraceUtils.split_trace.main" in cmd:
            out = Path(cmd[cmd.index("-o") + 1])
            _overrunning_trace(out / "steady_state_chunk.json.gz")
            (out / "sglang_iteration_0-15.json.gz").write_bytes(b"whole")
        else:
            config = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
            (Path(config["output_dir"]) / combined.AGGREGATED_CSVS_DIR).mkdir()
        return 0

    monkeypatch.setattr(combined, "run_command", _fake_run)
    monkeypatch.setattr(split_module, "run_command", _fake_run)
    monkeypatch.setenv("TRACELENS_SPLIT_NUM_STEPS", "16")
    out_dir = tmp_path / "round" / "tracelens"

    result = combined.run_pd_combined_report(
        {"decode": [str(p) for p in decode_traces], "prefill": [str(prefill_trace)]},
        combined.PdReportRun(
            trace_root=root,
            output_dir=out_dir,
            platform="MI300X",
            analysis_mode="inference",
            tracelens_root=tmp_path,
        ),
    )

    assert result["status"] == "ok"
    config = yaml.safe_load((out_dir / combined.PD_REPORT_CONFIG_NAME).read_text())
    assert config["analysis_mode"] == "inference"
    assert config["output_dir"] == str(out_dir)
    assert config["aggregation"] == {"policy": "combine_by_node"}
    assert "kernel_summary" not in config
    decode_chunks = [out_dir / "trace_split" / f"decode_{node}" / "steady_state_chunk.json.gz" for node in (0, 1)]
    prefill_chunk = out_dir / "trace_split" / "prefill_0" / prefill_trace.name
    assert config["jobs"] == [
        *(
            {
                "phase": "decode",
                "node": node,
                "trace_path": str(chunk),
                "platform": "MI300X",
                "capture_folder": str(decode_capture),
            }
            for node, chunk in enumerate(decode_chunks)
        ),
        {"phase": "prefill", "node": 0, "trace_path": str(prefill_chunk), "platform": "MI300X"},
    ]
    split_cmds = [cmd for cmd in commands if "TraceLens.TraceUtils.split_trace.main" in cmd]
    assert sorted(cmd[cmd.index("TraceLens.TraceUtils.split_trace.main") + 1] for cmd in split_cmds) == [
        str(p) for p in decode_traces
    ]
    assert all(cmd[-3:] == ["--find-steady-state", "--num-steps", "16"] for cmd in split_cmds)
    assert result["aggregated_csvs"] == str(out_dir / combined.AGGREGATED_CSVS_DIR)
    assert all(_durations(chunk)["index_elementwise_kernel"] == 4.5 for chunk in [*decode_chunks, prefill_chunk])
    assert _durations(prefill_trace)["index_elementwise_kernel"] == 533_100.0


def test_combined_report_fails_soft_when_a_split_overruns_its_budget(tmp_path: Path, monkeypatch) -> None:
    trace = tmp_path / "decode" / "1-TP-0.trace.json.gz"
    trace.parent.mkdir()
    trace.write_bytes(b"trace")

    def _overrun(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=1)

    monkeypatch.setattr(split_module, "run_command", _overrun)

    result = combined.run_pd_combined_report(
        {"decode": [str(trace)]},
        combined.PdReportRun(
            trace_root=tmp_path,
            output_dir=tmp_path / "out",
            platform="MI300X",
            analysis_mode="inference",
            tracelens_root=tmp_path,
        ),
    )

    assert result["status"] == "failed"
    assert "timed out" in result["error"]


def test_combined_report_fails_soft_when_a_decode_split_has_no_chunk(tmp_path: Path, monkeypatch) -> None:
    trace = tmp_path / "decode" / "1-TP-0.trace.json.gz"
    trace.parent.mkdir()
    trace.write_bytes(b"trace")
    monkeypatch.setattr(split_module, "run_command", lambda cmd, **_kwargs: 0)

    result = combined.run_pd_combined_report(
        {"decode": [str(trace)]},
        combined.PdReportRun(
            trace_root=tmp_path,
            output_dir=tmp_path / "out",
            platform="MI300X",
            analysis_mode="inference",
            tracelens_root=tmp_path,
        ),
    )

    assert result["status"] == "failed"
    assert "trace_split_no_steady_state" in result["error"]


def test_combined_report_splits_the_decode_nodes_concurrently_in_node_order(tmp_path: Path, monkeypatch) -> None:
    import threading

    traces = [tmp_path / "decode" / f"1-TP-{rank}.trace.json.gz" for rank in (0, 8, 16)]
    together = threading.Barrier(len(traces), timeout=10)

    def _split(trace_path: Path, split_dir: Path, _phase: str, **_kwargs) -> Path:
        together.wait()
        return split_dir / f"{trace_path.name}.chunk"

    def _report(cmd, **_kwargs):
        config = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
        (Path(config["output_dir"]) / combined.AGGREGATED_CSVS_DIR).mkdir()
        return 0

    monkeypatch.setattr(combined, "_node_chunk", _split)
    monkeypatch.setattr(combined, "run_command", _report)
    out_dir = tmp_path / "out"

    result = combined.run_pd_combined_report(
        {"decode": [str(p) for p in traces]},
        combined.PdReportRun(
            trace_root=tmp_path,
            output_dir=out_dir,
            platform="MI300X",
            analysis_mode="inference",
            tracelens_root=tmp_path,
        ),
    )

    assert result["status"] == "ok"
    assert [(job["node"], job["trace_path"]) for job in result["jobs"]] == [
        (node, str(out_dir / "trace_split" / f"decode_{node}" / f"{trace.name}.chunk"))
        for node, trace in enumerate(traces)
    ]


def test_the_cli_imports_without_tracelens() -> None:
    probe = "import sys; sys.modules['TraceLens'] = None; import hyperloom.inference_optimizer.cli.executors"
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]


def test_combined_report_fails_soft_when_its_output_dir_cannot_be_made(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    result = combined.run_pd_combined_report(
        {"decode": ["/t/1-TP-0.trace.json.gz"]},
        combined.PdReportRun(
            trace_root=tmp_path,
            output_dir=blocker / "tracelens",
            platform="MI300X",
            analysis_mode="inference",
            tracelens_root=tmp_path,
        ),
    )
    assert result["status"] == "failed"
    assert "file" in result["error"]
