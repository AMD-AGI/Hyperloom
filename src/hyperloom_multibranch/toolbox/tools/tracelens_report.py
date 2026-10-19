# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``report`` route of the ``tracelens`` tool: TraceLens' deterministic analysis of a serving trace, with no agent.

The agent route prepares the trace the same way before its model reads anything: it picks the rank trace that holds GPU
kernels, cuts a steady-state window out of it with TraceLens' splitter, finds the graph-capture folder beside the trace
and runs TraceLens' inference perf report on the window with that folder. The per-op table the op shares come from
(``perf_report_csvs/ops_summary.csv``) is that report's output; the model only writes prose over it, which takes about
half an hour. This route stops at the report: the same split, the same report command, and TraceLens' own graph-replay
check on the result. A report the check marks graph-collapsed gets TraceLens' degraded fallback ``analysis.md`` and
``analysis.json``, which is how :func:`~hyperloom_multibranch.toolbox.tools.op_shares.degraded_reason` knows to withhold
its shares. Without a model it takes about a minute.

It is for LLM serving traces (SGLang, vLLM, ATOM): the splitter reads their step annotations. Anything else asks for
``analysis_route: agent``.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer.session import paths as session_paths
from hyperloom.orchestrator.actions.executors._subprocess_kill import run_with_session_kill
from hyperloom.orchestrator.trace_analysis.tracelens_arch_benchmark import resolve_arch_json_path
from hyperloom.orchestrator.trace_analysis.tracelens_skill_runner import discover_capture_folder

if TYPE_CHECKING:
    from TraceLens.Agent.Analysis.utils import GraphReplayCoverage

REPORT_ROUTE = "report"

#: The platform ``tracelens_analysis`` assumes when it is not told one.
DEFAULT_PLATFORM = "MI355X"
_SPLITTER = "TraceLens.TraceUtils.split_trace.main"
_PERF_REPORT = "TraceLens.Reporting.generate_perf_report_pytorch_inference"
_SPLIT_NUM_STEPS = 32
_DEFAULT_MODE = "mixed"
#: Steady-state mode -> the file name prefix the splitter writes its window under.
_CHUNK_PREFIX = {
    "mixed": "mixed_steady_state",
    "decode_only": "decode_only_steady_state",
    "prefilldecode": "prefilldecode_steady_state",
    "generic": "steady_state",
}
_CHUNK_SUFFIXES = ("trace.json.gz", "json.gz", "trace.json", "json")
#: Trace candidates opened looking for GPU kernels; the likeliest one is first, so more than a few is a real problem.
_PROBE_LIMIT = 8
#: The splitter's hints and where each comes from: the request's own key, else the workload's environment variable.
_SPLIT_HINTS = (
    ("--CONC", "split_conc", "CONC"),
    ("--OSL", "split_osl", "OSL"),
    ("--R", "split_r", "RANDOM_RANGE_RATIO"),
)
_CAPTURE_KEYS = ("capture_folder", "graph_capture_path", "capture_folder_path")
_TAIL_CHARS = 1500


class _Refused(Exception):
    """A step of the route failed; ``error_class`` names which, the message is the reason the branch is told."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class _Run:
    """What one report run works in: the TraceLens checkout, the output directory, the per-command timeout."""

    tracelens_root: Path
    out_dir: Path
    timeout_sec: int

    @property
    def csv_dir(self) -> Path:
        return self.out_dir / "perf_report_csvs"

    @property
    def log(self) -> Path:
        return self.out_dir / "report.log"


def _text(payload: Mapping[str, Any], *keys: str) -> str:
    """The first of ``keys`` the payload sets, as stripped text; empty when none is."""
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _sh(run: _Run, step: str, cmd: list[str]) -> None:
    """Run ``cmd`` in the TraceLens checkout, in its own session reaped on every exit path; its output goes to the log."""
    try:
        done = run_with_session_kill(cmd, cwd=str(run.tracelens_root), timeout=run.timeout_sec, text=True)
    except subprocess.TimeoutExpired:
        raise _Refused("subprocess_timeout", f"{step} timed out after {run.timeout_sec}s") from None
    with run.log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(cmd)}\n{done.stdout}{done.stderr}\n[exit_code] {done.returncode}\n")
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "")[-_TAIL_CHARS:].strip()
        raise _Refused("tracelens_command_failed", f"{step} exited {done.returncode}: {tail}")


def _analysed_trace(trace_input: Path) -> Path:
    """The trace of the input that holds GPU kernels: the file itself, or the first trace of a directory that does."""
    from hyperloom.orchestrator.trace_analysis.tracelens_analysis import count_gpu_kernel_events, discover_trace_inputs

    try:
        _, files = discover_trace_inputs(trace_input)
    except FileNotFoundError as exc:
        raise _Refused("trace_input_missing", str(exc)) from None
    for candidate in files[:_PROBE_LIMIT]:
        if count_gpu_kernel_events(candidate):
            return candidate
    raise _Refused(
        "trace_without_gpu_kernels", f"none of the first {_PROBE_LIMIT} traces under {trace_input} has a GPU kernel"
    )


def _split_command(trace: Path, split_dir: Path, payload: Mapping[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", _SPLITTER, str(trace), "-o", str(split_dir), "--find-steady-state"]
    cmd += ["--num-steps", str(_SPLIT_NUM_STEPS), "--llm-inference"]
    for flag, key, env_name in _SPLIT_HINTS:
        hint = _text(payload, key) or os.environ.get(env_name, "").strip()
        if not hint:
            continue
        try:
            float(hint)
        except ValueError:
            raise _Refused("invalid_split_hint", f"{key}={hint!r} is not a number") from None
        cmd += [flag, hint]
    return cmd


def _steady_state_chunk(split_dir: Path, mode: str) -> Path:
    """The window the splitter wrote for ``mode``."""
    for suffix in _CHUNK_SUFFIXES:
        found = sorted(split_dir.rglob(f"{_CHUNK_PREFIX[mode]}_*.{suffix}"))
        if found:
            return found[0]
    written = sorted(path.name for path in split_dir.iterdir())[:8] if split_dir.is_dir() else []
    raise _Refused(
        "steady_state_chunk_missing", f"the splitter wrote no {mode} steady-state window under {split_dir}: {written}"
    )


def _perf_report_command(chunk: Path, run: _Run, arch: Path | None, capture: Path | None) -> list[str]:
    """TraceLens' inference perf report of the window, as its analysis skill runs it (graph-capture variant when a folder exists)."""
    cmd = [sys.executable, "-m", _PERF_REPORT, "--profile_json_path", str(chunk)]
    if arch is not None:
        cmd += ["--gpu_arch_json_path", str(arch)]
    cmd += ["--group_by_parent_module", "--enable_pseudo_ops", "--group_by_num_kernels", "--include_call_stack"]
    cmd += ["--output_xlsx_path", str(run.out_dir / "perf_report.xlsx"), "--output_csvs_dir", str(run.csv_dir)]
    if capture is not None:
        cmd += ["--capture_folder", str(capture)]
    return cmd


def _graph_replay_coverage(run: _Run) -> GraphReplayCoverage:
    """TraceLens' verdict on whether the perf report's work hides behind graph-replay launches."""
    from TraceLens.Agent.Analysis.utils import check_graph_replay_coverage

    unified = run.csv_dir / "unified_perf_summary.csv"
    if not unified.is_file() or not (run.csv_dir / "ops_summary.csv").is_file():
        raise _Refused(
            "perf_report_incomplete",
            f"the perf report wrote no unified_perf_summary.csv and ops_summary.csv in {run.csv_dir}",
        )
    return check_graph_replay_coverage(unified)


def _write_degraded_report(run: _Run, coverage: GraphReplayCoverage) -> None:
    """TraceLens' degraded fallback ``analysis.md`` and the ``analysis.json`` rendered from it: what the agent route writes."""
    from TraceLens.Agent.Analysis.post_processing import render_analysis_json
    from TraceLens.Agent.Analysis.utils.deterministic_fallback import render_fallback_report

    report = run.out_dir / "analysis.md"
    unified = run.csv_dir / "unified_perf_summary.csv"
    report.write_text(render_fallback_report(unified, coverage.graph_replay_fraction), encoding="utf-8")
    render_analysis_json(report)


def _analysed_trace(trace_input: Path) -> Path:
    """The trace of the input that holds GPU kernels: the file itself, or the first trace of a directory that does."""
    from hyperloom.orchestrator.trace_analysis.tracelens_analysis import count_gpu_kernel_events, discover_trace_inputs

    try:
        _, files = discover_trace_inputs(trace_input)
    except FileNotFoundError as exc:
        raise _Refused("trace_input_missing", str(exc)) from None
    for candidate in files[:_PROBE_LIMIT]:
        if count_gpu_kernel_events(candidate):
            return candidate
    raise _Refused(
        "trace_without_gpu_kernels", f"none of the first {_PROBE_LIMIT} traces under {trace_input} has a GPU kernel"
    )


def _split_command(trace: Path, split_dir: Path, payload: Mapping[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", _SPLITTER, str(trace), "-o", str(split_dir), "--find-steady-state"]
    cmd += ["--num-steps", str(_SPLIT_NUM_STEPS), "--llm-inference"]
    for flag, key, env_name in _SPLIT_HINTS:
        hint = _text(payload, key) or os.environ.get(env_name, "").strip()
        if not hint:
            continue
        try:
            float(hint)
        except ValueError:
            raise _Refused("invalid_split_hint", f"{key}={hint!r} is not a number") from None
        cmd += [flag, hint]
    return cmd


def _steady_state_chunk(split_dir: Path, mode: str) -> Path:
    """The window the splitter wrote for ``mode``."""
    for suffix in _CHUNK_SUFFIXES:
        found = sorted(split_dir.rglob(f"{_CHUNK_PREFIX[mode]}_*.{suffix}"))
        if found:
            return found[0]
    written = sorted(path.name for path in split_dir.iterdir())[:8] if split_dir.is_dir() else []
    raise _Refused(
        "steady_state_chunk_missing", f"the splitter wrote no {mode} steady-state window under {split_dir}: {written}"
    )


def _perf_report_command(chunk: Path, run: _Run, arch: Path | None, capture: Path | None) -> list[str]:
    """TraceLens' inference perf report of the window, as its analysis skill runs it (graph-capture variant when a folder exists)."""
    cmd = [sys.executable, "-m", _PERF_REPORT, "--profile_json_path", str(chunk)]
    if arch is not None:
        cmd += ["--gpu_arch_json_path", str(arch)]
    cmd += ["--group_by_parent_module", "--enable_pseudo_ops", "--group_by_num_kernels", "--include_call_stack"]
    cmd += ["--output_xlsx_path", str(run.out_dir / "perf_report.xlsx"), "--output_csvs_dir", str(run.csv_dir)]
    if capture is not None:
        cmd += ["--capture_folder", str(capture)]
    return cmd


def _check_graph_replay(run: _Run) -> GraphReplayCoverage:
    """TraceLens' verdict on whether the report's work hides behind graph-replay launches.

    A collapsed report gets TraceLens' degraded fallback ``analysis.md`` and the ``analysis.json`` rendered from it, the
    two files the agent route writes in that case.
    """
    from TraceLens.Agent.Analysis.post_processing import render_analysis_json
    from TraceLens.Agent.Analysis.utils import check_graph_replay_coverage
    from TraceLens.Agent.Analysis.utils.deterministic_fallback import render_fallback_report

    unified = run.csv_dir / "unified_perf_summary.csv"
    if not unified.is_file() or not (run.csv_dir / "ops_summary.csv").is_file():
        raise _Refused(
            "perf_report_incomplete",
            f"the perf report wrote no unified_perf_summary.csv and ops_summary.csv in {run.csv_dir}",
        )
    coverage = check_graph_replay_coverage(unified)
    if coverage.graph_under_recorded:
        report = run.out_dir / "analysis.md"
        report.write_text(render_fallback_report(unified, coverage.graph_replay_fraction), encoding="utf-8")
        render_analysis_json(report)
    return coverage


def _analyse(payload: Mapping[str, Any], session_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    trace_input = Path(str(payload["trace_input"])).expanduser().resolve()
    mode = _text(payload, "steady_state_mode") or os.environ.get("INFERENCE_OPTIMIZER_STEADY_STATE_MODE", "").strip()
    mode = mode or _DEFAULT_MODE
    if mode not in _CHUNK_PREFIX:
        raise _Refused("invalid_steady_state_mode", f"steady_state_mode {mode!r} is not one of {sorted(_CHUNK_PREFIX)}")
    root = session_paths.tracelens_root()
    if not root.is_dir():
        raise _Refused("tracelens_root_missing", f"TraceLens root not found: {root}; set TRACELENS_ROOT to a checkout")
    budget = _text(payload, "budget_minutes")
    run = _Run(root, session_dir / "tracelens", int(float(budget or 60) * 60))
    run.out_dir.mkdir(parents=True, exist_ok=True)

    trace = _analysed_trace(trace_input)
    _sh(run, "split", _split_command(trace, run.out_dir / "trace_split", payload))
    chunk = _steady_state_chunk(run.out_dir / "trace_split", mode)
    capture_text = _text(payload, *_CAPTURE_KEYS)
    capture = (
        Path(capture_text).expanduser().resolve() if capture_text else discover_capture_folder(trace_input, [trace])
    )
    arch = resolve_arch_json_path(_text(payload, "target_platform") or DEFAULT_PLATFORM)
    _sh(run, "perf report", _perf_report_command(chunk, run, arch, capture))
    coverage = _graph_replay_coverage(run)
    if coverage.graph_under_recorded:
        _write_degraded_report(run, coverage)

    warnings = [] if arch is not None else [{"code": "gpu_arch_json_unresolved", "severity": "warning"}]
    return {
        "status": "ok",
        "tool": "tracelens_report",
        "route": REPORT_ROUTE,
        "trace_report_path": str(run.out_dir / "perf_report.xlsx"),
        "analysis_md_path": str(run.out_dir / "analysis.md") if coverage.graph_under_recorded else "",
        "ops_summary_path": str(run.csv_dir / "ops_summary.csv"),
        "trace_input": str(trace_input),
        "analysed_trace": trace.name,
        "steady_state_mode": mode,
        "steady_state_chunk": chunk.name,
        "capture_folder": str(capture or ""),
        "graph_replay_fraction": round(coverage.graph_replay_fraction, 4),
        "graph_under_recorded": coverage.graph_under_recorded,
        "trace_health_warnings": warnings,
        "duration_sec": round(time.monotonic() - started, 3),
    }


async def report_handler(payload: dict, *, session_dir: Path) -> dict[str, Any]:
    """Run the report route on ``payload['trace_input']``; the result names where the ops table is, like the agent route's.

    The work runs on a thread so the reactor stays free; a cancelled job still ends the commands' process groups, because
    the thread inherits the job's cancel scope.
    """
    try:
        return await asyncio.to_thread(_analyse, payload, session_dir)
    except _Refused as exc:
        return {
            "status": "failed",
            "tool": "tracelens_report",
            "route": REPORT_ROUTE,
            "error_class": exc.error_class,
            "error": str(exc),
        }


__all__ = ["DEFAULT_PLATFORM", "REPORT_ROUTE", "report_handler"]
