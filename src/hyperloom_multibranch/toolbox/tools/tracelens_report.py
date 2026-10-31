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

The splitter's workload hints (``--CONC``, ``--OSL``, ``--R``), the prompt length the op shares are weighted by (``ISL``) and
the steady-state mode come from the request alone. A specialist's request gets the hints from the profile test's own point
(``toolbox/analysis.py``), not from anything the caller or the toolbox process says; the splitter picks its window without a
hint the request lacks.

The op shares do not come from TraceLens' ops table. Its durations are wrong on a decode window after a graph replay (the
ROCm profiler stamps the eager kernels queued behind the replay as lasting ~9 ms each), so the shares come from GPU events,
per kernel, summing to 100% of the kernel time they are taken over (:mod:`~.kernel_shares`): events that overlap the previous
event of their in-order stream are left out and counted (``excluded_events``, ``excluded_claimed_ms``), and a window is
reported degraded, with the reason and no shares, when its kernels are missing from the trace (they fill under
:data:`~.kernel_shares.MIN_GPU_BUSY_PERCENT` of its GPU span) or the excluded events could change the ranking.

They are not the perf report's window's either. That window (``steady_state_mode``, ``mixed`` unless asked otherwise) is
weighted by whatever steps it holds, and a graded wave is mostly decode, so the shares are those of the trace's decode window
and its prefill window, weighted by the time each phase takes in a wave of the profile point's workload (:mod:`~.phase_mix`:
``phase_fractions``, ``phase_windows``). TraceLens' report is still run on the perf report's window, and its own check on the
result still marks a graph-collapsed one degraded.

A kernel name that several modules launch (the four FlyDSL GEMMs of a layer are all ``kernel_gemm_0``) is split by the module,
``kernel_gemm_0 [Qwen2MLP.RowParallelLinear]`` (:mod:`~.kernel_launchers`): an eager kernel's launch call and the module frames
around it are in the window; a graph replay's are in the graph-capture folder beside the trace, which TraceLens also takes.
A replay that cannot be paired with its capture keeps its plain names, and ``launcher_notes`` says why.

It is for LLM serving traces (SGLang, vLLM, ATOM): the splitter reads their step annotations. Anything else asks for
``analysis_route: agent``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.outside_facts import ENV_CONC, ENV_ISL, ENV_OSL, ENV_RANDOM_RANGE_RATIO
from hyperloom.inference_optimizer.session import paths as session_paths
from hyperloom.measure.profile_traces import PROFILE_DIR
from hyperloom.measure.ruppell import CHECK_FILE, MODULE_SHARES_FILE, UNATTRIBUTED
from hyperloom.measure.ruppell import OUTPUT_DIR as RUPPELL_DIR
from hyperloom.orchestrator.actions.executors import RoundWatch, run_with_session_kill
from hyperloom.orchestrator.trace_analysis import count_gpu_kernel_events, discover_trace_inputs, is_capture_fragment
from hyperloom.orchestrator.trace_analysis.tracelens_arch_benchmark import resolve_arch_json_path
from hyperloom.orchestrator.trace_analysis.tracelens_skill_runner import discover_capture_folder
from hyperloom_multibranch.toolbox.tools.kernel_launchers import Relabelled, read_launched_window
from hyperloom_multibranch.toolbox.tools.kernel_shares import (
    KERNEL_SHARES_FILE,
    KernelTraceError,
    KernelWindow,
    kernel_window,
    read_gpu_events,
    write_kernel_table,
)
from hyperloom_multibranch.toolbox.tools.op_shares import (
    EXCLUDED_CLAIMED_KEY,
    EXCLUDED_EVENTS_KEY,
    KERNEL_SHARES_KEY,
    MODULE_SHARES_KEY,
    OWN_DEGRADED_KEY,
)
from hyperloom_multibranch.toolbox.tools.phase_mix import (
    PHASE_MODES,
    PhaseWindow,
    Workload,
    graded_mix,
    mix_fields,
    rank_trace_reader,
    read_phase_window,
    split_steps,
    write_mix_table,
)

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
#: The splitter's hints: its flag, the request key that carries it, and the benchmark environment variable of the
#: profile test's own point it is measured from (the server reads that, see ``toolbox/analysis.py``). The route reads the
#: request only, never the process environment: a toolbox serving many branches has no workload of its own.
SPLIT_HINTS = (
    ("--CONC", "split_conc", ENV_CONC),
    ("--OSL", "split_osl", ENV_OSL),
    ("--R", "split_r", ENV_RANDOM_RANGE_RATIO),
)
#: The workload the request carries: the splitter's hints, and the prompt length, which only the phase weights read.
WORKLOAD_HINTS = (*SPLIT_HINTS, ("--ISL", "split_isl", ENV_ISL))
_ENCODING = "utf-8"
#: The error class of a perf report that lacks a table the route reads.
_INCOMPLETE = "perf_report_incomplete"
_CAPTURE_KEYS = ("capture_folder", "graph_capture_path", "capture_folder_path")
#: How far above the trace input its ``profile`` directory may be (a trace file sits a level or two below it).
_PROFILE_DEPTH = 3
_OPS_SUMMARY = "ops_summary.csv"
_TAIL_CHARS = 1500
#: Decimal places of the times in the result.
_TIME_DIGITS = 3


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

    @property
    def split_dir(self) -> Path:
        return self.out_dir / "trace_split"


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
        done = run_with_session_kill(cmd, cwd=str(run.tracelens_root), watch=RoundWatch(timeout=run.timeout_sec))
    except subprocess.TimeoutExpired:
        raise _Refused("subprocess_timeout", f"{step} timed out after {run.timeout_sec}s") from None
    with run.log.open("a", encoding=_ENCODING) as handle:
        handle.write(f"$ {shlex.join(cmd)}\n{done.stdout}{done.stderr}\n[exit_code] {done.returncode}\n")
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "")[-_TAIL_CHARS:].strip()
        raise _Refused("tracelens_command_failed", f"{step} exited {done.returncode}: {tail}")


def _analysed_trace(trace_input: Path) -> Path:
    """The trace of the input that holds GPU kernels: the file itself, or the first trace of a directory that does.

    A CUDA-graph capture sidecar is never it: it holds the kernels of one graph capture (two fill kernels on a NEXTN head's
    profile), not serving steps, so the splitter finds no steady state in it. An input with nothing else is refused as
    ``trace_without_serving_steps``.
    """
    try:
        _, found = discover_trace_inputs(trace_input)
    except FileNotFoundError as exc:
        raise _Refused("trace_input_missing", str(exc)) from None
    root = trace_input if trace_input.is_dir() else trace_input.parent
    files = [path for path in found if not is_capture_fragment(path, root)]
    if found and not files:
        raise _Refused(
            "trace_without_serving_steps",
            f"every trace under {trace_input} is a CUDA-graph capture sidecar: the profiler recorded no serving step",
        )
    for candidate in files[:_PROBE_LIMIT]:
        if count_gpu_kernel_events(candidate):
            return candidate
    raise _Refused(
        "trace_without_gpu_kernels", f"none of the first {_PROBE_LIMIT} traces under {trace_input} has a GPU kernel"
    )


def unusable_trace(trace_input: str) -> dict[str, str] | None:
    """The failed result of an analysis whose ``trace_input`` cannot be analysed (it does not exist, or none of its first
    traces has a GPU kernel), or ``None`` when it can.

    The orchestrator's own analysis answers ``ok`` with no kernel for either, which reads as a trace with nothing in it.
    """
    try:
        _analysed_trace(Path(trace_input).expanduser().resolve())
    except _Refused as exc:
        return {"status": "failed", "error_class": exc.error_class, "error": str(exc)}
    return None


def split_hint(flag: str, raw: object) -> str | None:
    """``raw`` as the text the splitter's ``flag`` takes, or ``None`` when it is not a value that flag accepts."""
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    accepted = {
        "--CONC": value >= 1 and value.is_integer(),
        "--OSL": value > 0,
        "--ISL": value > 0,
        "--R": 0 <= value <= 1,
    }[flag]
    if not accepted:
        return None
    return str(int(value)) if value.is_integer() else repr(value)


def _split_hints(payload: Mapping[str, Any]) -> dict[str, str]:
    """The workload hints the request carries, by request key, each as the splitter takes it."""
    hints: dict[str, str] = {}
    for flag, key, _ in WORKLOAD_HINTS:
        raw = _text(payload, key)
        if not raw:
            continue
        hint = split_hint(flag, raw)
        if hint is None:
            raise _Refused("invalid_split_hint", f"{key}={raw!r} is not a value {flag} takes")
        hints[key] = hint
    return hints


def _split_command(trace: Path, split_dir: Path, hints: Mapping[str, str]) -> list[str]:
    cmd = [sys.executable, "-m", _SPLITTER, str(trace), "-o", str(split_dir), "--find-steady-state"]
    cmd += ["--num-steps", str(_SPLIT_NUM_STEPS), "--llm-inference"]
    for flag, key, _ in SPLIT_HINTS:
        if key in hints:
            cmd += [flag, hints[key]]
    return cmd


def _written_chunk(split_dir: Path, mode: str) -> Path | None:
    """The window the splitter wrote for ``mode``, ``None`` when it wrote none."""
    for suffix in _CHUNK_SUFFIXES:
        found = sorted(split_dir.rglob(f"{_CHUNK_PREFIX[mode]}_*.{suffix}"))
        if found:
            return found[0]
    return None


def _steady_state_chunk(split_dir: Path, mode: str) -> Path:
    """The window the splitter wrote for ``mode``."""
    chunk = _written_chunk(split_dir, mode)
    if chunk is None:
        written = sorted(path.name for path in split_dir.iterdir())[:8] if split_dir.is_dir() else []
        raise _Refused(
            "steady_state_chunk_missing",
            f"the splitter wrote no {mode} steady-state window under {split_dir}: {written}",
        )
    return chunk


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
    if not unified.is_file() or not (run.csv_dir / _OPS_SUMMARY).is_file():
        raise _Refused(
            _INCOMPLETE,
            f"the perf report wrote no unified_perf_summary.csv and ops_summary.csv in {run.csv_dir}",
        )
    return check_graph_replay_coverage(unified)


def _kernel_window(chunk: Path, trace: Path, capture: Path | None) -> tuple[KernelWindow, Relabelled]:
    """The kernel shares of the window's GPU events, the kernels of its graph replays named by the capture's launches; the
    whole rank trace is read only to price excluded events."""
    try:
        launched = read_launched_window(chunk, capture)
        return kernel_window(launched.events, lambda: read_gpu_events(trace)), launched
    except KernelTraceError as exc:
        raise _Refused("kernel_trace_unreadable", str(exc)) from None


def _phase_windows(split_dir: Path, trace: Path, capture: Path | None) -> list[PhaseWindow]:
    """The prefill window and the decode window the splitter wrote, with their kernels (the whole rank trace is read once,
    and only to price excluded events)."""
    steps, reference = split_steps(split_dir), rank_trace_reader(trace)
    try:
        return [
            read_phase_window(phase, _written_chunk(split_dir, mode), steps, capture, reference)
            for phase, mode in PHASE_MODES.items()
        ]
    except KernelTraceError as exc:
        raise _Refused("kernel_trace_unreadable", str(exc)) from None


def _write_degraded_report(run: _Run, coverage: GraphReplayCoverage) -> None:
    """TraceLens' degraded fallback ``analysis.md`` and the ``analysis.json`` rendered from it: what the agent route writes."""
    from TraceLens.Agent.Analysis.post_processing import render_analysis_json
    from TraceLens.Agent.Analysis.utils.deterministic_fallback import render_fallback_report

    report = run.out_dir / "analysis.md"
    unified = run.csv_dir / "unified_perf_summary.csv"
    report.write_text(render_fallback_report(unified, coverage.graph_replay_fraction), encoding=_ENCODING)
    render_analysis_json(report)


def _kernel_fields(kernels: KernelWindow, launched: Relabelled, table: Path) -> dict[str, Any]:
    """What a result says of its window's kernel shares: where the table is (empty when degraded), the window's numbers."""
    degraded = kernels.degraded_reason
    return {
        KERNEL_SHARES_KEY: "" if degraded else str(table),
        "window_gpu_busy_percent": round(kernels.busy_percent, 2),
        "window_kernel_ms": round(kernels.kernel_ms, _TIME_DIGITS),
        EXCLUDED_EVENTS_KEY: kernels.excluded_events,
        EXCLUDED_CLAIMED_KEY: round(kernels.excluded_claimed_ms, 2),
        "excluded_estimated_ms": round(kernels.excluded_estimated_ms, _TIME_DIGITS),
        "launcher_labelled_kernels": launched.launched,
        "graph_kernels_renamed": launched.renamed,
        "launcher_notes": list(launched.notes),
        OWN_DEGRADED_KEY: degraded,
    }


def _analysed_window(result: Mapping[str, Any]) -> str:
    """The steady-state window the orchestrator's analysis says it ran TraceLens on; empty when it names none."""
    artifacts = result.get("artifact_paths")
    fallback = artifacts.get("tracelens_steady_state_trace") if isinstance(artifacts, dict) else None
    return str(result.get("steady_state_trace") or fallback or "").strip()


def with_window_shares(result: Mapping[str, Any], trace_input: str, out_dir: Path) -> dict[str, Any]:
    """``result`` of an analysis the orchestrator ran (the agent route, a roofline) with the kernel shares of the window it
    analysed, which the op shares are then read from instead of the aten ops table of its perf report (the table whose
    durations the ROCm profiler's misreport breaks).

    The table is written to ``out_dir``. A result that names no window is returned as it is. One that names a window the
    kernel shares cannot be read from carries no shares and says why: the ops table is not read in its place.
    """
    named = _analysed_window(result)
    if not named or not trace_input:
        return dict(result)
    try:
        origin = Path(trace_input).expanduser()
        trace = _analysed_trace(origin)
        kernels, launched = _kernel_window(Path(named), trace, discover_capture_folder(origin, [trace]))
    except _Refused as exc:
        return {
            **result,
            KERNEL_SHARES_KEY: "",
            OWN_DEGRADED_KEY: f"the window {Path(named).name} cannot be read: {exc}",
        }
    table = out_dir / KERNEL_SHARES_FILE
    if not kernels.degraded_reason:
        write_kernel_table(kernels, table)
    return {**result, **_kernel_fields(kernels, launched, table)}


def _analyse(payload: Mapping[str, Any], session_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    trace_input = Path(str(payload["trace_input"])).expanduser().resolve()
    mode = _text(payload, "steady_state_mode") or _DEFAULT_MODE
    if mode not in _CHUNK_PREFIX:
        raise _Refused("invalid_steady_state_mode", f"steady_state_mode {mode!r} is not one of {sorted(_CHUNK_PREFIX)}")
    root = session_paths.tracelens_root()
    if not root.is_dir():
        raise _Refused("tracelens_root_missing", f"TraceLens root not found: {root}; set TRACELENS_ROOT to a checkout")
    budget = _text(payload, "budget_minutes")
    run = _Run(root, session_dir / "tracelens", int(float(budget or 60) * 60))
    run.out_dir.mkdir(parents=True, exist_ok=True)

    hints = _split_hints(payload)
    trace = _analysed_trace(trace_input)
    _sh(run, "split", _split_command(trace, run.split_dir, hints))
    chunk = _steady_state_chunk(run.split_dir, mode)
    capture_text = _text(payload, *_CAPTURE_KEYS)
    capture = (
        Path(capture_text).expanduser().resolve() if capture_text else discover_capture_folder(trace_input, [trace])
    )
    arch = resolve_arch_json_path(_text(payload, "target_platform") or DEFAULT_PLATFORM)
    _sh(run, "perf report", _perf_report_command(chunk, run, arch, capture))
    coverage = _graph_replay_coverage(run)
    if coverage.graph_under_recorded:
        _write_degraded_report(run, coverage)
    windows = _phase_windows(run.split_dir, trace, capture)
    mix = graded_mix(windows, Workload.from_hints(hints))
    table = run.out_dir / KERNEL_SHARES_FILE
    if not mix.degraded_reason:
        write_mix_table(mix, table)

    warnings = [] if arch is not None else [{"code": "gpu_arch_json_unresolved", "severity": "warning"}]
    return {
        "status": "ok",
        "tool": "tracelens_report",
        "route": REPORT_ROUTE,
        "trace_report_path": str(run.out_dir / "perf_report.xlsx"),
        "analysis_md_path": str(run.out_dir / "analysis.md") if coverage.graph_under_recorded else "",
        "ops_summary_path": str(run.csv_dir / _OPS_SUMMARY),
        "trace_input": str(trace_input),
        "analysed_trace": trace.name,
        "steady_state_mode": mode,
        "split_hints": hints,
        "steady_state_chunk": chunk.name,
        "capture_folder": str(capture or ""),
        "graph_replay_fraction": round(coverage.graph_replay_fraction, 4),
        "graph_under_recorded": coverage.graph_under_recorded,
        **mix_fields(windows, mix, table),
        "trace_health_warnings": warnings,
        **ruppell_fields(trace_input),
        "duration_sec": round(time.monotonic() - started, _TIME_DIGITS),
    }


def ruppell_fields(trace_input: Path) -> dict[str, Any]:
    """What ruppell kept of the profile whose traces are at ``trace_input`` (``<output>/profile`` or a file below it):
    its module table, whether its own check passed, and the share of decode GPU time it attributed to a module.

    Empty when ruppell did not trace the profile. ``ruppell_status`` is ``ok``, ``check_failed`` or ``unchecked`` (no
    readable check), and the decode share is ``None`` when the table has no decode step.
    """
    profile = next(
        (path for path in (trace_input, *trace_input.parents[:_PROFILE_DEPTH]) if path.name == PROFILE_DIR), None
    )
    kept = profile.parent / RUPPELL_DIR if profile is not None else None
    if kept is None or not (kept / MODULE_SHARES_FILE).is_file():
        return {}
    try:
        verdict = json.loads((kept / CHECK_FILE).read_text(encoding=_ENCODING)).get("ok")
    except (OSError, ValueError, AttributeError):
        verdict = None
    status = "unchecked" if verdict is None else "ok" if verdict is True else "check_failed"
    try:
        with (kept / MODULE_SHARES_FILE).open(encoding=_ENCODING, newline="") as handle:
            decode = [row for row in csv.DictReader(handle) if row.get("kind") == "decode"]
        total = sum(float(row.get("gpu_ms") or 0.0) for row in decode)
        unattributed = sum(float(row.get("gpu_ms") or 0.0) for row in decode if row.get("module") == UNATTRIBUTED)
    except (OSError, ValueError, csv.Error):
        return {MODULE_SHARES_KEY: "", "ruppell_status": "unreadable", "ruppell_check_ok": verdict is True}
    return {
        MODULE_SHARES_KEY: str(kept / MODULE_SHARES_FILE),
        "ruppell_status": status,
        "ruppell_check_ok": verdict is True,
        "ruppell_decode_attributed_share": round(1.0 - unattributed / total, 6) if total > 0 else None,
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


__all__ = [
    "DEFAULT_PLATFORM",
    "REPORT_ROUTE",
    "WORKLOAD_HINTS",
    "report_handler",
    "ruppell_fields",
    "split_hint",
    "unusable_trace",
    "with_window_shares",
]
