# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builders of a baseline-prelude child's session for the scheduling tests: a framework git checkout, the
``state.json`` of a completed child on it or of one whose baseline failed, a roofline profile, and an AgentX baseline
run."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
)

from hyperloom_multibranch.scheduling.prelude_reader import MB_LATEST_STATE_SCHEMA_VERSION

REMOTE_URL = "https://example.invalid/sglang.git"
COMPLETED_STOP_REASON = "time_exhausted"
#: How the main flow ends a run whose baseline kept failing, and the error class of the baseline attempt it failed on.
BASELINE_FAILED_STOP_REASON = "baseline_failed"
BASELINE_ERROR_CLASS = "subprocess_nonzero"
BASELINE_TPUT = 1234.5
ROOFLINE_PEAK = 4000.0
ROOFLINE_RUN_ID = "5f3e583beda4475e841f4bb7bacc9e4b"
TRACE_FILE = "1785743528.38-TP-0.trace.json.gz"
POINT_CONFIG = "benchmark:\n  envs:\n    CONC: 16\n    OSL: 1024\n"
#: The toolbox's copy of a real child's AgentX baseline run (GLM-5.3, TP4, CONC 10), its export cut to seven requests.
AGENTX_RUN_DATA = Path(__file__).parents[2] / "toolbox" / "tests" / "data" / "agentx_child_baseline"
AGENTX_RUN_ID = "fece9a47d3e148c4bc390eb760c5f4f4"
AGENTX_BENCHMARK = "benchmark_sglang_20261026_075016"
AGENTX_RUN_SECONDS = 5287.67
#: The values that run's report holds, as the child's ``baseline_perf`` records them.
AGENTX_RUN_PERF = {
    GRADED_INTVTY_P50: 132.1028,
    GRADED_INTVTY: 81.72424,
    GRADED_OUTPUT: 442.30405,
    GRADED_DURATION: 3601.52025,
    GRADED_ERROR_RATE: 0.06925207756232687,
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True, check=True).stdout.strip()


def framework_checkout(parent: Path, remote: str = REMOTE_URL) -> Path:
    """A git checkout with one commit and, unless ``remote`` is empty, an ``origin`` remote."""

    root = parent / "framework"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "f.py").write_text("a\n")
    _git(root, "add", "f.py")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "init")
    if remote:
        _git(root, "remote", "add", "origin", remote)
    return root


def head_of(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")


def state_document(framework: Path) -> dict[str, object]:
    """A completed child session's ``state.json``, run on the framework checkout ``framework``."""

    return {
        "schema_version": MB_LATEST_STATE_SCHEMA_VERSION,
        "stop_reason": COMPLETED_STOP_REASON,
        "baseline_tput": BASELINE_TPUT,
        "framework_version": "0.4.1",
        "gpu_type": "mi355x",
        "framework_repo_path": str(framework),
        "roofline_snapshots": [
            {"theoretical_peak_tok_per_sec": ROOFLINE_PEAK},
            {"theoretical_peak_tok_per_sec": None},
        ],
    }


def write_child_session(session_dir: Path, state: dict[str, object]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "state.json").write_text(json.dumps(state))
    return session_dir


def child_with_patch(tmp_path: Path) -> Path:
    """A completed child session on a fresh framework checkout."""

    return write_child_session(tmp_path / "child", state_document(framework_checkout(tmp_path)))


def failed_child(tmp_path: Path) -> Path:
    """The session of a child whose baseline failed until the main flow stopped it, on a fresh framework checkout."""

    state = {
        **state_document(framework_checkout(tmp_path)),
        "stop_reason": BASELINE_FAILED_STOP_REASON,
        "baseline_tput": 0.0,
        "roofline_snapshots": [],
        "baseline_attempts": [
            {"task_id": "b1", "status": "failed", "decision": "", "error_class": "oom"},
            {"task_id": "b2", "status": "failed", "decision": "", "error_class": BASELINE_ERROR_CLASS},
        ],
    }
    return write_child_session(tmp_path / "child", state)


def with_roofline_profile(session: Path, run_id: str = ROOFLINE_RUN_ID) -> Path:
    """Add to the child ``session`` a roofline run ``run_id`` with a trace and its point config; return the trace
    directory."""

    run = session / "runs" / "roofline" / run_id
    trace_dir = run / "benchmark_sglang_20260803_074937" / "torch_trace"
    (trace_dir / "capture_traces").mkdir(parents=True)
    (trace_dir / TRACE_FILE).write_bytes(b"trace")
    (trace_dir / "capture_traces" / "bs_8_rank0.json.gz").write_bytes(b"capture")
    (run / "baseline_config.with_envs.yaml").write_text(POINT_CONFIG)
    return trace_dir


def with_agentx_baseline_run(
    session: Path,
    perf: Mapping[str, float] | None = None,
    *,
    runtime_seconds: float = AGENTX_RUN_SECONDS,
    **anchor: object,
) -> Path:
    """Add to the child ``session`` the AgentX baseline run its anchor was promoted from; return the run directory.

    ``perf`` rewrites the run's report to hold those values instead of :data:`AGENTX_RUN_PERF`, and the child records
    them as its ``baseline_perf``; ``anchor`` sets fields of the promoted attempt, its ``extras`` included.
    """

    run = session / "runs" / "baseline" / AGENTX_RUN_ID
    shutil.copytree(AGENTX_RUN_DATA, run)
    values = dict(AGENTX_RUN_PERF if perf is None else perf)
    if perf is not None:
        _rewrite_report(run / AGENTX_BENCHMARK / "benchmark_report.json", values)
    state = json.loads((session / "state.json").read_text())
    attempt: dict[str, object] = {
        "task_id": AGENTX_RUN_ID,
        "decision": "promoted",
        "status": "succeeded",
        "workspace": str(run / AGENTX_BENCHMARK),
        "extras": {"baseline_tput": state["baseline_tput"]},
    }
    attempt.update(anchor)
    state["baseline_attempts"] = [attempt]
    state["baseline_runtime_sec"] = runtime_seconds
    state["baseline_measure_round_dropped"] = False
    state["baseline_perf"] = values
    write_child_session(session, state)
    return run


def _rewrite_report(report: Path, values: Mapping[str, float]) -> None:
    document = json.loads(report.read_text())
    metrics = document["agentx_metrics"]
    metrics["latency_seconds"]["e2e_norm_intvty"].update(p50=values[GRADED_INTVTY_P50], p90=values[GRADED_INTVTY])
    metrics["throughput"].update(
        output_tokens_per_second=values[GRADED_OUTPUT], duration_seconds=values[GRADED_DURATION]
    )
    metrics["requests"]["error_rate"] = values[GRADED_ERROR_RATE] / 100
    report.write_text(json.dumps(document))
