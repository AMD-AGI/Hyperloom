# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``normalize`` on the two AgentKernelArena layouts: a driver.log session, and a campaign log with its artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kernelforge.run_export.document import normalize

#: One printed message; a tuple is a message the arena logged as several lines at one instant.
Message = str | tuple[str, ...]

RUN = "run_20261001_120000_gemm-forge"
TASK_CONFIG = """description: Optimize the runtime entrypoint aiter.ops.gemm_a8w8. Keep numerics.
kernel_identity: {logical_operator: gemm_a8w8_n1024_k4096, source_owner: aiter}
candidate: {language: flydsl}
baseline: {diagnostic_reason: "deepseek-v4-flash family policy: keep the baseline kernel"}
"""
SESSION_TXT = """kernelforge : main (1c14e79e1 feat/forge)
arena       : main (abcdef1)
model       : claude-opus (effort=high, budget=2h)
KB          : mode=read
host        : mi355-node-7
"""
SETUP: list[Message] = [
    f"NEW RUN: {RUN}",
    "[FORGE] [forge-rewrite] op=gemm_a8w8 shapes=4",
    "[FORGE] [forge-rewrite] source baseline: 1.5 ms (full suite, cases=[m1, m2])",
    "[FORGE] [forge-rewrite] PORT OK (attempt 1, SNR=41.2)",
    "[FORGE] [forge-rewrite] interim (port only): flydsl=1.2 ms vs source=1.5 ms -> speedup=1.25x",
    "[FORGE] Starting autonomous iteration loop",
]
HEADER = "[FORGE] --- Iteration 1 (best mean case speedup: 1.250000x, remaining: 50 min) ---"
QUERY = "[FORGE]   [agent] Querying agent for kernel modification..."
KEEP_LINE = "[FORGE]   [KEEP] mean case speedup=1.305x — NEW BEST (+4.4%); raw mean=0.9000ms"
KEPT: list[Message] = [
    "[FORGE] [search-policy] mode=EXPLOIT reason=fresh",
    HEADER,
    QUERY,
    "[FORGE]   [session-end] backend=claude reason=agent_stopped edits=3 turns=20 pass=True",
    (
        "[FORGE]   [agent] Rationale: Tiled the K loop by 128 so each wave reuses its A tile.",
        "[FORGE] It halves LDS traffic. Then it unrolled twice. A fourth sentence.",
    ),
    "[FORGE]   [validate] Stage 1 correctness: PASS SNR=45.0dB",
    "[FORGE]   [bench] pristine-relative scores=[1.3, 1.31]; sigma=0.01; mean score=1.305x; required=1.27x; "
    + "raw mean=0.9 ms",
    "[FORGE]   [agent] Committed verified best: 0123abcd",
    KEEP_LINE,
    "[FORGE] [forge-rewrite] KEEP KB publish (0123abcd99): kb-record-77 published",
]
MERGED: list[Message] = [
    "[FORGE] --- Iteration 2 (best mean case speedup: 1.305000x, remaining: 40 min) ---",
    "[FORGE]   [merge] declined: iterations 1+3 share case m1",
    "[FORGE]   [merge] stacked candidates 1+3: Tile K by 128 | Vectorize the epilogue",
    "[FORGE]   [REVERT] mean case speedup=1.2x not better than best=1.305x (required 1.32x)",
]
UNFINISHED: list[Message] = [
    "[FORGE] --- Iteration 3 (best mean case speedup: 1.305000x, remaining: 30 min) ---",
    QUERY,
]


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _stamp(seconds: int) -> str:
    return f"2026-10-01 12:{seconds // 60:02d}:{seconds % 60:02d}"


def _log(messages: list[Message], *, lost: str = "") -> str:
    """Arena log lines, one message every three seconds; ``lost`` is a line this copy of the log does not hold."""

    lines = []
    for index, message in enumerate(messages):
        for line in message if isinstance(message, tuple) else (message,):
            if line != lost:
                lines.append(f"{_stamp(3 * index)},000 - INFO - {line}\n")
    return "".join(lines)


def _at(messages: list[Message], line: str) -> str:
    """When ``_log`` logs ``line``."""

    index = next(
        index
        for index, message in enumerate(messages)
        if line in (message if isinstance(message, tuple) else (message,))
    )
    return _stamp(3 * index).replace(" ", "T") + ".000"


def _session(root: Path, messages: list[Message], name: str = "sess-gemm") -> Path:
    """A forge-rewrite session directory; its arena log lost the KEEP line, which driver.log still holds."""

    session = root / name
    run_dir = session / "arena_ws_0" / "run_20261001_120000_gemm"
    task = run_dir / "Aiter-task_gemm"
    _write(task / "config.yaml", TASK_CONFIG)
    _write(
        task / "scripts" / "workload.json", json.dumps({"definition": {"inputs": {"a": {"dtype": "float8_e4m3fn"}}}})
    )
    _write(
        run_dir / ".task-sessions" / task.name / "runtime_identity.json",
        json.dumps({"flydsl": "0.3.2", "hip": "7.1.0"}),
    )
    _write(session / "config.yaml", "target_gpu_model: MI355X\n")
    _write(session / "session.txt", SESSION_TXT)
    _write(session / "arena_logs" / "gemm.log", _log(messages, lost=KEEP_LINE))
    _write(session / "driver.log", _log(messages))
    return session


def _units(root: Path) -> dict[str, dict[str, Any]]:
    return {unit["unit_id"]: unit for unit in normalize([root])["runs"]}


def test_a_session_yields_its_port_and_each_iteration(tmp_path: Path) -> None:
    messages = [*SETUP, *KEPT, *MERGED, *UNFINISHED]
    session = _session(tmp_path, messages)
    units = _units(tmp_path)

    assert list(units) == [f"sess-gemm/{name}" for name in ("port", "iteration-1", "iteration-2", "iteration-3")]
    port, kept, merged, unfinished = units.values()
    assert port["identity"] == {
        "kernel": "gemm_a8w8_n1024_k4096",
        "framework": "aiter",
        "backend": "flydsl",
        "model": "deepseek-v4-flash",
        "operator": "gemm_a8w8",
        "gpu": "mi355x",
        "backend_version": "0.3.2",
        "dtype": "float8_e4m3fn",
    }
    assert port["provenance"] == {
        "pipeline": "forge-rewrite-by-flydsl",
        "hip": "7.1.0",
        "kernelforge_commit": "1c14e79e1",
        "kernelforge_branch": "feat/forge",
        "arena_commit": "abcdef1",
        "llm_model": "claude-opus",
        "llm_effort": "high",
        "kb_mode": "read",
        "host": "mi355-node-7",
        "candidate_origin": "port",
    }
    task = session / "arena_ws_0" / "run_20261001_120000_gemm"
    assert port["log_files"] == [
        str(path.resolve())
        for path in (
            session / "session.txt",
            session / "config.yaml",
            session / "driver.log",
            session / "arena_logs" / "gemm.log",
            task / "Aiter-task_gemm" / "config.yaml",
            task / ".task-sessions" / "Aiter-task_gemm" / "runtime_identity.json",
            task / "Aiter-task_gemm" / "scripts" / "workload.json",
        )
    ]
    assert {key: port[key] for key in ("run_id", "seq", "decision", "mean_case_speedup", "source_ms", "approach")} == {
        "run_id": RUN,
        "seq": 0,
        "decision": "PORT OK",
        "mean_case_speedup": 1.25,
        "source_ms": 1.5,
        "approach": "op=gemm_a8w8 shapes=4",
    }
    assert (port["started_at"], port["completed_at"]) == ("2026-10-01T12:00:03.000", "2026-10-01T12:00:12.000")

    assert kept["started_at"] == _at(messages, KEPT[0])
    assert kept["completed_at"] == _at(messages, KEEP_LINE)
    assert kept["reasoning"] == (
        "Tiled the K loop by 128 so each wave reuses its A tile. It halves LDS traffic. Then it unrolled twice."
    )
    assert (kept["decision"], kept["mean_case_speedup"], kept["best_mean_case_speedup"]) == ("KEEP", 1.305, 1.25)
    assert kept["provenance"]["commit"] == "0123abcd"
    assert kept["provenance"]["kb_record"] == "kb-record-77"
    assert kept["provenance"]["candidate_origin"] == "session"
    assert kept["log_files"] == port["log_files"]

    assert merged["provenance"]["candidate_origin"] == "merge"
    assert merged["approach"] == "stacked candidates 1+3: Tile K by 128 | Vectorize the epilogue"
    assert merged["alternatives"] == ["merge declined: iterations 1+3 share case m1"]
    assert merged["decision"] == "REVERT_PERF"

    assert "completed_at" not in unfinished and "decision" not in unfinished
    assert (unfinished["stop"], unfinished["stop_line"]) == ("log-ended", QUERY.removeprefix("[FORGE]").strip())


VALIDATION_FAILED = "[FORGE]   [REVERT] Validation failed (stage 1)"


@pytest.mark.parametrize(
    ("lines", "label"),
    [
        ([KEEP_LINE], "KEEP"),
        (["[FORGE]   [REVERT] mean case speedup=1.1x not better than best=1.25x"], "REVERT_PERF"),
        (["[FORGE]   [REVERT] mean case speedup=None not better than best=1.25x"], "REVERT_PERF"),
        (["[FORGE]   [REVERT] Protected integrity violation; canonical validation skipped"], "REVERT_INTEGRITY"),
        (["[FORGE]   [REVERT] Workspace still contended; canonical measurement skipped"], "REVERT_CONTENDED"),
        (["[FORGE]   [REVERT] Device still contended; nothing planned, run or measured"], "REVERT_CONTENDED"),
        (["[FORGE]   [REVERT] Iteration crashed (exit 139)"], "CRASH"),
        (["[FORGE]   [REVERT] Commit failed after validation (12s)"], "CRASH"),
        ([VALIDATION_FAILED], "BUILD_FAILED"),
        (["[FORGE]   [validate] Stage 1 correctness: FAIL SNR=3.0dB", VALIDATION_FAILED], "REVERT_VALIDATION"),
        (["[FORGE]   [validate] Stage 2 benchmark: TIMEOUT", VALIDATION_FAILED], "REVERT_VALIDATION_TIMEOUT"),
        (["[FORGE]   [validate] Stage 1 correctness: ERROR", VALIDATION_FAILED], "REVERT_VALIDATION_ERROR"),
        ([QUERY, "[FORGE]   [agent] No tracked source changes; skipping candidate"], "NO_CHANGES"),
        ([QUERY, "[FORGE]   [agent] ERROR: sdk died", "[FORGE]   [agent] No tracked source changes"], "AGENT_ERROR"),
        (
            [
                QUERY,
                "[FORGE]   [session-end] backend=claude reason=api_error",
                "[FORGE]   [agent] No tracked source changes",
            ],
            "API_ERROR",
        ),
        ([QUERY, "[FORGE]   [orchestration] failed (planner timed out)"], "ORCHESTRATION_ERROR"),
        (["[FORGE]   [orchestration] failed (planner timed out)"], None),
        (["[FORGE]   [CRASH] worker died"], None),
    ],
)
def test_each_decision_line_shape_reads_back_as_its_label(tmp_path: Path, lines: list[str], label: str | None) -> None:
    _session(tmp_path, [*SETUP, HEADER, *lines])
    iteration = _units(tmp_path)["sess-gemm/iteration-1"]
    assert iteration.get("decision") == label


@pytest.mark.parametrize(
    ("ending", "stop", "stop_line"),
    [
        (
            ["[FORGE] ROUND REFUSED FOR BUDGET at iteration 1: 4 min left"],
            "budget-refused",
            "ROUND REFUSED FOR BUDGET at iteration 1: 4 min left",
        ),
        (
            [
                "[FORGE] Traceback (most recent call last):",
                "[FORGE]   File runner.py",
                "[FORGE] RuntimeError: device lost",
            ],
            "crashed",
            "RuntimeError: device lost",
        ),
        (
            ["[FORGE] [forge-rewrite] optimize cutoff reached (budget)"],
            "cut-off",
            "[forge-rewrite] optimize cutoff reached (budget)",
        ),
        (["[FORGE] FORCE STOP: operator request"], "cut-off", "FORCE STOP: operator request"),
        (
            ["[FORGE] BUDGET EXHAUSTED after 3 iterations", "Starting centralized kernel evaluation"],
            "no-decision",
            HEADER.removeprefix("[FORGE] "),
        ),
    ],
)
def test_an_iteration_without_a_decision_records_how_the_loop_stopped(
    tmp_path: Path, ending: list[str], stop: str, stop_line: str
) -> None:
    _session(tmp_path, [*SETUP, HEADER, QUERY, *ending])
    iteration = _units(tmp_path)["sess-gemm/iteration-1"]
    assert "completed_at" not in iteration
    assert (iteration["stop"], iteration["stop_line"]) == (stop, stop_line)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param({"a": {"shape": [4]}}, id="first-input-names-no-dtype"),
        pytest.param([{"dtype": "float8_e4m3fn"}], id="inputs-not-a-mapping"),
    ],
)
def test_a_task_that_names_its_entry_and_dtype_elsewhere_is_read_from_those_files(tmp_path: Path, inputs: Any) -> None:
    session = _session(tmp_path, [*SETUP, *KEPT])
    task_dir = session / "arena_ws_0" / "run_20261001_120000_gemm" / "Aiter-task_gemm"
    task_session = task_dir.parent / ".task-sessions" / task_dir.name
    _write(task_dir / "config.yaml", TASK_CONFIG.replace("runtime entrypoint aiter.ops.gemm_a8w8.", "the kernel."))
    _write(task_dir / "BUNDLE_README.md", "The baseline entry is `aiter.ops.gemm_a8w8_bpreshuffle`.\n")
    _write(task_dir / "scripts" / "workload.json", json.dumps({"definition": {"inputs": inputs}}))
    validation = {"result": {"cases": [{"dtype": "bfloat16"}, {"dtype": "float16"}]}}
    _write(task_session / "action-0001-task-validate-task.json", json.dumps(validation))

    port = _units(tmp_path)["sess-gemm/port"]

    assert (port["identity"]["operator"], port["identity"]["dtype"]) == ("gemm_a8w8_bpreshuffle", "bfloat16")
    assert str((task_dir / "BUNDLE_README.md").resolve()) in port["log_files"]
    assert port["log_files"][-1] == str((task_session / "action-0001-task-validate-task.json").resolve())


def test_a_session_without_one_arena_log_is_one_unreadable_unit(tmp_path: Path) -> None:
    _write(tmp_path / "broken" / "driver.log", _log(SETUP))
    assert normalize([tmp_path])["runs"] == [
        {
            "unit_id": "broken/session",
            "kind": "session",
            "read_error": ("expected one arena log, one arena run directory, and one task directory; found 0, 0, 0"),
            "log_files": [str((tmp_path / "broken" / "driver.log").resolve())],
        }
    ]
