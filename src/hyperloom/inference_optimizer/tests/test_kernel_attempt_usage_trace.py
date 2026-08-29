# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Token-trace coverage for out-of-process forge kernel backend attempts.

Forge runs as a kernel-agent subprocess; its token usage only survives as text
in each attempt's ``*_stdout.log`` (surfaced on the result as
``attempts[].optimized_path``). :func:`_trace_kernel_attempt_usage` mines that
log and folds the counts into the same ``llm_calls.jsonl`` ledger as the
in-process backends.

Contract pinned here:

* a forge attempt whose stdout carries ``FORGE_LLM_USAGE`` lands a
  ``component=forge`` token row keyed to the kernel id;
* backends that emit no usage (and non-traced backends like claude) stay a
  silent no-op — no fabricated zero rows;
* a missing / unreadable log never raises.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.kernel.request_handlers import (
    _trace_kernel_attempt_usage,
)
from hyperloom.inference_optimizer.session.session_paths import llm_calls_path


def _read_rows(session_dir: Path) -> list[dict]:
    path = llm_calls_path(session_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _attempt(backend: str, log_path: Path, *, write: str | None = None) -> dict:
    if write is not None:
        log_path.write_text(write, encoding="utf-8")
    return {
        "backend": backend,
        "status": "completed",
        "optimized_path": str(log_path) if log_path.exists() else "",
    }


def test_forge_attempt_usage_carries_reasoning_tokens(tmp_path: Path) -> None:
    """A reasoning model's hidden output reaches this ledger too, not just the
    in-process backends' rows."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "forge-reasoning_stdout.log"
    stdout = "FORGE_LLM_USAGE " + json.dumps(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_output_tokens": 4096,
        }
    )
    result = {
        "kernel_id": "k001",
        "attempts": [_attempt("forge", log, write=stdout)],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)

    rows = _read_rows(session_dir)
    assert len(rows) == 1
    assert rows[0]["reasoning_output_tokens"] == 4096
    assert rows[0]["output_tokens"] == 20


def test_forge_attempt_usage_lands_token_row(tmp_path: Path) -> None:
    """A forge attempt whose stdout carries the FORGE_LLM_USAGE marker yields a
    forge row (the Kernel-Forge loop aggregates its claude-agent-sdk spend)."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "forge-xy_stdout.log"
    stdout = (
        "forge done: baseline=92.3 best=85.1 improved=True kernel_backend=ck gpu=gfx942\n"
        + "FORGE_LLM_USAGE "
        + json.dumps(
            {
                "input_tokens": 5000,
                "output_tokens": 1200,
                "cache_creation_input_tokens": 400,
                "cache_read_input_tokens": 3000,
                "total_cost_usd": 18.5,
                "calls": 7,
            }
        )
        + "\n"
        + "Autonomous loop complete\n"
    )
    result = {
        "kernel_id": "k042",
        "attempts": [_attempt("forge", log, write=stdout)],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    rows = _read_rows(session_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "forge"
    assert row["task_id"] == "k042"
    assert row["input_tokens"] == 5000
    assert row["output_tokens"] == 1200
    assert row["cache_creation_input_tokens"] == 400
    assert row["cache_read_input_tokens"] == 3000


def test_forge_attempt_without_marker_is_noop(tmp_path: Path) -> None:
    """An older Forge build (no FORGE_LLM_USAGE marker) writes no row."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "forge-old_stdout.log"
    result = {
        "kernel_id": "k043",
        "attempts": [_attempt("forge", log, write="forge done: baseline=1 best=1\n")],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    assert _read_rows(session_dir) == []


def test_forge_steps_written_to_audit(tmp_path: Path) -> None:
    """A forge attempt's FORGE_STEPS marker lands per-iteration + summary rows
    in reports/trace/forge_steps.jsonl, keyed by kernel id."""
    from hyperloom.orchestrator.kernel.request_handlers import (
        _trace_kernel_attempt_steps,
    )
    from hyperloom.inference_optimizer.session.session_paths import forge_steps_path

    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "forge-steps_stdout.log"
    payload = {
        "steps": [
            {
                "iteration": 1,
                "decision": "KEEP",
                "wall_ms": 88.1,
                "rationale": "fuse epilogue",
                "validation_passed": True,
            },
            {"iteration": 2, "decision": "REVERT", "wall_ms": 90.0, "validation_passed": False},
        ],
        "summary": {"iterations": 2, "kept": 1, "speedup": 1.05, "termination_reason": "plateaued"},
    }
    stdout = "FORGE_STEPS " + json.dumps(payload) + "\n"
    result = {"kernel_id": "k099", "attempts": [_attempt("forge", log, write=stdout)]}
    _trace_kernel_attempt_steps(result, session_dir=session_dir)

    rows = [
        json.loads(line)
        for line in forge_steps_path(session_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    iters = [r for r in rows if r["kind"] == "iteration"]
    summary = [r for r in rows if r["kind"] == "summary"]
    assert len(iters) == 2 and len(summary) == 1
    assert all(r["kernel_id"] == "k099" for r in rows)
    assert iters[0]["decision"] == "KEEP"
    assert summary[0]["termination_reason"] == "plateaued"


def test_forge_steps_noop_without_marker(tmp_path: Path) -> None:
    from hyperloom.orchestrator.kernel.request_handlers import (
        _trace_kernel_attempt_steps,
    )
    from hyperloom.inference_optimizer.session.session_paths import forge_steps_path

    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "forge-nomark_stdout.log"
    result = {"kernel_id": "k1", "attempts": [_attempt("forge", log, write="forge done\n")]}
    _trace_kernel_attempt_steps(result, session_dir=session_dir)
    assert not forge_steps_path(session_dir).exists()


def test_no_usage_block_is_silent_noop(tmp_path: Path) -> None:
    """A traced attempt with no recoverable usage writes no row (no zeros)."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "forge-none_stdout.log"
    result = {
        "kernel_id": "k001",
        "attempts": [_attempt("forge", log, write="just some prose, no usage marker")],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    assert _read_rows(session_dir) == []


def test_non_traced_backend_is_skipped(tmp_path: Path) -> None:
    """Backends outside the traced set are not mined for token usage here."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "claude-1_stdout.log"
    stdout = json.dumps({"usage": {"input_tokens": 5, "output_tokens": 5}})
    result = {
        "kernel_id": "k003",
        "attempts": [_attempt("claude", log, write=stdout)],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    assert _read_rows(session_dir) == []


def test_missing_log_path_never_raises(tmp_path: Path) -> None:
    """An attempt whose stdout log is absent degrades to no row, no error."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    result = {
        "kernel_id": "k004",
        "attempts": [{"backend": "forge", "optimized_path": str(tmp_path / "gone.log")}],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    assert _read_rows(session_dir) == []


def test_mixed_attempts_only_traces_forge_with_usage(tmp_path: Path) -> None:
    """A multi-backend ladder traces only forge attempts that report usage."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    forge_log = tmp_path / "forge_stdout.log"
    claude_log = tmp_path / "claude_stdout.log"
    forge_stdout = "FORGE_LLM_USAGE " + json.dumps({"input_tokens": 30, "output_tokens": 40})
    result = {
        "kernel_id": "k010",
        "attempts": [
            _attempt("claude", claude_log, write=json.dumps({"usage": {"input_tokens": 1, "output_tokens": 1}})),
            _attempt("forge", forge_log, write=forge_stdout),
        ],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    rows = _read_rows(session_dir)
    components = sorted(r["component"] for r in rows)
    assert components == ["forge"]


def test_result_without_attempts_is_noop(tmp_path: Path) -> None:
    """A failed/short-circuited result (no attempts list) writes nothing."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    _trace_kernel_attempt_usage(
        {"status": "failed", "error": "boom"},
        session_dir=session_dir,
    )
    _trace_kernel_attempt_usage("not a dict", session_dir=session_dir)
    assert _read_rows(session_dir) == []
