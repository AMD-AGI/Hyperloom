# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Token-trace coverage for the out-of-process geak / oob kernel backends.

GEAK and OOB run as kernel-agent subprocesses; their token usage only
survives as text in each attempt's ``*_stdout.log`` (surfaced on the
result as ``attempts[].optimized_path``). :func:`_trace_kernel_attempt_usage`
mines that log with the matching parser (``geak`` → litellm/OpenAI usage,
``oob`` → ``oob run --json`` usage) and folds the counts into the same
``llm_calls.jsonl`` ledger as the in-process backends.

Contract pinned here:

* a geak attempt whose stdout carries an OpenAI-shape ``usage`` lands a
  ``component=geak`` token row keyed to the kernel id;
* an oob attempt whose ``--json`` stdout carries ``usage`` lands a
  ``component=oob`` row;
* backends that emit no usage (and non-traced backends like claude) stay a
  silent no-op — no fabricated zero rows;
* a missing / unreadable log never raises.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator.kernel_request_handlers import (
    _trace_kernel_attempt_usage,
)
from inference_optimizer.session_paths import llm_calls_path


def _read_rows(session_dir: Path) -> list[dict]:
    path = llm_calls_path(session_dir)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _attempt(backend: str, log_path: Path, *, write: str | None = None) -> dict:
    if write is not None:
        log_path.write_text(write, encoding="utf-8")
    return {
        "backend": backend,
        "status": "completed",
        "optimized_path": str(log_path) if log_path.exists() else "",
    }


def test_geak_attempt_usage_lands_token_row(tmp_path: Path) -> None:
    """A geak attempt with litellm/OpenAI-shape usage yields a geak row."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "geak-1234_stdout.log"
    stdout = json.dumps({
        "result": "optimized kernel",
        "usage": {"prompt_tokens": 1200, "completion_tokens": 340},
    })
    result = {
        "kernel_id": "k007",
        "attempts": [_attempt("geak", log, write=stdout)],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    rows = _read_rows(session_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "geak"
    assert row["task_id"] == "k007"
    assert row["input_tokens"] == 1200
    assert row["output_tokens"] == 340
    # GEAK/litellm has no prompt-cache split.
    assert row["cache_read_input_tokens"] is None
    assert row["cache_creation_input_tokens"] is None


def test_oob_attempt_usage_lands_token_row(tmp_path: Path) -> None:
    """An oob ``--json`` attempt with a usage block yields an oob row."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "oob-abcd_stdout.log"
    # oob streams JSONL; the usage rides on a late result event.
    stdout = (
        json.dumps({"type": "init", "session_id": "s"}) + "\n"
        + json.dumps({
            "type": "result",
            "usage": {"input_tokens": 88, "output_tokens": 12},
        }) + "\n"
    )
    result = {
        "kernel_id": "k002",
        "attempts": [_attempt("oob", log, write=stdout)],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    rows = _read_rows(session_dir)
    assert len(rows) == 1
    assert rows[0]["component"] == "oob"
    assert rows[0]["task_id"] == "k002"
    assert rows[0]["input_tokens"] == 88
    assert rows[0]["output_tokens"] == 12


def test_no_usage_block_is_silent_noop(tmp_path: Path) -> None:
    """A geak/oob attempt with no recoverable usage writes no row (no zeros)."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    log = tmp_path / "geak-none_stdout.log"
    result = {
        "kernel_id": "k001",
        "attempts": [_attempt("geak", log, write="just some prose, no JSON usage")],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    assert _read_rows(session_dir) == []


def test_non_traced_backend_is_skipped(tmp_path: Path) -> None:
    """claude/codex/cursor account spend elsewhere; they are not mined here."""
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
        "attempts": [{"backend": "oob", "optimized_path": str(tmp_path / "gone.log")}],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    assert _read_rows(session_dir) == []


def test_mixed_attempts_only_traces_geak_oob_with_usage(tmp_path: Path) -> None:
    """A multi-backend ladder traces only the geak/oob attempts that report usage."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    geak_log = tmp_path / "geak_stdout.log"
    oob_log = tmp_path / "oob_stdout.log"
    claude_log = tmp_path / "claude_stdout.log"
    result = {
        "kernel_id": "k010",
        "attempts": [
            _attempt("claude", claude_log, write=json.dumps(
                {"usage": {"input_tokens": 1, "output_tokens": 1}})),
            _attempt("geak", geak_log, write=json.dumps(
                {"usage": {"prompt_tokens": 10, "completion_tokens": 20}})),
            _attempt("oob", oob_log, write=json.dumps(
                {"usage": {"input_tokens": 30, "output_tokens": 40}})),
        ],
    }
    _trace_kernel_attempt_usage(result, session_dir=session_dir)
    rows = _read_rows(session_dir)
    components = sorted(r["component"] for r in rows)
    assert components == ["geak", "oob"]


def test_result_without_attempts_is_noop(tmp_path: Path) -> None:
    """A failed/short-circuited result (no attempts list) writes nothing."""
    session_dir = tmp_path / "SESSION"
    session_dir.mkdir()
    _trace_kernel_attempt_usage(
        {"status": "failed", "error": "boom"}, session_dir=session_dir,
    )
    _trace_kernel_attempt_usage("not a dict", session_dir=session_dir)
    assert _read_rows(session_dir) == []
