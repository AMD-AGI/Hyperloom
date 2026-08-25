# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace import llm_trace


def test_llm_call_record_from_metadata_coerces_and_appends(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_trace, "_now_iso", lambda **_: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(llm_trace, "get_emitter", lambda _session_dir: None, raising=False)

    record = llm_trace.LLMCallRecord.from_metadata(
        session_id="s1",
        component="forge",
        metadata={
            "model": "m",
            "input_tokens": "12",
            "output_tokens": 3.0,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": "4",
            "latency_ms": "99",
        },
        task_id="k1",
    )
    record.reviewed_msg_ids = [" a ", "", 7]

    llm_trace.append_llm_call(session_dir=tmp_path, record=record)
    row = json.loads((tmp_path / "reports" / "trace" / "llm_calls.jsonl").read_text())

    assert row["component"] == "forge"
    assert row["input_tokens"] == 12
    assert row["output_tokens"] == 3
    assert row["cache_read_input_tokens"] == 4
    assert row["latency_ms"] == 99
    assert row["reviewed_msg_ids"] == ["a", "7"]


def test_from_metadata_carries_call_id_and_reasoning_tokens(tmp_path: Path, monkeypatch):
    """The backend's call_id + reasoning spend reach the unified ledger."""
    monkeypatch.setattr(llm_trace, "_now_iso", lambda **_: "2026-01-01T00:00:00Z")

    record = llm_trace.LLMCallRecord.from_metadata(
        session_id="s1",
        component="specialist",
        metadata={
            "model": "gpt-5-codex",
            "call_id": "abc123",
            "input_tokens": 10,
            "output_tokens": 5,
            "reasoning_output_tokens": "2048",
        },
    )
    llm_trace.append_llm_call(session_dir=tmp_path, record=record)
    row = json.loads((tmp_path / "reports" / "trace" / "llm_calls.jsonl").read_text())

    assert row["call_id"] == "abc123"
    # Reasoning output is reported alongside the visible reply, not folded into it.
    assert row["reasoning_output_tokens"] == 2048
    assert row["output_tokens"] == 5


def test_new_call_id_is_unique():
    assert llm_trace.new_call_id() != llm_trace.new_call_id()


def test_llm_call_record_rejects_unknown_component(tmp_path: Path):
    with pytest.raises(llm_trace.LLMTraceRowError):
        llm_trace.append_llm_call(
            session_dir=tmp_path,
            record=llm_trace.LLMCallRecord(session_id="s1", component="removed_backend"),
        )


def test_llm_trace_handles_unusable_reviewed_ids_and_io_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_trace, "_now_iso", lambda **_: "2026-01-01T00:00:00Z")
    record = llm_trace.LLMCallRecord(
        session_id="s2",
        component="forge",
        reviewed_msg_ids=object(),
    )
    row = record.to_row()
    assert row["reviewed_msg_ids"] is None

    def _raise_mkdir(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", _raise_mkdir)
    llm_trace.append_llm_call(session_dir=tmp_path, record=record)


def test_coerce_optional_str_list_variants():
    # Bare string is wrapped into a single-element list.
    assert llm_trace._coerce_optional_str_list("abc") == ["abc"]
    # None -> None; non-iterable -> None; all-empty -> None.
    assert llm_trace._coerce_optional_str_list(None) is None
    assert llm_trace._coerce_optional_str_list(123) is None
    assert llm_trace._coerce_optional_str_list(["", "  "]) is None
    assert llm_trace._coerce_optional_str_list(["x", "", " y "]) == ["x", "y"]


def test_for_failure_records_error_without_token_counters(tmp_path: Path, monkeypatch):
    """A failed call lands in the ledger as an ``error`` row, tokens unmeasured."""
    monkeypatch.setattr(llm_trace, "_now_iso", lambda **_: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(llm_trace, "get_emitter", lambda _session_dir: None, raising=False)

    record = llm_trace.LLMCallRecord.for_failure(
        session_id="s1",
        component="orchestration",
        role="orchestration",
        error=RuntimeError("litellm.BadRequestError: AnthropicException"),
        model="claude-opus-5",
        tick=4,
        phase="EXPLORE",
        latency_ms=1200,
    )
    llm_trace.append_llm_call(session_dir=tmp_path, record=record)
    row = json.loads((tmp_path / "reports" / "trace" / "llm_calls.jsonl").read_text())

    assert row["status"] == llm_trace.LLM_STATUS_ERROR
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "litellm.BadRequestError: AnthropicException"
    # Join keys survive so the failure lands on the same phase/agent as its peers.
    assert (row["component"], row["tick"], row["phase"]) == ("orchestration", 4, "EXPLORE")
    assert row["latency_ms"] == 1200
    # Nothing was measured, so no counter may claim zero.
    assert all(
        row[k] is None
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


def test_for_failure_accepts_plain_message_and_truncates(tmp_path: Path):
    """A gateway body can be huge; the ledger keeps a bounded slice of it."""
    record = llm_trace.LLMCallRecord.for_failure(
        session_id="s1",
        component="forge",
        error="x" * 5000,
    )
    row = record.to_row()
    assert row["error_type"] is None
    assert len(row["error_message"]) == llm_trace._ERROR_MESSAGE_MAX


def test_success_rows_default_to_ok_status(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_trace, "_now_iso", lambda **_: "2026-01-01T00:00:00Z")
    row = llm_trace.LLMCallRecord(session_id="s1", component="forge").to_row()
    assert row["status"] == llm_trace.LLM_STATUS_OK
    assert row["error_type"] is None and row["error_message"] is None


def test_append_llm_call_rejects_unknown_status(tmp_path: Path):
    record = llm_trace.LLMCallRecord(session_id="s1", component="forge")
    record.status = "partially_ok"
    with pytest.raises(llm_trace.LLMTraceRowError):
        llm_trace.append_llm_call(session_dir=tmp_path, record=record)


def test_llm_call_failed_is_a_backend_error(tmp_path: Path):
    """The marker must stay catchable as ``BackendError``.

    Retry and error-streak accounting are written against ``BackendError``; if
    the marker were a sibling type instead of a subclass, marking a call site
    would silently change failure handling as well as tracing.
    """
    from hyperloom.orchestrator.roles.base import BackendError, LLMCallFailed

    assert issubclass(LLMCallFailed, BackendError)
    with pytest.raises(BackendError):
        raise LLMCallFailed("gateway 400")


def test_llm_trace_langfuse_mirror_failure_swallowed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_trace, "_now_iso", lambda **_: "2026-01-01T00:00:00Z")

    import hyperloom.orchestrator.trace.langfuse_emitter as le

    def _boom(_dir):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(le, "get_emitter", _boom)

    record = llm_trace.LLMCallRecord(session_id="s3", component="forge")
    # Must not raise even though the Langfuse mirror blows up.
    llm_trace.append_llm_call(session_dir=tmp_path, record=record)
