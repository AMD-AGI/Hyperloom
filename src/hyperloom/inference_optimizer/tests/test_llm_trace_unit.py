# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
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


def test_llm_trace_langfuse_mirror_failure_swallowed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm_trace, "_now_iso", lambda **_: "2026-01-01T00:00:00Z")

    import hyperloom.orchestrator.trace.langfuse_emitter as le

    def _boom(_dir):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(le, "get_emitter", _boom)

    record = llm_trace.LLMCallRecord(session_id="s3", component="forge")
    # Must not raise even though the Langfuse mirror blows up.
    llm_trace.append_llm_call(session_dir=tmp_path, record=record)
