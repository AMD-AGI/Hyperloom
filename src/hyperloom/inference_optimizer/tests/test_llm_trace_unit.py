# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
            "resume_downgraded": True,
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
    assert row["resume_downgraded"] is True


def test_llm_call_record_rejects_unknown_component(tmp_path: Path):
    with pytest.raises(llm_trace.LLMTraceRowError):
        llm_trace.append_llm_call(
            session_dir=tmp_path,
            record=llm_trace.LLMCallRecord(session_id="s1", component="removed_backend"),
        )
