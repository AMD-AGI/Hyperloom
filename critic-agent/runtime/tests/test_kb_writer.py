"""End-to-end tests for :class:`runtime.kb_writer.KBWriter`.

We use :class:`runtime.in_memory_kb_client.InMemoryKBClient` so the tests
exercise the same surface a production HTTP client would, but without
network IO. Dead-letter and session-memory paths use the temp fixtures.
"""

from __future__ import annotations

import json
import os

import pytest

from runtime.dead_letter import DeadLetter
from runtime.in_memory_kb_client import InMemoryKBClient
from runtime.kb_writer import KBWriter, WriteContext
from runtime.session_memory import SessionMemory


@pytest.fixture()
def packet_context():
    return {
        "framework": "sglang",
        "model": "deepseek-r1-0528-fp8",
        "model_family": "deepseek",
        "workload": "decode",
        "precision": "fp8",
    }


@pytest.fixture()
def writer(tmp_path):
    kb = InMemoryKBClient()
    sm = SessionMemory(root=tmp_path / "sm")
    dlq = DeadLetter(root=tmp_path / "dlq")
    return KBWriter(kb, session_memory=sm, dead_letter=dlq), kb, sm, dlq


def test_write_verdict_skipped_for_advise(writer, packet_context):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={
            "verdict": "advise",
            "reasoning": "small concurrency mismatch",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "skipped"
    assert kb.all_rows() == []


def test_write_verdict_reject_creates_pitfall(writer, packet_context):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
            "kb_evidence": [],
            "confidence": "high",
            "risks": [{"type": "active_path_unproven", "severity": "blocker"}],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1", review_id="rev_1", topic="active dispatch path unproven"),
    )
    assert res.status == "ok"
    rows = kb.all_rows()
    assert len(rows) == 1
    assert rows[0]["kind"] == "pitfall"
    assert rows[0]["importance"] <= 0.84


def test_write_verdict_skipped_when_disabled(monkeypatch, writer, packet_context):
    w, kb, _, _ = writer
    monkeypatch.setenv("KB_WRITE_ENABLED", "false")
    w.write_enabled = False  # honour env at runtime
    res = w.write_verdict(
        verdict={"verdict": "reject", "reasoning": "x"},
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "disabled"


def test_write_verdict_dead_letters_on_validation_error(writer, packet_context):
    w, kb, sm, dlq = writer
    kb.simulate_failure(endpoint="upsert", times=1, error={"code": 422})
    res = w.write_verdict(
        verdict={
            "verdict": "reject",
            "reasoning": "active dispatch path unproven for this kernel",
            "packet_evidence": ["benchmark.after.gain_pct"],
        },
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1", review_id="rev"),
    )
    assert res.status == "dead_lettered"
    files = dlq.files()
    assert len(files) == 1
    line = files[0].read_text("utf-8").splitlines()[0]
    record = json.loads(line)
    assert record["endpoint"] == "upsert"
    assert "session_id" in record["context"]


def test_write_kb_drafts_batch_inserts_and_filters_unknown_categories(writer, packet_context):
    w, kb, _, _ = writer
    drafts = [
        {
            "category": "kernel_optimization",
            "action": "Patch fused attention kernel for Qwen3-14B on MI355X.",
            "lesson": "Active dispatch path must be updated jointly.",
            "tags": ["attention"],
            "result": {"status": "KEEP", "gain_pct": 4.2},
            "confidence": 0.9,
        },
        {
            "category": "definitely_not_a_real_category",
            "action": "x",
        },
    ]
    res = w.write_kb_drafts(
        kb_drafts=drafts,
        packet_context=packet_context,
        ctx=WriteContext(session_id="s1", review_id="rev_2"),
    )
    assert res.status == "ok"
    assert len(kb.all_rows()) == 1
    assert kb.all_rows()[0]["kind"] == "technique"
    rejected = res.detail["rejected"]
    assert rejected and rejected[0]["draft"]["category"] == "definitely_not_a_real_category"


def test_list_priors_uses_session_memory_cache(writer, packet_context):
    w, kb, sm, _ = writer
    kb.upsert({
        "scope": {
            "org": "hyperloom",
            **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
        },
        "kind": "pitfall",
        "slug": "active-path-unproven-pitfall",
        "importance": 0.5,
        "metadata": {"topic": "active path"},
    })
    scope = {
        "org": "hyperloom",
        **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
    }
    ctx = WriteContext(session_id="s_cache")
    first = w.list_priors(scope=scope, kind="pitfall", topic="active path", ctx=ctx)
    assert first["cache"] == "miss"
    assert first["priors"]
    second = w.list_priors(scope=scope, kind="pitfall", topic="active path", ctx=ctx)
    assert second["cache"] == "hit"
    assert second["priors"] == first["priors"]


def test_add_contradiction_writes_edge(writer, packet_context):
    w, kb, _, _ = writer
    a = kb.upsert({
        "scope": {
            "org": "hyperloom",
            **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
        },
        "kind": "pitfall", "slug": "abcdef-1", "importance": 0.5,
    })["row"]["id"]
    b = kb.upsert({
        "scope": {
            "org": "hyperloom",
            **{k: packet_context[k] for k in ("framework", "model", "model_family", "workload", "precision")},
        },
        "kind": "pitfall", "slug": "abcdef-2", "importance": 0.5,
    })["row"]["id"]
    res = w.add_contradiction(
        new_id=a, old_ids=[b], ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "ok"
    rows = {r["id"]: r for r in kb.all_rows()}
    assert b in rows[a]["edges"]["contradicts"]
    assert a in rows[b]["edges"]["contradicts"]


def test_write_verdict_with_missing_critical_scope_skipped(writer):
    w, kb, _, _ = writer
    res = w.write_verdict(
        verdict={"verdict": "reject", "reasoning": "x"},
        packet_context={"framework": "sglang"},  # model missing
        ctx=WriteContext(session_id="s1"),
    )
    assert res.status == "skipped"
    assert "scope_construction_failed" in res.detail.get("reason", "")
